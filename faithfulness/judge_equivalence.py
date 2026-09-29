"""Judge equivalence harness: local vs served backend, claim by claim.

Runs every claim/span pair from one or more (unit, answer) pairs through BOTH
judge backends and reports, per ensemble member and for the ensemble mean:

- per-claim score deltas and the maximum absolute delta,
- the count of **tau crossings** — claims whose supported/unsupported status
  (score > tau_citation, 0.5) differs between backends. This is the number
  that matters: a delta that flips no claim past tau changes no unit score.

The report is written as JSON and summarized on stdout. It is the evidence
basis for the serving-pin tolerance; the harness itself asserts
nothing — it measures.

Usage::

    python -m faithfulness.judge_equivalence \
        --pair units/t4-EXAMPLE-eps-beat:answers/t4-EXAMPLE-eps-beat.answer.json \
        [--pair UNIT_DIR:ANSWER_JSON ...] \
        --served-url http://JUDGE_HOST:8080 [--served-token TOK] \
        --out equivalence_report.json

The local backend needs the pinned weights cached locally (first use of the
real judge downloads them where the hub is reachable); the served backend is
any organizer-side deployment answering the ``/entail`` contract that
``ServedNLIJudge`` in ``faithfulness/judge.py`` speaks. That deployment is not
published here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from faithfulness.judge import (
    NLI_MODEL_IDS,
    TAU_CITATION,
    DeBERTaNLIJudge,
    ServedNLIJudge,
)

_MANIFEST_NAME = "manifest.json"


def _doc_text(doc: dict[str, Any]) -> str:
    """Concatenate spans (or use a flat ``text`` field) — mirrors the scorer."""
    text = doc.get("text")
    if isinstance(text, str):
        return text
    spans = doc.get("spans")
    if isinstance(spans, list):
        return " ".join(sp.get("text", "") for sp in spans if isinstance(sp, dict))
    return ""


def load_corpus_texts(unit_dir: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for path in sorted((unit_dir / "corpus").glob("*.json")):
        if path.name == _MANIFEST_NAME:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        texts[doc.get("doc_id", path.stem)] = _doc_text(doc)
    return texts


def collect_pairs(unit_dir: Path, answer_path: Path) -> list[dict[str, str]]:
    """Yield one {premise, hypothesis, where} record per claim in the answer."""
    texts = load_corpus_texts(unit_dir)
    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    records: list[dict[str, str]] = []
    for entity in answer.get("entity_predictions", []):
        for i, claim in enumerate(entity.get("claims", [])):
            doc_text = texts.get(claim.get("doc_id", ""), "")
            premise = doc_text[claim.get("span_start", 0) : claim.get("span_end", 0)]
            records.append(
                {
                    "premise": premise,
                    "hypothesis": claim.get("claim", ""),
                    "where": (f"{unit_dir.name}/{entity.get('entity_id', '?')}#c{i}"),
                }
            )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        metavar="UNIT_DIR:ANSWER_JSON",
        help="Unit directory and answer file, colon-separated. Repeatable.",
    )
    parser.add_argument("--served-url", required=True)
    parser.add_argument("--served-token", default=None)
    parser.add_argument("--cache-dir", default=None, help="Local weights cache dir.")
    parser.add_argument("--out", type=Path, default=Path("equivalence_report.json"))
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=4,
        help=(
            "torch.set_num_threads for the LOCAL backend. Must match the "
            "server's T4_JUDGE_TORCH_THREADS: CPU matmul reduction order "
            "depends on thread count, and a mismatch alone produces ~1e-4 "
            "score deltas (measured 2026-07-31). Matched threads on the same "
            "fp32 code path reproduce bit-exact (delta 0.0)."
        ),
    )
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(args.torch_threads)

    local_kwargs = {"cache_dir": args.cache_dir} if args.cache_dir else {}
    backends: dict[str, dict[str, DeBERTaNLIJudge | ServedNLIJudge]] = {
        "local": {
            mid: DeBERTaNLIJudge(model_id=mid, **local_kwargs) for mid in NLI_MODEL_IDS
        },
        "served": {
            mid: ServedNLIJudge(
                model_id=mid, url=args.served_url.rstrip("/"), token=args.served_token
            )
            for mid in NLI_MODEL_IDS
        },
    }

    records: list[dict[str, str]] = []
    for spec in args.pair:
        unit_raw, _, answer_raw = spec.partition(":")
        if not answer_raw:
            parser.error(f"--pair {spec!r} is not UNIT_DIR:ANSWER_JSON")
        records.extend(collect_pairs(Path(unit_raw), Path(answer_raw)))
    if not records:
        print("no claims found in the given pairs", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {"where": record["where"], "models": {}}
        ensemble: dict[str, float] = {}
        for backend_name, judges in backends.items():
            scores = {
                mid: judge.entail(record["premise"], record["hypothesis"])
                for mid, judge in judges.items()
            }
            ensemble[backend_name] = sum(scores.values()) / len(scores)
            for mid, score in scores.items():
                row["models"].setdefault(mid, {})[backend_name] = score
        row["ensemble"] = {
            **ensemble,
            "delta": abs(ensemble["local"] - ensemble["served"]),
            "tau_crossing": (ensemble["local"] > TAU_CITATION)
            != (ensemble["served"] > TAU_CITATION),
        }
        rows.append(row)

    deltas = [r["ensemble"]["delta"] for r in rows]
    crossings = [r for r in rows if r["ensemble"]["tau_crossing"]]
    per_model_max = {
        mid: max(
            abs(r["models"][mid]["local"] - r["models"][mid]["served"]) for r in rows
        )
        for mid in NLI_MODEL_IDS
    }
    summary = {
        "claims": len(rows),
        "max_abs_ensemble_delta": max(deltas),
        "mean_abs_ensemble_delta": sum(deltas) / len(deltas),
        "exact_matches": sum(1 for d in deltas if d == 0.0),
        "tau_crossings": len(crossings),
        "tau_crossing_claims": [r["where"] for r in crossings],
        "per_model_max_abs_delta": per_model_max,
        "tau_citation": TAU_CITATION,
        "model_ids": NLI_MODEL_IDS,
    }

    args.out.write_text(
        json.dumps({"summary": summary, "claims": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"\nfull report: {args.out}")
    print(
        "verdict basis: "
        + (
            "no tau crossings — no unit score can differ between backends"
            if not crossings
            else f"{len(crossings)} tau crossing(s) — backends CAN change scores"
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
