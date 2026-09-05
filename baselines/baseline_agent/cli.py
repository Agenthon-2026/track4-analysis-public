"""`analyze` CLI entry point for the minimal Track 4 baseline.

Usage — exactly the argv the scoring harness issues (see SUBMISSION_CLI.md)::

    analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json

The leading ``analyze`` is the **container command**: the harness runs
``docker run <image> analyze --task ... --corpus ... --out ...``, so the verb
arrives as the first argument. This CLI therefore accepts and validates it. It
is optional when you run the baseline by hand::

    python -m baseline_agent.cli \
      --task   units/t4-EXAMPLE-eps-beat/task.json \
      --corpus units/t4-EXAMPLE-eps-beat/corpus \
      --out    /tmp/answer.json

Pure standard library — no network, no model weights — so it runs even under a
local ``--network=none`` smoke run out of the box. At official scoring time,
units run on a restricted network (no open internet; egress only via the
organizer's audited proxy to the organizer-hosted ``$MODEL_ENDPOINT`` and
nothing else -- vendor model APIs are refused, policy 2026-08-04); this minimal
baseline simply never uses it. A stronger LLM tier (see baselines/README.md,
Baseline 3) may call ``$MODEL_ENDPOINT`` through the proxy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .formatter import build_answer, build_entity_prediction, filter_embargoed
from .indexer import IndexedDoc, build_index
from .reader import predict_entity
from .retriever import retrieve


def _target_type(task: dict) -> str | None:
    """The unit's target type, read from BOTH published shapes of ``task.json``.

    ``SUBMISSION_CLI.md`` (the T4 row of the interface table, and contract invariant 7)
    publishes it as a **top-level** ``target_type``; every ``task.json`` in this repo —
    the exemplar and all ten practice units — declares it nested at ``target.type``.
    Reading only one shape silently mislabels the other: a task in exactly the documented
    top-level shape used to fall through to ``"classification"``, which on a regression or
    ranking unit is ``t4.target_type_mismatch`` -> ``SCHEMA_INVALID_OUTPUT``, ``W = -0.27``.

    The documented field wins where both are present. ``None`` means the task declares
    neither, and the caller then omits ``target_type`` from the answer instead of guessing.
    """
    nested = task.get("target")
    return task.get("target_type") or (
        nested.get("type") if isinstance(nested, dict) else None
    )


def _newest_eligible(docs: list[IndexedDoc], cutoff: str) -> IndexedDoc | None:
    """The newest document whose ``doc_date`` is <= ``cutoff``, or ``None`` if there is none.

    ``None`` is the honest answer, not a reason to substitute some other document. Measured
    against the scorer, every alternative costs the same and one of them is a rule violation:
    citing a post-cutoff document under ``strict = true`` embargo is ``t4.citation_post_cutoff``
    (score -0.27), and citing a document the corpus does not contain is ``t4.citation_unresolved``
    (score -0.27) — while an undated corpus document cannot be cited at all, because the scorer
    refuses to load a unit containing one (``T4OrganizerFault``: "carries no doc_date"). The
    previous fallback returned ``docs[0]`` — simply the first file in sorted order, whose date
    is unrelated to the cutoff — and was measured citing a post-cutoff document.
    """
    dated = [d for d in docs if d.doc_date is not None and d.doc_date <= cutoff]
    if not dated:
        return None
    return max(dated, key=lambda d: d.doc_date)


def run(task_path: Path, corpus_dir: Path, out_path: Path) -> dict:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    cutoff = task["cutoff_date"]
    target = task.get("target") or {}
    labels = target.get("labels")
    docs = build_index(corpus_dir)
    doc_dates = {d.doc_id: d.doc_date for d in docs}

    entity_predictions = []
    for entity in task.get("entities", []):
        eid = entity.get("entity_id", "")
        query = " ".join(
            str(entity.get(k, ""))
            for k in ("name", "entity_id", "sector")
        ) + " earnings per share diluted EPS revenue"
        hit = retrieve(query, docs, cutoff)
        claims = []
        span_text = ""
        if hit is not None:
            span_text = hit.text
            claims.append(
                {
                    "doc_id": hit.doc_id,
                    "span_start": hit.span_start,
                    "span_end": hit.span_end,
                    "claim": f"Evidence for {eid}: {hit.text[:160]}",
                }
            )
        pred = predict_entity(entity, span_text, labels=labels)
        # Embargo filter BEFORE the >=1-claim fallback: a claim dropped as post-cutoff
        # must be replaced by the fallback's eligible citation, never leave the entity
        # with an empty claims list (the schema requires >=1 claim per entity).
        claims = filter_embargoed(claims, cutoff, doc_dates)
        if not claims:
            # Ensure schema validity (>=1 claim) even with no retrieval hit, and make
            # that claim survivable. Measured on the scorer, all three of the old
            # fallback's failure modes cost the full W = -0.27: citing docs[0] when it
            # post-dates the cutoff is t4.citation_post_cutoff, and even a correctly
            # embargo-eligible document cited at (0, 1) hands the NLI judge a one-character
            # premise, which is t4.evidence_unsupported. The same document cited at
            # (0, 240) scored 0.67. Hence: newest eligible document, 240 characters.
            fallback = _newest_eligible(docs, cutoff)
            claims.append(
                {
                    "doc_id": fallback.doc_id if fallback else "UNKNOWN",
                    "span_start": 0,
                    "span_end": min(240, len(fallback.text)) if fallback else 1,
                    "claim": f"No strong evidence retrieved for {eid}; default neutral prediction.",
                }
            )
        entity_predictions.append(
            build_entity_prediction(
                eid,
                pred["label"],
                pred["point_forecast"],
                pred["lo"],
                pred["hi"],
                claims,
                cutoff=cutoff,
                doc_dates=doc_dates,
            )
        )

    answer = build_answer(
        task_id=task.get("task_id", task_path.stem),
        entity_predictions=entity_predictions,
        target_type=_target_type(task),
        trace=(
            f"Indexed {len(docs)} corpus docs; embargo cutoff {cutoff}. "
            f"Lexical retrieval + rule-based EPS classifier. "
            f"Predicted {len(entity_predictions)} entit(y/ies)."
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(answer, indent=2, ensure_ascii=False), encoding="utf-8")
    return answer


VERB = "analyze"


def add_verb_argument(parser: argparse.ArgumentParser) -> None:
    """Accept the track verb as an optional leading positional.

    Required for the image to be runnable at all: the harness passes the verb as the container
    command, so argv[0] is ``analyze``. Without this argparse rejects it as an unrecognized
    positional and the image exits 2 on every unit — before reading a single input file, and
    identically for every submission built from this baseline.

    Optional so that running the module by hand (``python -m baseline_agent.cli --task ...``)
    keeps working unchanged.
    """
    parser.add_argument(
        "verb", nargs="?", default=VERB, choices=[VERB],
        help=f"track verb; the harness passes {VERB!r} as the container command",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Track 4 minimal RAG baseline (analyze).")
    add_verb_argument(parser)
    parser.add_argument("--task", required=True, type=Path, help="Path to task.json")
    parser.add_argument("--corpus", required=True, type=Path, help="Path to corpus/ directory")
    parser.add_argument("--out", required=True, type=Path, help="Output answer.json path")
    args = parser.parse_args()
    run(args.task, args.corpus, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
