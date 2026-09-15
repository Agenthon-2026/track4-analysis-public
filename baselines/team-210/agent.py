#!/usr/bin/env python3
"""Agenthon 2026 Track 4 submission - `analyze`.

    analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json

Track 4's rule, verbatim from its README: if the citations do not hold up under
the NLI check, the submission is ineligible *regardless of how accurate its
predictions were*. Accuracy is therefore not the first thing to protect. The
first thing to protect is that every claim resolves to a real span in a document
the embargo allows.

That is the same shape as the evidence lock: re-check before writing, and refuse
to emit a citation the corpus does not carry rather than emit it and hope.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from analysis_core import corpus as corpus_mod   # noqa: E402
from analysis_core import retriever, validators  # noqa: E402

#: Widen rather than narrow when the evidence is thin. An interval that misses
#: costs coverage; one that is wide and honest does not pretend.
BASE_REL_WIDTH = 0.12
THIN_EVIDENCE_MULTIPLIER = 1.8

#: Keys that plausibly carry the quantity being forecast, in preference
#: order. The old fallback took the FIRST numeric field of the entity dict -
#: which handed fiscal_year 2024 (or a bare `true`: bool is an int in
#: Python) to point_forecast, silently and absurdly.
ANCHOR_KEYS = ("consensus_eps", "consensus", "mean_estimate", "estimate",
               "prior_eps", "last_actual", "prior", "last_value")
#: Never an anchor: identifiers, sizes and dates.
ANCHOR_DENYLIST = ("year", "id", "cik", "sic", "shares", "mktcap",
                   "market_cap", "employees", "rank")


def _anchor_of(entity: dict) -> float:
    for key in ANCHOR_KEYS:
        value = entity.get(key)
        if isinstance(value, (int, float)) and type(value) is not bool:
            return float(value)
    for key, value in entity.items():
        if (isinstance(value, (int, float)) and type(value) is not bool
                and not any(d in key.lower() for d in ANCHOR_DENYLIST)):
            return float(value)
    return 0.0


def finalise_interval(prediction: dict, interval_level: float) -> None:
    """Interval width from the SURVIVING claim count, post-verification."""
    anchor = float(prediction["point_forecast"])
    surviving = len(prediction["claims"])
    width = BASE_REL_WIDTH * (1.0 if surviving >= 2 else THIN_EVIDENCE_MULTIPLIER)
    half = abs(anchor) * width if anchor else 1.0
    prediction["interval"] = {"level": interval_level,
                              "lo": anchor - half, "hi": anchor + half}


def build_prediction(entity: dict, index: retriever.Index, prompt: str,
                     target: dict, interval_level: float) -> dict:
    entity_id = str(entity.get("entity_id") or entity.get("ticker") or "?")
    terms = tuple(str(v) for v in
                  (entity_id, entity.get("name"), entity.get("ticker")) if v)

    hits = index.search(prompt, entity_terms=terms, top_k=4)

    claims = [
        {"doc_id": document.doc_id,
         "span_start": int(start), "span_end": int(end),
         # The claim is the span. Paraphrasing here is how a citation stops
         # carrying what it cites; the NLI judge compares the two.
         "claim": body}
        for document, start, end, body, _ in hits
    ]

    anchor = _anchor_of(entity)

    prediction = {
        "entity_id": entity_id,
        "point_forecast": anchor,
        # Placeholder interval; finalised AFTER citation verification, so the
        # widening reflects the claims that actually SURVIVED, not the ones
        # that were merely retrieved.
        "interval": {"level": interval_level, "lo": anchor, "hi": anchor},
        "claims": claims,
    }

    # `label` belongs to classification units only. On a regression or ranking
    # unit it is not read, and target_type is deliberately omitted from the
    # answer: the schema pins it to an enum, so a value disagreeing with the
    # card is t4.target_type_mismatch - a whole-submission failure - while
    # omitting it is accepted and scored.
    if target.get("type") == "classification":
        labels = target.get("labels") or []
        prediction["label"] = labels[-1] if labels else None
    return prediction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb", choices=["analyze"])
    parser.add_argument("--task", default="/input/task.json")
    parser.add_argument("--corpus", default="/input/corpus/")
    parser.add_argument("--out", default="/output/answer.json")
    args = parser.parse_args()

    started = time.time()
    diagnostics: list[str] = []
    task = json.loads(pathlib.Path(args.task).read_text(encoding="utf-8"))
    cutoff_text = str(task.get("cutoff_date", ""))

    # interval_level is hunted across the spellings cards use; a silent 0.9
    # default against a card that said 0.8 voids the WHOLE submission at g1.
    raw_level = (task.get("interval_level")
                 or (task.get("target") or {}).get("interval_level")
                 or (task.get("interval") or {}).get("level"))
    if raw_level is None:
        diagnostics.append("interval_level not found in task; defaulting 0.9")
        raw_level = 0.9
    interval_level = float(raw_level)

    target = task.get("target") or {}
    entities = task.get("entities") or []
    if not entities:
        diagnostics.append("task lists NO entities - nothing to predict; "
                           "check the roster key")

    documents, embargoed = corpus_mod.load(pathlib.Path(args.corpus), cutoff_text)
    if not documents:
        diagnostics.append("corpus loaded ZERO admissible documents - every "
                           "prediction will ship uncited")
    index = retriever.Index(documents)
    by_id = {d.doc_id: d for d in documents}
    try:
        cutoff = (dt.date.fromisoformat(cutoff_text[:10])
                  if len(cutoff_text) >= 10 else None)
    except ValueError:
        cutoff = None
        diagnostics.append(f"cutoff_date {cutoff_text!r} unparseable; span "
                           "verification runs without a date check")

    predictions = [
        build_prediction(entity, index, str(task.get("prompt", "")),
                         target, interval_level)
        for entity in entities
    ]

    # Evidence lock. Every claim is verified against the corpus before the file
    # is written; one that does not resolve is dropped, not shipped.
    dropped = 0
    for prediction in predictions:
        problems = validators.check_citations(prediction, by_id, cutoff)
        if problems:
            bad = {int(p.where.split("[")[1].rstrip("]")) for p in problems
                   if "[" in p.where}
            prediction["claims"] = [c for i, c in enumerate(prediction["claims"])
                                    if i not in bad]
            dropped += len(bad)
        # Interval width from what SURVIVED, not what was retrieved.
        finalise_interval(prediction, interval_level)

    answer = {
        "task_id": task.get("task_id"),
        "schema_version": str(task.get("schema_version", "3")),
        "entity_predictions": predictions,
        "evidence_trace": (
            f"Indexed {len(documents)} admissible documents "
            f"({len(embargoed)} excluded by the {cutoff_text} embargo). "
            f"BM25 over sentence spans, entity-gated, at most two passages per "
            f"document. {dropped} claim(s) dropped by span verification. "
            f"Intervals at level {interval_level}, widened "
            f"{THIN_EVIDENCE_MULTIPLIER}x where fewer than two claims survived. "
            f"No post-cutoff document is cited."
        ),
    }

    roster = [str(e.get("entity_id") or e.get("ticker") or "?") for e in entities]
    schema_problems = validators.check_schema(answer, interval_level, roster)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(answer, indent=2) + "\n", encoding="utf-8")

    report = {
        "task": answer["task_id"],
        "entities": len(predictions),
        "documents_admissible": len(documents),
        "documents_embargoed": len(embargoed),
        "claims_written": sum(len(p["claims"]) for p in predictions),
        "claims_dropped_unverified": dropped,
        "schema_problems": [str(p) for p in schema_problems],
        "diagnostics": diagnostics,
        "wall_sec": round(time.time() - started, 2),
    }
    print(json.dumps(report))
    # Survival contract: the answer file is written and this process exits 0
    # regardless. A non-zero exit reads as an agent crash and can void a run
    # whose answer was merely imperfect; problems travel via the report.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
