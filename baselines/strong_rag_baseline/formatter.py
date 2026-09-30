"""Assemble and self-check the final answer JSON.

The self-check enforces the mistakes that are cheap to catch locally and fatal
at scoring time: claims whose spans do not resolve in the corpus, missing
interval bounds, and a wrong top-level shape (``notes`` must be an object).
It is a last-line assertion — grounding already happened in the agent.
"""
from __future__ import annotations

from .agent import EntityResult
from .indexer import IndexedCorpus


def build_answer(
    task: dict, results: list[EntityResult], corpus: IndexedCorpus
) -> dict:
    total_dropped = sum(r.dropped_claims for r in results)
    total_claims = sum(len(r.prediction["claims"]) for r in results)
    fallback_ids = [r.prediction["entity_id"] for r in results if r.fallback]
    fallback_claim_ids = [
        r.prediction["entity_id"] for r in results if r.fallback_claim
    ]
    interval_fallback_ids = [
        r.prediction["entity_id"] for r in results if r.interval_fallback
    ]
    budget_denied = any(r.budget_denied for r in results)
    answer = {
        "task_id": task.get("task_id", ""),
        "schema_version": task.get("schema_version", "3"),
        "entity_predictions": [r.prediction for r in results],
        "evidence_trace": (
            f"strong_rag_baseline: BM25 span-chunk retrieval + $MODEL_ENDPOINT. "
            f"{total_claims} grounded claims kept, {total_dropped} ungroundable "
            f"evidence items dropped. All cited spans resolved in the frozen "
            f"corpus; embargo enforced at retrieval time. "
            f"{len(fallback_ids)} fallback rows (model reply or call not usable); "
            f"{len(fallback_claim_ids)} rows whose evidence did not ground cite a "
            f"verbatim corpus passage instead (counted in the claims above)."
        ),
        "notes": {
            "agent": "strong_rag_baseline",
            "retrieval": "bm25-span-chunks",
            "dropped_evidence_items": total_dropped,
            # Rows whose model reply could not be used: placeholder prediction, not a forecast.
            "fallback_entities": fallback_ids,
            # Rows whose model evidence did not ground: one verbatim passage stands in.
            "fallback_claim_entities": fallback_claim_ids,
            # Rows that keep the model's forecast but not its interval (missing, non-finite
            # or lo above hi): a band of point +/- max(|point|/2, 1) is substituted.
            "interval_fallback_entities": interval_fallback_ids,
            # Fallback rows written because the model request allowance was used up
            # (House 403 grant_denied); also listed in fallback_entities.
            "budget_denied_entities": [
                r.prediction["entity_id"] for r in results if r.budget_denied
            ],
            # True when that refusal came before any request in the unit got a reply (only
            # failed calls before it), or before any reply that could be used.
            "budget_refused_before_any_reply": budget_denied
            and all(r.call_failed or r.budget_denied for r in results),
            "budget_refused_before_any_usable_reply": budget_denied
            and all(r.fallback for r in results),
            # Fallback rows written because the model call failed after the client's
            # retries (network error, timeout, server error); also in fallback_entities.
            "call_failed_entities": [
                r.prediction["entity_id"] for r in results if r.call_failed
            ],
            # True when every entity's model call failed: the whole answer is placeholders.
            "every_model_call_failed": bool(results)
            and all(r.call_failed for r in results),
        },
    }
    _assert_valid(answer, corpus)
    return answer


def _assert_valid(answer: dict, corpus: IndexedCorpus) -> None:
    assert isinstance(answer["notes"], dict), "top-level notes must be an object"
    for entity in answer["entity_predictions"]:
        interval = entity.get("interval") or {}
        assert "lo" in interval and "hi" in interval, (
            f"{entity.get('entity_id')}: interval must contain lo and hi"
        )
        for claim in entity.get("claims", []):
            doc_text = corpus.doc_texts.get(claim["doc_id"], "")
            assert 0 <= claim["span_start"] < claim["span_end"] <= len(doc_text), (
                f"{entity.get('entity_id')}: span "
                f"[{claim['span_start']}, {claim['span_end']}) does not resolve "
                f"in {claim['doc_id']!r}"
            )
