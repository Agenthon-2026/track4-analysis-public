"""Per-entity orchestration: retrieve → reason → verify spans → assemble claims.

The pipeline is intentionally strict about citations: a claim survives only if
its quote resolves to an exact span in the cited document (or, failing that, if
the quote's source chunk is identifiable so the chunk's own offsets can stand
in). Claims that cannot be grounded are dropped — an ungrounded claim risks the
faithfulness gate, while a dropped one merely loses a little coverage.
"""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass

from .client import ModelBudgetExhausted, ModelCallError, ModelClient
from .indexer import Chunk, IndexedCorpus
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retriever import BM25Index
from .span_finder import find_span

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

#: Query terms appended to every entity query — steer retrieval toward result
#: and outlook language regardless of family.
_QUERY_SUFFIX = "results revenue earnings guidance outlook growth"


#: Characters of the top retrieved excerpt a fallback row quotes: enough for the judge to read
#: a real passage, short enough to stay inside one excerpt (chunks run to 1M characters).
_FALLBACK_QUOTE_CHARS = 200


@dataclass
class EntityResult:
    prediction: dict  # entity_predictions[] element
    dropped_claims: int
    model_raw: str
    fallback: bool = False  # True when the model reply was unusable (see _fallback_prediction)
    fallback_claim: bool = False  # True when no model evidence grounded (see _fallback_claims)
    call_failed: bool = False  # True when the model call itself failed after retries
    interval_fallback: bool = False  # True when the band replaced the model's interval
    budget_denied: bool = False  # True when the House's request allowance was used up


def _parse_model_json(raw: str) -> dict:
    """Extract the first JSON object from the model reply (tolerates fences)."""
    match = _JSON_BLOCK.search(raw)
    if match is None:
        raise ValueError("model reply contains no JSON object")
    return json.loads(match.group(0))


def _entity_query(entity: dict) -> str:
    parts = [
        str(entity.get(key, ""))
        for key in ("name", "entity_id", "sector", "series_id", "description")
    ]
    return " ".join(p for p in parts if p) + " " + _QUERY_SUFFIX


def _ground_claims(
    evidence: list[dict],
    corpus: IndexedCorpus,
    retrieved: list[Chunk],
) -> tuple[list[dict], int]:
    """Map model evidence to exact-span claims; count what had to be dropped."""
    claims: list[dict] = []
    dropped = 0
    retrieved_by_doc: dict[str, list[Chunk]] = {}
    for chunk in retrieved:
        retrieved_by_doc.setdefault(chunk.doc_id, []).append(chunk)

    for item in evidence:
        if not isinstance(item, dict):
            dropped += 1
            continue
        doc_id = item.get("doc_id", "")
        quote = str(item.get("quote", ""))
        claim_text = str(item.get("claim", "")).strip()
        doc_text = corpus.doc_texts.get(doc_id)
        if doc_text is None or not claim_text:
            dropped += 1
            continue

        span = find_span(doc_text, quote)
        if span is None:
            # Fallback: cite the retrieved chunk the quote most plausibly came
            # from (longest token overlap). Chunk offsets are known-good.
            chunk = _best_chunk(quote, retrieved_by_doc.get(doc_id, []))
            if chunk is None:
                dropped += 1
                continue
            span = (chunk.span_start, chunk.span_end)

        claims.append(
            {
                "doc_id": doc_id,
                "span_start": span[0],
                "span_end": span[1],
                "claim": claim_text,
            }
        )
    return claims, dropped


def _best_chunk(quote: str, chunks: list[Chunk]) -> Chunk | None:
    q_tokens = set(quote.lower().split())
    if not q_tokens:
        return None
    best: tuple[int, Chunk] | None = None
    for chunk in chunks:
        overlap = len(q_tokens & set(chunk.text.lower().split()))
        if best is None or overlap > best[0]:
            best = (overlap, chunk)
    return best[1] if best and best[0] > 0 else None


def _finite_number(value: object) -> bool:
    """A number that converts to a finite float (not NaN, not infinite, not too large)."""
    if not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _interval_problem(parsed: dict) -> str | None:
    """Why the model's interval cannot be used as given, or None when it can."""
    interval = parsed.get("interval") or {}
    lo, hi = interval.get("lo"), interval.get("hi")
    if lo is None or hi is None:
        return "interval missing"
    # A NaN, infinite or too-large bound counts as a missing bound: the scorer refuses a
    # non-finite number for the whole submission, and the model's point stays usable.
    if not (_finite_number(lo) and _finite_number(hi)):
        return "interval bound is not a finite number"
    if lo > hi:
        return "interval lo is above hi"
    return None


def _safe_interval(parsed: dict, level: float, point: float | None) -> dict:
    if _interval_problem(parsed) is None:
        interval = parsed["interval"]
        return {"level": level, "lo": float(interval["lo"]), "hi": float(interval["hi"])}
    # Fallback: wide symmetric band around the point forecast (or zero). A missing
    # lo/hi is not a per-entity coverage penalty -- it fails g1_schema and the WHOLE
    # submission is scored t4.schema_invalid at W = -0.27 -- so any sane band beats none.
    center = float(point) if isinstance(point, (int, float)) else 0.0
    half = max(abs(center) * 0.5, 1.0)
    return {"level": level, "lo": center - half, "hi": center + half}


def _verbatim_claim(doc_id: str, text: str, start: int) -> dict:
    """A claim whose text is a prefix of ``text``, which starts at offset ``start`` of the doc."""
    quote = text[:_FALLBACK_QUOTE_CHARS]
    if len(text) > _FALLBACK_QUOTE_CHARS:
        quote = quote.rsplit(" ", 1)[0] or quote
    return {
        "doc_id": doc_id,
        "span_start": start,
        "span_end": start + len(quote),
        "claim": quote,
    }


def _fallback_claims(
    task: dict, corpus: IndexedCorpus, retrieved: list[Chunk]
) -> list[dict]:
    """The >=1-claim fallback, for a row with no grounded claim of its own.

    An empty `claims` array on any row fails `g1_schema` for the WHOLE submission. The claim
    quotes the top retrieved excerpt verbatim at its known-good offsets. With nothing retrieved
    it quotes the start of the newest document dated on or before the cutoff, chosen as
    `baseline_agent` chooses the document it cites in the same case (ties included). With no such document there is nothing
    eligible to cite: `baseline_agent` then cites a doc_id the corpus does not contain, which
    the scorer refuses. This agent leaves `claims` empty instead, which the scorer also refuses
    (`t4.schema_invalid`); a made-up doc_id would fail this agent's own span self-check
    (`formatter._assert_valid`) and end the run.
    """
    if retrieved:
        chunk = retrieved[0]
        return [_verbatim_claim(chunk.doc_id, chunk.text, chunk.span_start)]
    cutoff = task.get("cutoff_date")
    # Same choice as `baseline_agent`'s `_newest_eligible`: corpus files in sorted-path order
    # (`doc_dates` keeps it), the first one with the newest doc_date on or before the cutoff.
    # One difference: a document with no text is skipped, because a zero-length span fails
    # this agent's own span self-check and would end the run.
    eligible = [
        (doc_date, doc_id)
        for doc_id, doc_date in corpus.doc_dates.items()
        if isinstance(doc_date, str)
        and isinstance(cutoff, str)
        and doc_date <= cutoff
        and corpus.doc_texts.get(doc_id, "").strip()
    ]
    if not eligible:
        return []
    _, doc_id = max(eligible, key=lambda item: item[0])
    return [_verbatim_claim(doc_id, corpus.doc_texts[doc_id], 0)]


def _require_finite(value: object, what: str) -> None:
    """A NaN or infinite point forecast makes the reply unusable (the scorer refuses it).

    ``float`` of an integer too large for a float raises OverflowError, also an unusable reply.
    """
    if isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise ValueError(f"model reply carries a non-finite {what}")


def _fallback_prediction(
    task: dict, entity: dict, corpus: IndexedCorpus, retrieved: list[Chunk]
) -> dict:
    """The row written when the model reply for this entity cannot be used.

    Every entity must still get a row: a missing row, a null field or an empty `claims` array
    fails `g1_schema` for the WHOLE submission (`t4.schema_invalid`), so one bad reply would
    otherwise cost every other entity's prediction too. The row follows the `--mock`
    convention: `point_forecast` is the visible placeholder 0.0 (a number invented here is not
    a forecast), the label is the first allowed label, the interval is `_safe_interval`'s
    fallback band, and the claim is `_fallback_claims`: corpus text quoted verbatim, so it
    asserts nothing the corpus does not say.
    """
    target = task.get("target", {})
    labels = target.get("labels") or []
    prediction: dict = {
        "entity_id": entity.get("entity_id", ""),
        "point_forecast": 0.0,
        "interval": _safe_interval({}, task.get("interval_level", 0.90), 0.0),
        "claims": _fallback_claims(task, corpus, retrieved),
    }
    if labels:
        prediction["label"] = labels[0]
    return prediction


def _prediction_from_reply(
    task: dict,
    entity: dict,
    raw: str,
    corpus: IndexedCorpus,
    retrieved: list[Chunk],
) -> tuple[dict, int, bool, str | None]:
    """Parse the model reply into one prediction row; raise if the reply is unusable."""
    parsed = _parse_model_json(raw)

    target = task.get("target", {})
    labels = target.get("labels") or []
    label = parsed.get("label")
    if labels and label not in labels:
        label = labels[0]  # deterministic fallback for off-vocabulary labels

    point = parsed.get("point_forecast")
    _require_finite(point, "point_forecast")
    point_value = float(point) if isinstance(point, (int, float)) else None
    if point_value is None and target.get("type") in ("regression", "ranking"):
        # The scorer refuses the whole submission when this row has no number.
        raise ValueError("model reply carries no numeric point_forecast")
    claims, dropped = _ground_claims(
        parsed.get("evidence") or [], corpus, retrieved
    )
    fallback_claim = not claims
    if fallback_claim:
        claims = _fallback_claims(task, corpus, retrieved)

    prediction: dict = {"entity_id": entity.get("entity_id", "")}
    # A null `label` or `point_forecast` fails the published schema for the whole
    # submission; an absent one does not, so a missing value is left out, never written null.
    if isinstance(label, str):
        prediction["label"] = label
    if point_value is not None:
        prediction["point_forecast"] = point_value
    prediction["interval"] = _safe_interval(
        parsed, task.get("interval_level", 0.90), point_value
    )
    interval_problem = _interval_problem(parsed)
    if not (
        _finite_number(prediction["interval"]["lo"])
        and _finite_number(prediction["interval"]["hi"])
    ):
        # A finite point near the float limit gives a band that overflows.
        raise ValueError("the fallback band around this point_forecast is not finite")
    prediction["claims"] = claims
    # `rank` is never written, even when the reply carries one. It is not scored (ranking
    # quality reads `point_forecast`), and ranks asked for one entity at a time rarely form a
    # full permutation of 1..n: the scorer refuses the whole unit for a partial or
    # non-permutation `rank`, and a fallback row next to ranked rows is always partial.
    return prediction, dropped, fallback_claim, interval_problem


def run_entity(
    task: dict,
    entity: dict,
    index: BM25Index,
    corpus: IndexedCorpus,
    client: ModelClient,
    top_k: int,
) -> EntityResult:
    retrieved = [s.chunk for s in index.search(_entity_query(entity), top_k)]
    entity_id = entity.get("entity_id", "")
    try:
        raw = client.complete(
            SYSTEM_PROMPT, build_user_prompt(task, entity, retrieved)
        )
    except ModelBudgetExhausted as exc:
        # The unit's request allowance is used up: every later call is refused the same way.
        # Not a configuration error, so this entity (and each one after it) gets a fallback
        # row, before or after a reply; `notes` records it.
        print(
            f"strong_rag_baseline: entity {entity_id!r}: {exc}; writing a fallback row",
            file=sys.stderr,
        )
        return EntityResult(
            prediction=_fallback_prediction(task, entity, corpus, retrieved),
            dropped_claims=0,
            model_raw="",
            fallback=True,
            budget_denied=True,
        )
    except ModelCallError as exc:
        # The model call failed after the client's retries (network, timeout, server error).
        # That costs this entity only. A configuration error (no MODEL_ENDPOINT, a rejected
        # token, an unknown URL) is not caught and ends the run. When every entity's call
        # failed, `cli.run` still writes the fallback rows and says so in `notes`.
        print(
            f"strong_rag_baseline: entity {entity_id!r}: model call failed ({exc}); "
            "writing a fallback row",
            file=sys.stderr,
        )
        return EntityResult(
            prediction=_fallback_prediction(task, entity, corpus, retrieved),
            dropped_claims=0,
            model_raw="",
            fallback=True,
            call_failed=True,
        )
    try:
        prediction, dropped, fallback_claim, interval_problem = _prediction_from_reply(
            task, entity, raw, corpus, retrieved
        )
    except (
        ValueError,
        TypeError,
        AttributeError,
        OverflowError,
        RecursionError,
    ) as exc:
        # One unusable reply (no JSON, bad JSON, a field of the wrong type, a non-finite or
        # float-overflowing number, JSON nested too deep to parse) costs this entity only.
        # No re-ask: at temperature 0 with a fixed seed the same prompt returns the same reply.
        print(
            f"strong_rag_baseline: entity {entity_id!r}: model reply "
            f"not usable ({type(exc).__name__}: {str(exc)[:200]}); writing a fallback row",
            file=sys.stderr,
        )
        return EntityResult(
            prediction=_fallback_prediction(task, entity, corpus, retrieved),
            dropped_claims=0,
            model_raw=raw,
            fallback=True,
        )
    if interval_problem is not None:
        # The model's forecast is kept; the band `_safe_interval` substitutes is not its own.
        print(
            f"strong_rag_baseline: entity {entity_id!r}: {interval_problem}; a band of "
            "point +/- max(|point|/2, 1) is substituted",
            file=sys.stderr,
        )
    return EntityResult(
        prediction=prediction,
        dropped_claims=dropped,
        model_raw=raw,
        fallback_claim=fallback_claim,
        interval_fallback=interval_problem is not None,
    )
