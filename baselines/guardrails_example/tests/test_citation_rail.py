"""Unit tests for the citation rail checks (pure stdlib, no network)."""
from __future__ import annotations

import pytest

from baselines.guardrails_example.citation_rail import (
    CorpusDoc,
    check_answer,
    filter_retrieved,
)

CUTOFF = "2024-03-15"
DOC = CorpusDoc(doc_id="d1", text="alpha beta gamma delta", doc_date="2024-02-01")
STALE = CorpusDoc(doc_id="d2", text="post cutoff text", doc_date="2024-05-02")
UNDATED = CorpusDoc(doc_id="d3", text="no date here", doc_date=None)
CORPUS = {d.doc_id: d for d in (DOC, STALE, UNDATED)}


def make_answer(**claim_overrides) -> dict:
    claim = {
        "doc_id": "d1",
        "span_start": 0,
        "span_end": 10,
        "claim": "alpha is first",
    }
    claim.update(claim_overrides)
    return {"entity_predictions": [{"entity_id": "E", "claims": [claim]}]}


def codes(answer: dict) -> list[str]:
    return sorted(f.code for f in check_answer(answer, CORPUS, CUTOFF))


def test_clean_claim_passes():
    assert codes(make_answer()) == []


@pytest.mark.parametrize("value", [[{"doc_id": "d1", "span_start": 0, "span_end": 5}], [], None])
def test_a_claim_carrying_citations_is_flagged(value):
    """Scorer 5.2.2 removed the claim-level `citations` list: the scorer counts such a claim as
    false (malformed). The claim's own span is still checked (clean here)."""
    findings = check_answer(make_answer(citations=value), CORPUS, CUTOFF)
    assert [(f.code, f.claim_index) for f in findings] == [("claim_citations", 0)]
    assert "citations" in findings[0].message and "5.2.2" in findings[0].message


def test_stale_doc_flagged():
    assert codes(make_answer(doc_id="d2")) == ["stale_doc"]


def test_undated_doc_flagged_stale():
    assert codes(make_answer(doc_id="d3")) == ["stale_doc"]


def test_unknown_doc_flagged():
    assert codes(make_answer(doc_id="nope")) == ["unknown_doc"]


def test_missing_fields_flagged():
    answer = {"entity_predictions": [{"entity_id": "E", "claims": [{"doc_id": "d1"}]}]}
    assert codes(answer) == ["missing_field"]


@pytest.mark.parametrize(
    "start,end",
    [(-1, 5), (5, 5), (9, 3)],
)
def test_invalid_span_ranges_flagged(start, end):
    assert codes(make_answer(span_start=start, span_end=end)) == ["bad_span"]


def test_non_integer_offsets_flagged():
    assert codes(make_answer(span_start="0", span_end=4)) == ["bad_span"]


def test_span_past_document_end_flagged():
    assert codes(make_answer(span_end=len(DOC.text) + 1)) == ["bad_span"]


def test_empty_claim_text_flagged():
    assert codes(make_answer(claim="  ")) == ["empty_claim"]


def test_filter_retrieved_splits_on_cutoff():
    usable, stale = filter_retrieved([DOC, STALE, UNDATED], CUTOFF)
    assert [d.doc_id for d in usable] == ["d1"]
    assert sorted(d.doc_id for d in stale) == ["d2", "d3"]
