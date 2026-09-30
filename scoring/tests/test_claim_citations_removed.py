"""Scorer 5.2.2: the claim-level `citations` list is removed.

A claim cites one span of one document through its own `doc_id`, `span_start` and `span_end`
(the schema requires all three). Before 5.2.2 a claim could also carry a `citations` list, and
the scorer then read that list INSTEAD of the claim's own span, so one claim could anchor
figures from two documents: on a public unit a two-document claim went from false to neutral.

From 5.2.2 a claim that carries a `citations` key is false, as a malformed claim is: it is never
put to the judge, the list is not read, and the claim's own span is checked like every other
claim's. The unit is still scored; the key never refuses it.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from qfbench2_track_analysis.alignment import EntityRoster, align_predictions
from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import UnitOutcome, score_unit

from .synthetic import (
    PRE_CUTOFF_DOC,
    SUPPORTING_TEXT,
    StubJudge,
    _doc,
    answer_for,
    build_unit,
)

REVENUE_DOC = "SYNTHDOC_REVENUE_20260120"
MARGIN_DOC = "SYNTHDOC_MARGIN_20260125"
REVENUE_TEXT = "Synthetic Issuer A reported revenue of 5.2 billion for the quarter."
MARGIN_TEXT = (
    "Synthetic Issuer A reported a gross margin of 46 percent for the quarter."
)
#: Two figures, one from each document: no single span carries both.
#: The removed claim-level key (written out, so this file also runs against a scorer before
#: 5.2.2 and shows what the removal changes).
NESTED_CITATIONS_KEY = "citations"
TWO_DOC_CLAIM = "Revenue was 5.2 billion and gross margin was 46 percent."

DOCS = {
    PRE_CUTOFF_DOC: _doc(PRE_CUTOFF_DOC, "2026-02-01", SUPPORTING_TEXT),
    REVENUE_DOC: _doc(REVENUE_DOC, "2026-01-20", REVENUE_TEXT),
    MARGIN_DOC: _doc(MARGIN_DOC, "2026-01-25", MARGIN_TEXT),
}


def _cite(doc_id: str, text: str) -> dict[str, Any]:
    return {"doc_id": doc_id, "span_start": 0, "span_end": len(text)}


def _two_doc_answer(*, nested: bool) -> dict[str, Any]:
    """SYN-A's one claim cites the revenue document; with `nested`, it also carries the old
    `citations` list naming both documents."""
    answer = answer_for()
    answer.pop("target_type", None)
    claim: dict[str, Any] = {**_cite(REVENUE_DOC, REVENUE_TEXT), "claim": TWO_DOC_CLAIM}
    if nested:
        claim[NESTED_CITATIONS_KEY] = [
            _cite(REVENUE_DOC, REVENUE_TEXT),
            _cite(MARGIN_DOC, MARGIN_TEXT),
        ]
    answer["entity_predictions"][0]["claims"] = [claim]
    return answer


def _score(tmp_path: pathlib.Path, answer: dict[str, Any]) -> UnitOutcome:
    unit = build_unit(tmp_path, with_outcome=True, docs=DOCS)
    out = tmp_path / "res"
    out.mkdir()
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out},
        judge=StubJudge((SUPPORTING_TEXT,)),
        judge_provenance=provenance,
    )


def test_a_two_document_claim_no_longer_gains_from_a_citations_list(
    tmp_path: pathlib.Path,
) -> None:
    """Before 5.2.2 the list anchored both figures and the claim was neutral, so the unit
    scored higher than with the one-span claim. Now both are false and score the same."""
    flat = _score(tmp_path / "flat", _two_doc_answer(nested=False))
    nested = _score(tmp_path / "nested", _two_doc_answer(nested=True))
    assert flat.state == nested.state == "participant_success"
    # The one-span claim is false because a figure is in no span it cites ...
    assert flat.diagnostics["unanchored_claim_count"] == 1
    assert flat.diagnostics["false_claim_count"] == 1
    # ... and the claim with the list is false because it carries the list.
    assert nested.diagnostics["malformed_claim_count"] == 1
    assert nested.diagnostics["false_claim_count"] == 1
    assert nested.score == pytest.approx(flat.score)


@pytest.mark.parametrize(
    "value",
    [
        [{"doc_id": PRE_CUTOFF_DOC, "span_start": 0, "span_end": len(SUPPORTING_TEXT)}],
        [],
        None,
        "not a list",
    ],
    ids=["one_citation", "empty", "null", "string"],
)
def test_a_claim_carrying_citations_is_false_and_the_unit_is_still_scored(
    tmp_path: pathlib.Path, value: object
) -> None:
    """Whatever the key holds, the claim is false (malformed) and never reaches the judge; an
    empty list no longer refuses the whole unit."""
    answer = answer_for()
    answer.pop("target_type", None)
    answer["entity_predictions"][0]["claims"][0][NESTED_CITATIONS_KEY] = value
    outcome = _score(tmp_path, answer)
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["malformed_claim_count"] == 1
    assert outcome.diagnostics["false_claim_count"] == 1

    control = answer_for()
    control.pop("target_type", None)
    clean = _score(tmp_path / "control", control)
    assert clean.diagnostics["false_claim_count"] == 0
    assert outcome.score < clean.score


def test_only_the_claims_own_span_is_read() -> None:
    """The list is not read: the aligned claim holds exactly its own citation, which the
    embargo and entity checks then see like any other claim's."""
    answer = _two_doc_answer(nested=True)
    aligned = align_predictions(
        answer,
        EntityRoster(entity_ids=("SYN-A", "SYN-B", "SYN-C")),
        target_type="classification",
        interval_level=0.90,
    )
    [claim] = aligned.claims_by_entity[0]
    assert claim.malformed
    assert [c["doc_id"] for c in claim.citations] == [REVENUE_DOC]
    assert MARGIN_DOC not in {c["doc_id"] for c in aligned.all_citations()}
