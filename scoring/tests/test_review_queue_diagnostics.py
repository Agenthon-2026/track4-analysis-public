"""What the claim check records for the operator's review queue beyond the score (5.2.0).

* an **unanchored claim** -- one whose every figure is absent from the passage it cites -- is
  recorded with its entity and its own text, as before; from 5.2.0 it is also FALSE and costs its
  share of the unit;
* every **false claim** is recorded with its entity, its text and its reasons (`false_claims`).

The retired 5.1.x diagnostic `entities_without_supported_claim` is gone: 5.2.0 asks no
"supported" question, and an entity whose claims the judge neither contradicts nor entails costs
nothing (it is neutral), so there is nothing left for that list to warn about.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import (
    ClaimReport,
    ClaimVerdict,
    UnitOutcome,
    score_unit,
)

from .synthetic import (
    PRE_CUTOFF_DOC,
    HypothesisAwareJudge,
    StubJudge,
    _doc,
    answer_for,
    build_unit,
)

SPAN = "Synthetic Issuer A reported revenue of $5.2 billion, up 8% from the prior year."


def _score(
    tmp_path: pathlib.Path,
    claims_by_entity: dict[str, list[str]],
    judge: object,
) -> UnitOutcome:
    roster = tuple(claims_by_entity)
    unit = build_unit(
        tmp_path,
        entities=roster,
        with_outcome=True,
        docs={PRE_CUTOFF_DOC: _doc(PRE_CUTOFF_DOC, "2026-02-01", SPAN)},
    )
    answer = answer_for(entities=roster)
    for row in answer["entity_predictions"]:
        first = row["claims"][0]
        row["claims"] = [
            {**first, "claim": text, "span_end": len(SPAN)}
            for text in claims_by_entity[row["entity_id"]]
        ]
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out}, judge=judge, judge_provenance=provenance
    )


def test_an_admitted_submission_still_records_its_unanchored_claims(
    tmp_path: pathlib.Path,
) -> None:
    """Six claims for one entity, one unanchored: the soft floor keeps 1 - 1/(1 + min(5, 3)) = 3/4
    of the score (five non-false claims, at most 3 x E = 3 dilute), and the unanchored claim
    is named in the diagnostics with its entity and its text."""
    claims = {
        "SYN-A": [
            "Revenue was $5.9 billion",  # unanchored: no figure of it is in the passage
            "Revenue rose 8% from the prior year",
            "Revenue of $5.2 billion was reported",
            "Growth of 8% was reported",
            "The $5.2 billion figure is quarterly revenue",
            "Revenue grew 8%",
        ]
    }
    outcome = _score(tmp_path, claims, StubJudge((SPAN,)))
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["unanchored_claim_count"] == 1
    assert outcome.diagnostics["unanchored_claims"] == [
        {"entity_id": "SYN-A", "claim": "Revenue was $5.9 billion"}
    ]
    assert outcome.diagnostics["false_claims"] == [
        {
            "entity_id": "SYN-A",
            "claim": "Revenue was $5.9 billion",
            "reasons": ["unanchored"],
        }
    ]
    assert outcome.diagnostics["faithfulness_factor"] == 1 - 1 / (1 + 3)
    assert "entities_without_supported_claim" not in outcome.diagnostics


def test_an_unentailed_but_uncontradicted_claim_is_neutral(
    tmp_path: pathlib.Path,
) -> None:
    """Five entities, one claim each; SYN-E's claim is not entailed. Under 5.1.x it counted
    against the 80% gate; under 5.2.0 it is neutral (the judge does not contradict it), so the
    unit keeps its whole score."""
    claims = {
        "SYN-A": ["Revenue rose 8% from the prior year"],
        "SYN-B": ["Revenue rose 8% from the prior year"],
        "SYN-C": ["Revenue rose 8% from the prior year"],
        "SYN-D": ["Revenue rose 8% from the prior year"],
        "SYN-E": ["The weather was mild"],  # judged, not entailed
    }
    judge = HypothesisAwareJudge(SPAN, required_in_hypothesis="rose")
    outcome = _score(tmp_path, claims, judge)
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["faithfulness"] == 1.0
    assert outcome.diagnostics["false_claims"] == []
    assert outcome.diagnostics["unanchored_claims"] == []


def test_the_false_claim_list_is_in_roster_order_with_every_reason() -> None:
    report = ClaimReport(
        (
            ClaimVerdict("E1", "a", "neutral", 0.1),
            ClaimVerdict("E2", "b", "contradicted", 0.95),
            ClaimVerdict(
                "E2", "c", "neutral", 0.2, wrong_entity=True, out_of_range=True
            ),
            ClaimVerdict("E3", "d", "unanchored", 0.0),
            ClaimVerdict("E3", "e", "malformed", 0.0, wrong_entity=True),
        )
    )
    assert report.unanchored_claims == ({"entity_id": "E3", "claim": "d"},)
    assert report.false_count == 4
    assert report.diagnostics()["false_claims"] == [
        {"entity_id": "E2", "claim": "b", "reasons": ["contradicted"]},
        {"entity_id": "E2", "claim": "c", "reasons": ["wrong_entity", "out_of_range"]},
        {"entity_id": "E3", "claim": "d", "reasons": ["unanchored"]},
        {"entity_id": "E3", "claim": "e", "reasons": ["wrong_entity", "malformed"]},
    ]
    assert report.faithfulness == pytest.approx(1 / 5)
    assert report.penalty_factor(1.0, entity_count=3) == pytest.approx(0.2)
    # The smoke path charges only the deterministic reasons: "b" is contradicted only.
    assert report.penalty_factor(
        1.0, entity_count=3, judge_verdicts=False
    ) == pytest.approx(0.4)
