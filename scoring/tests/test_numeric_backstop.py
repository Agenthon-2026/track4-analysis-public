"""The numeric backstop: a claim's figures must be anchored in the passage it cites.

An NLI cross-encoder reads "$5.9 billion" against a passage that says "$5.2 billion" as
consistent. So the figures are checked by exact code before the judge, and the judge is asked
only about claims whose figures the cited passage carries. The rule is deliberately narrow --
"no figure in the claim appears in the passage" -- because an honest claim legitimately states
figures the passage does not carry beside the one it does: a change computed from two rows, a
value copied from the task table, a rounded total. A false refusal costs W. See `numeric.py` for
the reasoning behind each tolerance; the cases below are the specification.
"""

from __future__ import annotations

import json
import pathlib
from decimal import Decimal
from typing import Any

import pytest

from qfbench2_track_analysis.alignment import EntityRoster, align_predictions
from qfbench2_track_analysis.codes import T4Reason
from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.numeric import claim_number_status, figures
from qfbench2_track_analysis.scoring import UnitOutcome, evaluate_claims, score_unit

from .synthetic import StubJudge, answer_for, build_unit

ROSTER = ("SYN-A",)


# --- the rule, value to value -------------------------------------------------------------------
@pytest.mark.parametrize(
    ("claim", "span"),
    [
        ("Revenue rose 12%", "revenue was up 12.0% year over year"),
        ("Sales of 1,234 units", "shipped 1234 units in the quarter"),
        ("Revenue was $1.5 billion", "gizmo sales $ 1,497,313 (in thousands)"),
        ("The fee is $8.5 million", "termination fee $8.5M"),
        ("Fees of about $4.08 million", "a fee of $4,075,000"),
        ("A ratio of 0.28%", "widgets to total widgets of .28% at period end"),
        (
            "Held at 7.25 to 7.5 percent",
            "kept the synthetic band at 7-1/4 to 7-1/2 percent",
        ),
        ("A decline of 2%", "Sales (2) %"),
        ("A margin of 12%", "margin of 0.12 on"),
        ("Purchase price of $58,428,612.00", "The price is $58,428,612.00."),
    ],
)
def test_a_figure_present_in_the_span_anchors_the_claim(claim: str, span: str) -> None:
    assert claim_number_status(claim, [span]) == "anchored"


def test_a_fabricated_figure_is_unanchored() -> None:
    assert claim_number_status(
        "Revenue was $5.9 billion", ["Revenue was $5.2 billion"]
    ) == ("unanchored")


def test_one_anchored_figure_admits_a_claim_with_a_derived_one() -> None:
    """The stated limit: exact code cannot tell a computed change from an invented one."""
    claim = "NCLNLSR was 0.2375 percent and then 0.5124 percent, a change of +27.49 bp"
    assert claim_number_status(claim, ["| 0.2375 | ... | 0.5124 |"]) == "anchored"


def test_dates_years_identifiers_and_form_names_are_not_figures() -> None:
    text = (
        "Filed 2025-03-31 (March 31, 2025; 31 March 2025; 3/31/2025) for FY2024, Q1 2026 and "
        "2026Q1, the 3rd 10-K, Item 2.02, CIK 1033225, CERT 58210, S000002564, the 10-year "
        "note, the meeting of June 9-10, 2031."
    )
    assert figures(text) == ()
    assert claim_number_status(text, ["nothing numeric here"]) == "no_figures"


def test_a_claim_without_figures_goes_to_the_judge() -> None:
    assert (
        claim_number_status("Management expects margins to widen", ["12%"])
        == "no_figures"
    )


def test_the_participants_own_forecast_is_not_a_figure_the_span_must_carry() -> None:
    claim = "This baseline forecasts -2.55% for the quarter"
    assert claim_number_status(claim, ["net revenue was $248 million"]) == "unanchored"
    assert (
        claim_number_status(
            claim, ["net revenue was $248 million"], submitted=[-2.55, -5.0, 0.0, None]
        )
        == "no_figures"
    )


def test_figures_are_read_as_absolute_values() -> None:
    assert figures("(2) % and -3.5 and +7bn") == (
        Decimal("2"),
        Decimal("3.5"),
        Decimal("7E+9"),
    )


# --- the rule inside the gate -------------------------------------------------------------------
SPAN = "Synthetic Issuer A reported revenue of $5.2 billion, up 8% year over year."


def _aligned(claims: list[str]) -> Any:
    answer = answer_for(entities=ROSTER, claim_text=claims[0])
    first = answer["entity_predictions"][0]["claims"][0]
    # In range for the lookup's document (`SPAN`): from 5.1.1 an end past the text names no
    # passage instead of being clamped to the whole text.
    answer["entity_predictions"][0]["claims"] = [
        {**first, "claim": text, "span_end": len(SPAN)} for text in claims
    ]
    return align_predictions(
        answer,
        EntityRoster(entity_ids=ROSTER),
        target_type="classification",
        interval_level=0.90,
    )


def test_an_unanchored_claim_never_reaches_the_judge_and_counts_as_false() -> None:
    judge = StubJudge((SPAN,))
    report = evaluate_claims(
        _aligned(["Revenue was $5.9 billion", "Revenue rose 8%"]),
        lambda _doc_id: {"text": SPAN},
        judge,
        target_type="classification",
        interval_scored=True,
        contradiction_bar=0.9,
    )
    assert [v.status for v in report.verdicts] == ["unanchored", "neutral"]
    assert report.faithfulness == 0.5
    assert [h for _, h in judge.calls] == ["Revenue rose 8%"]


def _score(tmp_path: pathlib.Path, claims: list[str], judge: object) -> UnitOutcome:
    """Score `claims`, all citing one synthetic document whose text is `SPAN`."""
    from .synthetic import PRE_CUTOFF_DOC

    document = {
        "doc_id": PRE_CUTOFF_DOC,
        "doc_date": "2026-02-01",
        "source": "synthetic",
        "spans": [{"span_id": 0, "start": 0, "end": len(SPAN), "text": SPAN}],
        "text": SPAN,
    }
    unit = build_unit(
        tmp_path, entities=ROSTER, with_outcome=True, docs={PRE_CUTOFF_DOC: document}
    )
    answer = answer_for(entities=ROSTER, claim_text=claims[0])
    first = answer["entity_predictions"][0]["claims"][0]
    answer["entity_predictions"][0]["claims"] = [
        {**first, "claim": text, "span_end": len(SPAN)} for text in claims
    ]
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out}, judge=judge, judge_provenance=provenance
    )


def test_an_unanchored_claim_costs_its_share_and_nothing_more(
    tmp_path: pathlib.Path,
) -> None:
    """5.2.0: the backstop no longer refuses the unit. One unanchored claim among five for one
    entity costs 1/(1 + min(4, 3)) = 1/4 of the score (the soft floor, k = 1: at most three claims
    per entity dilute a false one); the four anchored claims the judge does not contradict are
    neutral, whatever their entailment."""
    claims = [
        "Revenue was $5.9 billion",  # unanchored: false
        "Revenue rose 8% on weak demand",  # anchored, not contradicted: neutral
        "Revenue of $5.2 billion beat guidance",
        "Growth of 8% was the slowest in years",
        "The $5.2 billion figure excludes divestitures",
    ]
    outcome = _score(tmp_path, claims, StubJudge(()))
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["unanchored_claim_count"] == 1
    assert outcome.diagnostics["false_claim_count"] == 1
    assert outcome.diagnostics["claim_count"] == 5
    assert outcome.diagnostics["faithfulness_factor"] == pytest.approx(0.75)
    assert outcome.score == pytest.approx(
        0.75 * outcome.diagnostics["composite_before_penalty"]
    )


def test_a_unit_whose_every_claim_is_unanchored_scores_zero_but_is_not_refused(
    tmp_path: pathlib.Path,
) -> None:
    outcome = _score(tmp_path, ["Revenue was $5.9 billion"], StubJudge((SPAN,)))
    assert outcome.state == "participant_success"
    assert outcome.failure_code is None
    assert outcome.diagnostics["unanchored_claim_count"] == 1
    assert outcome.diagnostics["faithfulness_factor"] == 0.0
    assert outcome.score == 0.0


def test_the_public_label_for_the_backstop_is_the_frozen_unfaithful_citation_label() -> (
    None
):
    from qfbench2_common.failure_labels import FailureLabel

    from qfbench2_track_analysis.codes import public_code_for, public_label_for

    assert (
        public_label_for(T4Reason.CLAIM_NUMBER_NOT_IN_SPAN)
        is FailureLabel.T4_UNFAITHFUL_CITATION
    )
    assert (
        public_code_for(T4Reason.CLAIM_NUMBER_NOT_IN_SPAN).value == "domain_gate_failed"
    )
