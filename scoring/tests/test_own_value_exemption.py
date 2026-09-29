"""The own-value exemption of the numeric backstop is narrow (scorer 5.1.3).

A claim figure need not appear in the
cited passage ONLY when it exactly equals a value the participant submitted AND the unit scores:

- the point forecast on a regression or ranking unit;
- the interval bounds only when the interval leg is actually scored: ``interval_leg`` true AND
  the resolved outcome carries numeric truth (one predicate, shared with the composite; a
  pure-label unit never scores its interval);
- never the rank, never an unscored bound, never an unscored point value;
- exact numeric equality, with no scale step and no rounding.

Before 5.1.3 every submitted value (point, lo, hi, rank) exempted any claim figure it matched after
the span tolerances (re-basing by powers of a thousand or by 100, rounding to the claim's
precision). An unscored bound costs nothing to set, so a participant could set one to a fabricated
amount and the claim skipped the figure check and went to the judge, which reads "$5.9 billion"
against "$5.2 billion" as consistent.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from qfbench2_track_analysis.alignment import EntityRoster, align_predictions
from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.numeric import claim_number_status
from qfbench2_track_analysis.scoring import (
    ScoringParams,
    UnitOutcome,
    _interval_leg_scored,
    _scored_own_values,
    claim_interval_scored,
    evaluate_claims,
    score_unit,
)

from .synthetic import PRE_CUTOFF_DOC, StubJudge, answer_for, build_unit

ROSTER = ("SYN-A",)
SPAN = "Synthetic Issuer A reported revenue of $5.2 billion, up 8% year over year."


def _aligned(
    claim: str,
    *,
    target_type: str,
    point: float = 1.0,
    lo: float = 0.5,
    hi: float = 3.5,
    rank: int | None = None,
) -> Any:
    answer = answer_for(
        entities=ROSTER, claim_text=claim, point_forecast=point, lo=lo, hi=hi
    )
    answer["target_type"] = target_type
    row = answer["entity_predictions"][0]
    row["claims"][0]["span_end"] = len(SPAN)
    if rank is not None:
        row["rank"] = rank
    return align_predictions(
        answer,
        EntityRoster(entity_ids=ROSTER),
        target_type=target_type,
        interval_level=0.90,
    )


def _status(
    claim: str, *, target_type: str, interval_scored: bool, **values: Any
) -> str:
    """The claim's verdict through the gate's own claim loop. The judge contradicts nothing, as
    an NLI model often does for a near-miss figure, so only the backstop can make it false."""
    judge = StubJudge((SPAN,))
    report = evaluate_claims(
        _aligned(claim, target_type=target_type, **values),
        lambda _doc_id: {"text": SPAN},
        judge,
        target_type=target_type,
        interval_scored=interval_scored,
        contradiction_bar=0.9,
    )
    (verdict,) = report.verdicts
    return verdict.status


# --- the review's abuse case ----------------------------------------------------------------------
def test_an_unscored_interval_bound_no_longer_exempts_a_fabricated_figure() -> None:
    """interval_leg = false: the bounds are scored nowhere, so "$5.9 billion" against a passage
    saying $5.2 billion is unanchored even though hi = 5.9."""
    status = _status(
        "Revenue was $5.9 billion",
        target_type="classification",
        interval_scored=False,
        lo=5.2,
        hi=5.9,
    )
    assert status == "unanchored"


def _span_unit(
    tmp_path: pathlib.Path, *, leg_off: bool, pure_label: bool = False
) -> pathlib.Path:
    """A one-entity classification unit whose only document is `SPAN`. `leg_off` declares
    `interval_leg = false` in the card; `pure_label` strips every numeric target from the
    resolved outcome (labels only), leaving `interval_leg` as declared."""
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
    if leg_off:
        card = unit / "card.toml"
        anchor = "tau_citation           = 0.5\n"
        text = card.read_text(encoding="utf-8")
        assert text.count(anchor) == 1
        card.write_text(
            text.replace(anchor, anchor + "interval_leg           = false\n"),
            encoding="utf-8",
        )
    if pure_label:
        path = unit / "reference" / "outcome.json"
        outcome = json.loads(path.read_text(encoding="utf-8"))
        for row in outcome["outcomes"]:
            row.pop("y", None)
            row.pop("true_value", None)
        path.write_text(json.dumps(outcome), encoding="utf-8")
    return unit


def _score_abuse(tmp_path: pathlib.Path, unit: pathlib.Path) -> UnitOutcome:
    """lo/hi 5.2-5.9 and the claim "$5.9 billion" against `SPAN` ($5.2 billion); the judge
    entails the passage, so only the backstop can make the claim false."""
    answer = answer_for(
        entities=ROSTER, claim_text="Revenue was $5.9 billion", lo=5.2, hi=5.9
    )
    answer["entity_predictions"][0]["claims"][0]["span_end"] = len(SPAN)
    out = tmp_path / "res"
    out.mkdir()
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out},
        judge=StubJudge((SPAN,)),
        judge_provenance=provenance,
    )


def test_the_abuse_case_is_false_for_its_numbers_through_score_unit(
    tmp_path: pathlib.Path,
) -> None:
    """The same case end to end: the card declares interval_leg = false and the flag reaches the
    claim check. Before 5.1.3 the judge admitted the claim and the unit scored."""
    outcome = _score_abuse(tmp_path, _span_unit(tmp_path, leg_off=True))
    # 5.2.0: the unanchored claim is false and, as the unit's only claim, zeroes it.
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["unanchored_claim_count"] == 1
    assert outcome.diagnostics["false_claim_count"] == 1
    assert outcome.score == 0.0


# --- what stays exempt ----------------------------------------------------------------------------
@pytest.mark.parametrize("target_type", ["regression", "ranking"])
def test_an_honest_restatement_of_the_exact_point_forecast_stays_exempt(
    target_type: str,
) -> None:
    extra = {"rank": 1} if target_type == "ranking" else {}
    status = _status(
        "We forecast revenue of $5.9 billion",
        target_type=target_type,
        interval_scored=True,
        point=5.9,
        **extra,
    )
    # No figure left to anchor: the judge decides, and here it entails.
    assert status == "neutral"


def test_an_exact_point_forecast_is_exempt_at_any_written_precision() -> None:
    """Exact means numerically equal: 5.90 in the claim is the submitted 5.9."""
    status = _status(
        "We forecast revenue of $5.90 billion",
        target_type="regression",
        interval_scored=True,
        point=5.9,
    )
    assert status == "neutral"


def test_interval_bounds_are_exempt_when_the_interval_leg_is_scored() -> None:
    status = _status(
        "Our band runs from 4.1 to 6.3",
        target_type="classification",
        interval_scored=True,
        lo=4.1,
        hi=6.3,
    )
    assert status == "neutral"


def test_the_same_bounds_are_not_exempt_when_the_interval_leg_is_off() -> None:
    status = _status(
        "Our band runs from 4.1 to 6.3",
        target_type="classification",
        interval_scored=False,
        lo=4.1,
        hi=6.3,
    )
    assert status == "unanchored"


# --- what no longer exempts -----------------------------------------------------------------------
def test_a_near_match_to_the_point_forecast_is_no_longer_exempt() -> None:
    """5.87 rounded to the claim's precision is 5.9; that tolerance is for passages, not for the
    participant's own values."""
    status = _status(
        "Revenue was $5.9 billion",
        target_type="regression",
        interval_scored=True,
        point=5.87,
    )
    assert status == "unanchored"


def test_a_scale_step_of_the_point_forecast_is_no_longer_exempt() -> None:
    status = _status(
        "Revenue was $5.9 billion",
        target_type="regression",
        interval_scored=True,
        point=5900.0,
    )
    assert status == "unanchored"


def test_the_rank_no_longer_exempts() -> None:
    status = _status(
        "Revenue rose to $1 billion",
        target_type="ranking",
        interval_scored=True,
        point=7.3,
        lo=6.0,
        hi=8.0,
        rank=1,
    )
    assert status == "unanchored"


def test_an_unscored_point_value_on_a_classification_unit_does_not_exempt() -> None:
    status = _status(
        "Revenue was $5.9 billion",
        target_type="classification",
        interval_scored=False,
        point=5.9,
    )
    assert status == "unanchored"


# --- the selection, value by value ----------------------------------------------------------------
@pytest.mark.parametrize(
    ("target_type", "interval_scored", "rank", "expected"),
    [
        ("regression", True, None, (7.0, 0.5, 3.5)),
        ("ranking", True, 1, (7.0, 0.5, 3.5)),
        ("classification", True, None, (0.5, 3.5)),
        ("classification", False, None, ()),
    ],
)
def test_scored_own_values_selects_only_scored_fields(
    target_type: str,
    interval_scored: bool,
    rank: int | None,
    expected: tuple[float, ...],
) -> None:
    aligned = _aligned("x", target_type=target_type, point=7.0, rank=rank)
    assert (
        _scored_own_values(
            aligned, 0, target_type=target_type, interval_scored=interval_scored
        )
        == expected
    )


def test_an_unknown_target_type_is_an_organizer_fault() -> None:
    aligned = _aligned("x", target_type="regression")
    with pytest.raises(T4OrganizerFault, match="target_type"):
        _scored_own_values(aligned, 0, target_type="survival", interval_scored=True)


def test_claim_number_status_compares_own_values_exactly() -> None:
    span = ["net revenue was $248 million"]
    assert claim_number_status("forecast 5.9", span, submitted=[5.9]) == "no_figures"
    assert claim_number_status("forecast 5.9", span, submitted=[5.87]) == "unanchored"
    assert claim_number_status("forecast 5.9", span, submitted=[5900]) == "unanchored"
    assert claim_number_status("forecast 590%", span, submitted=[5.9]) == "unanchored"


# --- a pure-label unit never scores its interval, so its bounds never exempt ----------------------
def test_a_pure_label_unit_with_the_default_interval_leg_does_not_exempt_its_bounds(
    tmp_path: pathlib.Path,
) -> None:
    """No `interval_leg = false` in the card, but the outcome carries no numeric truth: the
    composite drops the interval leg, so the bounds are unscored and "$5.9 billion" (= hi) must
    appear in the passage like any other figure."""
    unit = _span_unit(tmp_path, leg_off=False, pure_label=True)
    outcome = _score_abuse(tmp_path, unit)
    # 5.2.0: the unanchored claim is false and, as the unit's only claim, zeroes it.
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["unanchored_claim_count"] == 1
    assert outcome.diagnostics["false_claim_count"] == 1
    assert outcome.score == 0.0


def test_the_same_unit_with_numeric_truth_still_exempts_its_bounds(
    tmp_path: pathlib.Path,
) -> None:
    """Control for the test above: identical unit and answer, numeric truth kept, so the leg is
    scored and the claim restating hi goes to the judge (which entails it)."""
    unit = _span_unit(tmp_path, leg_off=False, pure_label=False)
    outcome = _score_abuse(tmp_path, unit)
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["unanchored_claim_count"] == 0


def test_the_local_check_reads_the_same_predicate_as_the_gate(
    tmp_path: pathlib.Path,
) -> None:
    """`check_answer` on the organizer side (outcome mounted) agrees with the gate on the
    pure-label unit: the claim is unanchored there too."""
    from faithfulness.judge import build_unit_context, check_answer

    unit = _span_unit(tmp_path, leg_off=False, pure_label=True)
    answer = answer_for(
        entities=ROSTER, claim_text="Revenue was $5.9 billion", lo=5.2, hi=5.9
    )
    answer["entity_predictions"][0]["claims"][0]["span_end"] = len(SPAN)
    ctx = build_unit_context(unit)
    assert claim_interval_scored(ctx) is False
    check = check_answer(answer, ctx, StubJudge((SPAN,)))
    assert [c.status for c in check.claims] == ["unanchored"]


@pytest.mark.parametrize(
    ("leg", "numeric_truth", "expected"),
    [
        (True, True, True),
        (True, False, False),  # pure-label: never scored, whatever the flag says
        (False, True, False),
        (False, False, False),
        (True, None, True),  # no outcome mounted: the declared flag is all there is
        (False, None, False),
    ],
)
def test_one_predicate_decides_whether_the_interval_leg_is_scored(
    leg: bool, numeric_truth: bool | None, expected: bool
) -> None:
    params = ScoringParams(
        target_type="classification",
        interval_level=0.9,
        faithfulness_threshold=0.8,
        tau_citation=0.5,
        composite_weights=(0.7, 0.3),
        interval_leg=leg,
    )
    assert _interval_leg_scored(params, numeric_truth=numeric_truth) is expected
