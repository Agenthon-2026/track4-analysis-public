"""Numeric targets, plus the frozen failure semantics: W, the denominator, and clipping.

Four measured defects live here, and they pointed in three different directions:

* an **inverted interval** (``lo > hi``) was accepted and merely scored zero coverage;
* a **wrong interval level** was caught only by the public path's JSON-schema ``const``, so the
  sealed scorer admitted and scored the same submission — a hard divergence between the preview
  score and the sealed score;
* ``Inf``/``NaN`` passed ``float(... or 0.0)`` unfiltered, and NaN rows were then **dropped from the
  coverage denominator**, which is the forbidden NaN-row drop;
* an **unknown target type** silently fell back to label accuracy.

And the frozen policy they all feed: an inadmissible unit contributes ``W = -0.27`` and **stays in
the denominator**. ``score=None`` was what removed it. Real scores are clipped into the same
domain, so failing is never strictly better than any attainable real outcome.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any

import pytest

from qfbench2_common.contracts import FailureCode
from qfbench2_common.failure_labels import DEFAULT_ADMISSIBILITY_CODE

from qfbench2_track_analysis.alignment import (
    AlignedPredictions,
    EntityRoster,
    align_predictions,
)
from qfbench2_track_analysis.codes import (
    DEFAULT_CODE,
    T4OrganizerFault,
    T4ParticipantFailure,
    T4Reason,
    public_code_for,
    t4_public_detail,
)
from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    ScoringParams,
    UnitOutcome,
    clip_to_domain,
    score_unit,
    worst_case_score,
)

from .synthetic import SUPPORTING_TEXT, StubJudge, answer_for, build_unit

ROSTER = EntityRoster(
    entity_ids=("SYN-A", "SYN-B", "SYN-C"), labels=("beat", "miss", "inline")
)


def _align(
    answer: dict[str, Any],
    *,
    target_type: str = "classification",
    interval_level: float = 0.90,
) -> AlignedPredictions:
    return align_predictions(
        answer, ROSTER, target_type=target_type, interval_level=interval_level
    )


def _score(tmp_path: pathlib.Path, answer: dict[str, Any]) -> UnitOutcome:
    unit = build_unit(tmp_path, with_outcome=True)
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out},
        judge=StubJudge((SUPPORTING_TEXT,)),
        judge_provenance=provenance,
    )


# --- positive control ---------------------------------------------------------------------------
def test_a_well_formed_numeric_answer_aligns(tmp_path: pathlib.Path) -> None:
    aligned = _align(answer_for())
    assert aligned.count == 3
    assert all(math.isfinite(x) for x in aligned.lo + aligned.hi + aligned.pred_values)


# --- intervals ------------------------------------------------------------------------------------
def test_inverted_interval_is_refused(tmp_path: pathlib.Path) -> None:
    answer = answer_for()
    answer["entity_predictions"][0]["interval"] = {"level": 0.90, "lo": 9.0, "hi": 1.0}
    with pytest.raises(T4ParticipantFailure) as excinfo:
        _align(answer)
    assert excinfo.value.reason is T4Reason.INTERVAL_INVALID


def test_wrong_interval_level_is_refused_in_the_alignment_layer_not_only_by_jsonschema() -> (
    None
):
    """The sealed path has no JSON-schema gate to lean on, so the check must live here too."""
    answer = answer_for(interval_level=0.50)
    with pytest.raises(T4ParticipantFailure) as excinfo:
        _align(answer)
    assert excinfo.value.reason is T4Reason.INTERVAL_LEVEL_MISMATCH


def test_absent_interval_level_is_an_error_not_the_declared_one() -> None:
    answer = answer_for()
    del answer["entity_predictions"][0]["interval"]["level"]
    with pytest.raises(T4ParticipantFailure) as excinfo:
        _align(answer)
    assert excinfo.value.reason is T4Reason.INTERVAL_LEVEL_MISMATCH


# --- nonfinite --------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_point_forecast_is_a_participant_failure(bad: float) -> None:
    answer = answer_for()
    answer.pop("target_type")
    answer["entity_predictions"][0]["point_forecast"] = bad
    with pytest.raises(T4ParticipantFailure) as excinfo:
        _align(answer, target_type="regression")
    assert excinfo.value.reason is T4Reason.NONFINITE_VALUE


def test_nonfinite_interval_bound_is_a_participant_failure() -> None:
    answer = answer_for()
    answer["entity_predictions"][0]["interval"]["hi"] = float("nan")
    with pytest.raises(T4ParticipantFailure):
        _align(answer)


def test_no_row_is_ever_dropped_to_shrink_a_denominator(tmp_path: pathlib.Path) -> None:
    """A NaN row used to vanish from the coverage denominator. It now fails the whole unit."""
    answer = answer_for()
    answer["entity_predictions"][1]["interval"]["lo"] = float("nan")
    outcome = _score(tmp_path, answer)
    assert outcome.state == "participant_failure"
    assert outcome.score == DOMAIN_MIN


def test_coverage_denominator_is_the_full_roster(tmp_path: pathlib.Path) -> None:
    answer = answer_for(lo=0.0, hi=0.5)  # covers none of y = 1.0, 2.0, 3.0
    outcome = _score(tmp_path, answer)
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["interval_coverage"] == 0.0
    assert outcome.diagnostics["graded_entity_count"] == 3


# --- target type ---------------------------------------------------------------------------------
def test_unknown_target_type_is_refused_not_defaulted_to_classification() -> None:
    with pytest.raises(T4OrganizerFault):
        ScoringParams(
            target_type="regressionn",
            interval_level=0.90,
            faithfulness_threshold=0.80,
            tau_citation=0.5,
            composite_weights=(0.7, 0.3),
        )


def test_an_answer_declaring_a_different_target_type_is_refused() -> None:
    answer = answer_for()
    answer["target_type"] = "ranking"
    with pytest.raises(T4ParticipantFailure) as excinfo:
        _align(answer, target_type="classification")
    assert excinfo.value.reason is T4Reason.TARGET_TYPE_MISMATCH


def test_a_card_value_that_disagrees_with_the_trusted_plan_is_refused() -> None:
    with pytest.raises(T4OrganizerFault, match="disagrees with the trusted plan"):
        ScoringParams.from_sources(
            trusted={"interval_level": 0.90, "target_type": "classification"},
            card_params={"interval_level": 0.50},
        )


# --- frozen failure semantics -------------------------------------------------------------------
def test_worst_case_is_zero_under_the_interval_ratio() -> None:
    """5.1.0: both composite legs are ratios in [0, 1], so W moved from -0.27 to 0."""
    assert worst_case_score((0.7, 0.3), 0.90) == 0.0
    assert DOMAIN_MIN == 0.0


def test_a_failed_unit_scores_w_and_is_never_none(tmp_path: pathlib.Path) -> None:
    outcome = _score(tmp_path, answer_for(entities=("SYN-A",)))
    assert outcome.score is not None
    assert outcome.score == DOMAIN_MIN
    assert isinstance(outcome.score, float)


def test_real_scores_are_clipped_into_the_same_domain() -> None:
    assert clip_to_domain(-5.0) == DOMAIN_MIN
    assert clip_to_domain(5.0) == DOMAIN_MAX
    assert clip_to_domain(0.25) == pytest.approx(0.25)


def test_failure_can_never_beat_an_attainable_real_score() -> None:
    """The property the domain floor exists for: W is the worst end of the domain, and reals are clipped in."""
    for real in (-1.0, -0.27, 0.0, 0.5, 1.0, 2.0):
        assert clip_to_domain(real) >= DOMAIN_MIN


# --- public projection ----------------------------------------------------------------------------
def test_public_detail_is_code_plus_counts_and_carries_no_text() -> None:
    detail = t4_public_detail(
        T4Reason.ENTITY_MISSING, missing_count=2, expected_count=3
    )
    assert detail == {
        "code": "incomplete_output",
        "missing_count": 2,
        "expected_count": 3,
    }
    assert all(isinstance(v, int) for k, v in detail.items() if k != "code")


def test_a_schema_rejection_does_not_leak_the_validator_message(
    tmp_path: pathlib.Path,
) -> None:
    """Pre-fix this returned {"reason": str(e)} — a ~1.5 KB dump of the whole answer schema."""
    answer = answer_for()
    del answer["entity_predictions"][0]["interval"]
    outcome = _score(tmp_path, answer)
    assert outcome.state == "participant_failure"
    assert set(outcome.detail) <= {
        "code",
        "missing_count",
        "extra_count",
        "invalid_row_count",
        "nonfinite_count",
        "expected_count",
        "observed_count",
        "rejected_node_count",
        "violation_count",
    }
    assert not any(isinstance(v, str) for k, v in outcome.detail.items() if k != "code")


def test_an_unknown_count_key_is_rejected_rather_than_silently_dropped() -> None:
    with pytest.raises(ValueError):
        t4_public_detail(T4Reason.ENTITY_MISSING, note=1)


# --- the twelfth code -----------------------------------------------------------------------------
# Registry 1.1.0 added `domain_gate_failed`. Before it, every reason without an exact counterpart
# reported `schema_invalid`, whose published gloss says the output did not match the published
# output schema — false for a roster mismatch, an invalid interval, a nonfinite value and
# unsupported evidence, and a participant reading it goes hunting for a schema defect that is not
# there. These pin the split, in both directions.
@pytest.mark.parametrize(
    "reason",
    [
        T4Reason.ENTITY_UNKNOWN,
        T4Reason.ENTITY_DUPLICATE,
        T4Reason.INTERVAL_INVALID,
        T4Reason.INTERVAL_LEVEL_MISMATCH,
        T4Reason.NONFINITE_VALUE,
        T4Reason.LABEL_INVALID,
        T4Reason.EVIDENCE_UNSUPPORTED,
        T4Reason.CITATION_MALFORMED,
        T4Reason.CITATION_UNRESOLVED,
        T4Reason.TARGET_TYPE_MISMATCH,
        T4Reason.TASK_ID_MISMATCH,
    ],
)
def test_a_well_formed_output_that_fails_a_domain_check_says_so(
    reason: T4Reason,
) -> None:
    assert public_code_for(reason) is FailureCode.DOMAIN_GATE_FAILED


@pytest.mark.parametrize(
    ("reason", "code"),
    [
        (T4Reason.SCHEMA_INVALID, FailureCode.SCHEMA_INVALID),
        (T4Reason.NO_ANSWER, FailureCode.NO_OUTPUT),
        (T4Reason.MALFORMED_ANSWER, FailureCode.MALFORMED_OUTPUT),
        (T4Reason.ENTITY_MISSING, FailureCode.INCOMPLETE_OUTPUT),
        (T4Reason.CITATION_POST_CUTOFF, FailureCode.CUTOFF_VIOLATION),
        (T4Reason.CITATION_UNDATED, FailureCode.CUTOFF_VIOLATION),
    ],
)
def test_a_reason_with_an_exact_counterpart_keeps_it(
    reason: T4Reason, code: FailureCode
) -> None:
    """A track that means *schema* still says schema. The default must not swallow these."""
    assert public_code_for(reason) is code


def test_the_default_matches_the_hubs_own_default() -> None:
    """Two projections of one verdict: the code we stamp and the code the labels derive to."""
    assert DEFAULT_CODE is DEFAULT_ADMISSIBILITY_CODE
