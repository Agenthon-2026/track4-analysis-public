"""A regression unit is scored against its DECLARED naive rule, read from the unit's reference.

Since scorer 5.1.0: regression predictive quality is ``naive_mae / (naive_mae + mae)``, where the
naive rule's point forecasts come from ``reference/naive_answer.json`` -- a full answer file in the
analysis schema, aligned to the trusted roster exactly as the participant's answer is. A scored
regression unit without that file is an ORGANIZER fault (a missing scoring parameter): it must
never become a participant zero and never fall back to the retired formula.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import pytest

from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import (
    UnitOutcome,
    build_smoke_verifier,
    score_unit,
)

from .synthetic import SUPPORTING_TEXT, StubJudge, answer_for, build_unit

W_ACC, W_CAL, LEVEL = 0.7, 0.3, 0.90
# outcome_for: y = 1.0, 2.0, 3.0. 5.1.0 interval leg: every answer here and the naive rule carry
# answer_for's interval [0.5, 3.5], so interval quality is exactly 0.5.
INTERVAL_LEG = W_CAL * 0.5


def _regression_answer(
    forecasts: tuple[float, ...], *, baseline_id: str | None = None
) -> dict[str, Any]:
    answer = answer_for()
    answer["target_type"] = "regression"
    for row, value in zip(answer["entity_predictions"], forecasts):
        row["point_forecast"] = value
    if baseline_id is not None:
        answer["notes"] = {"baseline_id": baseline_id}
    return answer


def _unit(
    tmp_path: pathlib.Path,
    *,
    target_type: str = "regression",
    naive: dict[str, Any] | None,
    with_outcome: bool = True,
) -> pathlib.Path:
    unit = build_unit(
        tmp_path, target_type=target_type, with_outcome=with_outcome, with_naive=False
    )
    if naive is not None:
        (unit / "reference").mkdir(exist_ok=True)
        (unit / "reference" / "naive_answer.json").write_text(
            json.dumps(naive, indent=1) + "\n", encoding="utf-8"
        )
    return unit


def _score(
    tmp_path: pathlib.Path, unit: pathlib.Path, answer: dict[str, Any], **kwargs: Any
) -> UnitOutcome:
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out},
        judge=StubJudge((SUPPORTING_TEXT,)),
        judge_provenance=provenance,
        **kwargs,
    )


NAIVE = _regression_answer(
    (2.0, 2.0, 2.0), baseline_id="flat-two"
)  # naive MAE (1 + 0 + 1) / 3


def test_a_regression_unit_scores_the_soft_ratio_against_its_naive_rule(
    tmp_path: pathlib.Path,
) -> None:
    unit = _unit(tmp_path, naive=NAIVE)
    outcome = _score(tmp_path, unit, _regression_answer((1.0, 2.0, 4.0)))  # MAE 1/3
    assert outcome.state == "participant_success"
    expected_pq = (2.0 / 3.0) / (2.0 / 3.0 + 1.0 / 3.0)
    assert outcome.diagnostics["predictive_quality"] == pytest.approx(
        expected_pq, abs=1e-15
    )
    assert outcome.score == pytest.approx(W_ACC * expected_pq + INTERVAL_LEG, abs=1e-12)


def test_answering_the_naive_rule_scores_parity(tmp_path: pathlib.Path) -> None:
    unit = _unit(tmp_path, naive=NAIVE)
    outcome = _score(tmp_path, unit, _regression_answer((2.0, 2.0, 2.0)))
    assert outcome.diagnostics["predictive_quality"] == 0.5


def test_the_outcome_records_which_naive_rule_scored_the_unit(
    tmp_path: pathlib.Path,
) -> None:
    unit = _unit(tmp_path, naive=NAIVE)
    blob = (unit / "reference" / "naive_answer.json").read_bytes()
    outcome = _score(tmp_path, unit, _regression_answer((1.0, 2.0, 4.0)))
    assert outcome.diagnostics["naive_answer_file"] == "naive_answer.json"
    assert (
        outcome.diagnostics["naive_answer_sha256"] == hashlib.sha256(blob).hexdigest()
    )
    assert outcome.diagnostics["naive_baseline_id"] == "flat-two"


def test_a_scored_regression_unit_without_its_naive_rule_is_an_organizer_fault(
    tmp_path: pathlib.Path,
) -> None:
    unit = _unit(tmp_path, naive=None)
    with pytest.raises(T4OrganizerFault, match="naive_answer.json"):
        _score(tmp_path, unit, _regression_answer((1.0, 2.0, 4.0)))


def test_the_verifier_path_scores_the_soft_ratio_and_faults_without_the_rule(
    tmp_path: pathlib.Path,
) -> None:
    unit = _unit(tmp_path / "with", naive=NAIVE)
    out = tmp_path / "with" / "res"
    out.mkdir()
    (out / "answer.json").write_text(
        json.dumps(_regression_answer((1.0, 2.0, 4.0))), encoding="utf-8"
    )
    ctx: dict[str, Any] = {
        "unit_dir": unit,
        "output_dir": out,
        "failure_map": tmp_path / "f.jsonl",
    }
    verdict = build_smoke_verifier(ctx).run(ctx)
    assert verdict.detail["predictive_quality"] == pytest.approx(2.0 / 3.0, abs=1e-15)

    bare = _unit(tmp_path / "without", naive=None)
    out2 = tmp_path / "without" / "res"
    out2.mkdir()
    (out2 / "answer.json").write_text(
        json.dumps(_regression_answer((1.0, 2.0, 4.0))), encoding="utf-8"
    )
    ctx2: dict[str, Any] = {
        "unit_dir": bare,
        "output_dir": out2,
        "failure_map": tmp_path / "g.jsonl",
    }
    with pytest.raises(T4OrganizerFault, match="naive_answer.json"):
        build_smoke_verifier(ctx2).run(ctx2)


def test_a_naive_answer_that_fails_alignment_is_an_organizer_fault(
    tmp_path: pathlib.Path,
) -> None:
    """The naive file is organizer material: a defect in it never becomes a participant failure."""
    broken = _regression_answer((2.0, 2.0, 2.0))
    broken["entity_predictions"] = broken["entity_predictions"][
        :2
    ]  # drops a roster entity
    unit = _unit(tmp_path, naive=broken)
    with pytest.raises(T4OrganizerFault, match="naive_answer.json fails alignment"):
        _score(tmp_path, unit, _regression_answer((1.0, 2.0, 4.0)))


def test_a_practice_regression_unit_without_an_outcome_needs_no_naive_rule(
    tmp_path: pathlib.Path,
) -> None:
    """No outcome, nothing to score: admissibility only, and no scoring parameter is missing."""
    unit = _unit(tmp_path, naive=None, with_outcome=False)
    outcome = _score(
        tmp_path, unit, _regression_answer((1.0, 2.0, 4.0)), require_outcome=False
    )
    assert outcome.state == "unrankable"


def test_classification_accuracy_is_unchanged_and_its_naive_rule_anchors_only_the_interval(
    tmp_path: pathlib.Path,
) -> None:
    """POSITIVE CONTROL: classification accuracy is untouched by the naive file.

    5.0.0 read no naive file on a classification unit. From 5.1.0 a classification unit WITH
    numeric truth needs one for the interval leg (`test_interval_score_leg.py` pins the refusal
    without it); its labels never enter the accuracy leg, which is shown here by a naive rule
    whose labels are all wrong while the participant's accuracy stays 1.0.
    """
    naive = answer_for()
    naive["notes"] = {"baseline_id": "all-inline"}
    for row in naive["entity_predictions"]:
        row["label"] = "inline"
    unit = _unit(tmp_path, target_type="classification", naive=naive)
    answer = answer_for()
    for row, label in zip(answer["entity_predictions"], ("beat", "miss", "miss")):
        row["label"] = label
    outcome = _score(tmp_path, unit, answer)
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["predictive_quality"] == 1.0
    assert outcome.score == pytest.approx(W_ACC * 1.0 + INTERVAL_LEG)
    assert outcome.diagnostics["naive_baseline_id"] == "all-inline"
