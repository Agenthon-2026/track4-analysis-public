"""Scorer 5.2.0: classification and ranking are anchored to the declared naive rule.

The raw quality q (label accuracy, or Spearman rescaled to [0, 1]) maps piecewise linearly so that
0 -> 0, the anchor -> 0.5 and a perfect answer -> 1. The anchor is the declared naive rule's own quality, or, as a
safeguard, the stronger of that and a trivial constant rule known ex ante (ranking: a constant forecast, q = 0.5;
classification has none). A unit without an interval leg scores the prediction leg alone (no 0.7 ceiling).
Regression is unchanged (its soft ratio is already naive-anchored).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import (
    UnitOutcome,
    anchored_quality,
    hydrate,
    score_unit,
)

from .synthetic import SUPPORTING_TEXT, StubJudge, answer_for, build_unit

ROSTER = (
    "SYN-A",
    "SYN-B",
    "SYN-C",
)  # outcome_for: labels beat, miss, miss; y = 1, 2, 3


@pytest.mark.parametrize(
    ("q", "anchor", "expected"),
    [
        (0.0, 0.4, 0.0),
        (0.4, 0.4, 0.5),
        (1.0, 0.4, 1.0),
        (0.2, 0.4, 0.25),
        (0.7, 0.4, 0.75),
        (1.0, 1.0, 1.0),
        (0.5, 1.0, 0.25),
        (0.0, 0.0, 0.5),
        (0.6, 0.0, 0.8),
    ],
)
def test_the_piecewise_linear_map(q: float, anchor: float, expected: float) -> None:
    assert anchored_quality(q, anchor) == pytest.approx(expected)


def _score(
    tmp_path: pathlib.Path,
    target_type: str,
    answer_values: dict[str, Any],
    naive_values: dict[str, Any],
    *,
    interval_leg: bool = True,
) -> UnitOutcome:
    unit = build_unit(
        tmp_path, entities=ROSTER, target_type=target_type, with_outcome=True
    )
    if not interval_leg:
        card = unit / "card.toml"
        text = card.read_text(encoding="utf-8")
        anchor = "tau_citation           = 0.5\n"
        card.write_text(
            text.replace(anchor, anchor + "interval_leg           = false\n"),
            encoding="utf-8",
        )
    naive_path = unit / "reference" / "naive_answer.json"
    naive = json.loads(naive_path.read_text(encoding="utf-8"))
    for row in naive["entity_predictions"]:
        row.update(naive_values[row["entity_id"]])
    naive_path.write_text(json.dumps(naive), encoding="utf-8")
    answer = answer_for(entities=ROSTER)
    answer["target_type"] = target_type
    for row in answer["entity_predictions"]:
        row.update(answer_values[row["entity_id"]])
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out},
        judge=StubJudge((SUPPORTING_TEXT,)),
        judge_provenance=provenance,
    )


def _labels(*labels: str) -> dict[str, Any]:
    return {eid: {"label": lab} for eid, lab in zip(ROSTER, labels, strict=True)}


def _points(*values: float) -> dict[str, Any]:
    return {eid: {"point_forecast": v} for eid, v in zip(ROSTER, values, strict=True)}


NAIVE_ALL_BEAT = _labels("beat", "beat", "beat")  # accuracy 1/3


@pytest.mark.parametrize(
    ("labels", "leg"),
    [
        (("beat", "beat", "beat"), 0.5),
        (("beat", "miss", "miss"), 1.0),
        (("miss", "beat", "beat"), 0.0),
        (("beat", "miss", "beat"), 0.75),
    ],
)
def test_classification_is_anchored_to_the_declared_naive(
    tmp_path: pathlib.Path, labels: tuple[str, ...], leg: float
) -> None:
    outcome = _score(tmp_path, "classification", _labels(*labels), NAIVE_ALL_BEAT)
    assert outcome.diagnostics["naive_predictive_quality"] == pytest.approx(1 / 3)
    assert outcome.diagnostics["predictive_anchor"] == pytest.approx(1 / 3)
    assert outcome.diagnostics["predictive_quality"] == pytest.approx(leg)


def test_matching_the_naive_rule_scores_exactly_one_half(
    tmp_path: pathlib.Path,
) -> None:
    """Naive labels AND naive interval: the whole composite is 0.5, as on a regression unit."""
    outcome = _score(tmp_path, "classification", NAIVE_ALL_BEAT, NAIVE_ALL_BEAT)
    assert outcome.diagnostics["interval_quality"] == pytest.approx(0.5)
    assert outcome.score == pytest.approx(0.5)


def test_a_unit_without_an_interval_leg_scores_the_leg_alone(
    tmp_path: pathlib.Path,
) -> None:
    perfect = _score(
        tmp_path / "p",
        "classification",
        _labels("beat", "miss", "miss"),
        NAIVE_ALL_BEAT,
        interval_leg=False,
    )
    assert perfect.diagnostics["interval_quality"] is None
    assert perfect.score == pytest.approx(1.0)  # no 0.7 ceiling ("R")
    naive = _score(
        tmp_path / "n",
        "classification",
        NAIVE_ALL_BEAT,
        NAIVE_ALL_BEAT,
        interval_leg=False,
    )
    assert naive.score == pytest.approx(0.5)


def test_ranking_anchor_is_the_stronger_of_naive_and_the_constant_rule(
    tmp_path: pathlib.Path,
) -> None:
    """A declared naive ranking WORSE than a constant forecast (reversed: q = 0) is not the anchor; the
    constant rule (q = 0.5) is. So a constant forecast scores 0.5 and the weak naive itself scores 0."""
    reversed_naive = _points(3.0, 2.0, 1.0)
    const = _score(tmp_path / "c", "ranking", _points(2.0, 2.0, 2.0), reversed_naive)
    assert const.diagnostics["naive_predictive_quality"] == pytest.approx(0.0)
    assert const.diagnostics["predictive_anchor"] == pytest.approx(0.5)
    assert const.diagnostics["predictive_quality"] == pytest.approx(0.5)
    weak = _score(tmp_path / "w", "ranking", reversed_naive, reversed_naive)
    assert weak.diagnostics["predictive_quality"] == pytest.approx(0.0)
    perfect = _score(tmp_path / "p", "ranking", _points(1.0, 2.0, 3.0), reversed_naive)
    assert perfect.diagnostics["predictive_quality"] == pytest.approx(1.0)


def test_a_ranking_naive_stronger_than_constant_is_the_anchor(
    tmp_path: pathlib.Path,
) -> None:
    good_naive = _points(1.0, 3.0, 2.0)  # rho 0.5 -> q 0.75
    outcome = _score(tmp_path, "ranking", good_naive, good_naive)
    assert outcome.diagnostics["predictive_anchor"] == pytest.approx(0.75)
    assert outcome.diagnostics["predictive_quality"] == pytest.approx(0.5)


def test_regression_is_unchanged(tmp_path: pathlib.Path) -> None:
    outcome = _score(
        tmp_path, "regression", _points(2.0, 2.0, 2.0), _points(2.0, 2.0, 2.0)
    )
    assert "predictive_anchor" not in outcome.diagnostics
    assert outcome.diagnostics["predictive_quality"] == pytest.approx(0.5)


@pytest.mark.parametrize("target_type", ["classification", "regression", "ranking"])
def test_an_outcome_without_a_naive_file_is_refused_at_load(
    tmp_path: pathlib.Path, target_type: str
) -> None:
    """Every unit with a resolved outcome needs its declared naive rule, and loading the unit
    refuses without it, before any answer is read."""
    missing = build_unit(
        tmp_path / "missing",
        target_type=target_type,
        with_outcome=True,
        with_naive=False,
    )
    with pytest.raises(T4OrganizerFault, match="naive_answer.json"):
        hydrate({"unit_dir": missing})
    # controls: the same unit with its naive file loads, and a practice unit needs none
    present = build_unit(
        tmp_path / "present", target_type=target_type, with_outcome=True
    )
    hydrate({"unit_dir": present})
    practice = build_unit(
        tmp_path / "practice",
        target_type=target_type,
        with_outcome=False,
        with_naive=False,
    )
    hydrate({"unit_dir": practice})
