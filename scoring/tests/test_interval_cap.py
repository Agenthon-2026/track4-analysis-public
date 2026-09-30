"""Scorer 5.2.1: the interval leg is capped by the prediction leg.

    interval_quality = min(naive_IS / (naive_IS + IS), max(0.5, predictive_quality))

The interval part can score above 0.5 only as far as the point forecast beats the naive rule. Under
5.2.0 an answer that copied the naive rule's points (prediction leg exactly 0.5) and narrowed the
naive band scored above the naive rule with no information: with a naive band that covers every
row, scaling it by c about the naive point gives ``IS = c x naive_IS`` and a composite of
``0.35 + 0.3 / (1 + c)``, 0.526 at c = 0.7. Below 0.5 nothing changes, so a band that misses still
costs. The uncapped value is recorded as ``raw_interval_quality``.

Synthetic unit (`outcome_for`): roster SYN-A, SYN-B, SYN-C with y = 1, 2, 3 and labels beat, miss,
miss. The naive rule here predicts 2 on every row (labels beat, beat, beat: accuracy 1/3) with the
band [-1, 5], which covers every row, so narrowing it is free until a row falls outside.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Sequence
from typing import Any

import pytest

from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import UnitOutcome, score_unit

from .synthetic import SUPPORTING_TEXT, StubJudge, answer_for, build_unit

W_ACC, W_CAL = 0.7, 0.3
ROSTER = ("SYN-A", "SYN-B", "SYN-C")
TRUTH = (1.0, 2.0, 3.0)
NAIVE_POINT = 2.0
NAIVE_HALF_WIDTH = 3.0  # band [-1, 5]: covers every row
NAIVE_LABELS = (
    "beat",
    "beat",
    "beat",
)  # accuracy 1/3, below perfect, so the anchor is < 1
TARGET_TYPES = ("regression", "classification", "ranking")

Band = tuple[float, float]


def _is_row(lo: float, hi: float, y: float, alpha: float = 0.1) -> float:
    return (
        (hi - lo) + (2.0 / alpha) * max(lo - y, 0.0) + (2.0 / alpha) * max(y - hi, 0.0)
    )


def _mean_is(bands: Sequence[Band]) -> float:
    return sum(
        _is_row(lo, hi, y) for (lo, hi), y in zip(bands, TRUTH, strict=True)
    ) / len(TRUTH)


def _scaled(c: float, shift: float = 0.0) -> tuple[Band, ...]:
    """The naive band scaled by c about the naive point, then shifted."""
    half = c * NAIVE_HALF_WIDTH
    return ((NAIVE_POINT - half + shift, NAIVE_POINT + half + shift),) * len(ROSTER)


NAIVE_BANDS = _scaled(1.0)
NAIVE_IS = _mean_is(NAIVE_BANDS)


def _rows(
    bands: Sequence[Band],
    points: Sequence[float],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {
            "label": label,
            "point_forecast": point,
            "interval": {"level": 0.90, "lo": lo, "hi": hi},
        }
        for (lo, hi), point, label in zip(bands, points, labels, strict=True)
    ]


def _score(
    tmp_path: pathlib.Path,
    target_type: str,
    bands: Sequence[Band],
    *,
    points: Sequence[float] = (NAIVE_POINT,) * 3,
    labels: Sequence[str] = NAIVE_LABELS,
    interval_leg: bool = True,
    numeric_truth: bool = True,
    naive_labels: Sequence[str] = NAIVE_LABELS,
) -> UnitOutcome:
    unit = build_unit(
        tmp_path, entities=ROSTER, target_type=target_type, with_outcome=True
    )
    if not interval_leg:
        card = unit / "card.toml"
        text = card.read_text(encoding="utf-8")
        anchor = "tau_citation           = 0.5\n"
        assert anchor in text
        card.write_text(
            text.replace(anchor, anchor + "interval_leg           = false\n"),
            encoding="utf-8",
        )
    if not numeric_truth:
        truth_file = unit / "reference" / "outcome.json"
        truth_rows = json.loads(truth_file.read_text(encoding="utf-8"))
        for row in truth_rows["outcomes"]:
            row["y"] = None
        truth_file.write_text(json.dumps(truth_rows), encoding="utf-8")
    naive_path = unit / "reference" / "naive_answer.json"
    naive = json.loads(naive_path.read_text(encoding="utf-8"))
    for row, values in zip(
        naive["entity_predictions"],
        _rows(NAIVE_BANDS, (NAIVE_POINT,) * 3, naive_labels),
        strict=True,
    ):
        row.update(values)
    naive_path.write_text(json.dumps(naive), encoding="utf-8")
    answer = answer_for(entities=ROSTER)
    answer["target_type"] = target_type
    for row, values in zip(
        answer["entity_predictions"], _rows(bands, points, labels), strict=True
    ):
        row.update(values)
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    outcome = score_unit(
        {"unit_dir": unit, "output_dir": out},
        judge=StubJudge((SUPPORTING_TEXT,)),
        judge_provenance=provenance,
    )
    assert outcome.state == "participant_success"
    # No false claim, so the score is the composite itself.
    assert outcome.diagnostics["faithfulness_factor"] == 1.0
    return outcome


NAIVE_ONLY_BANDS = {
    "c=0": _scaled(0.0),
    "c=0.3": _scaled(0.3),
    "c=0.7": _scaled(0.7),
    "c=1": _scaled(1.0),
    "c=1.5": _scaled(1.5),
    "c=0.7-shifted": _scaled(0.7, shift=0.5),
    "asymmetric": ((NAIVE_POINT - 1.05, NAIVE_POINT + 2.0),) * 3,
}


@pytest.mark.parametrize("target_type", TARGET_TYPES)
@pytest.mark.parametrize("band_id", list(NAIVE_ONLY_BANDS))
def test_naive_points_cannot_beat_half_with_any_band(
    tmp_path: pathlib.Path, target_type: str, band_id: str
) -> None:
    """The naive rule's points (or labels) with any band score at most 0.5, and exactly 0.5 with
    the naive band itself."""
    outcome = _score(tmp_path, target_type, NAIVE_ONLY_BANDS[band_id])
    assert outcome.diagnostics["predictive_quality"] == 0.5
    assert (
        outcome.score <= 0.5 + 1e-12
    ), f"{target_type} {band_id}: a naive-points answer scored {outcome.score:.4f}"
    if band_id == "c=1":
        assert outcome.score == pytest.approx(0.5, abs=1e-15)


def test_the_fixture_is_exploitable_without_the_cap(tmp_path: pathlib.Path) -> None:
    """Positive control for the test above: at c = 0.7 every row is still covered, so the
    uncapped interval quality is 1 / 1.7 and the uncapped composite would be 0.526."""
    bands = _scaled(0.7)
    assert all(lo <= y <= hi for (lo, hi), y in zip(bands, TRUTH, strict=True))
    outcome = _score(tmp_path, "regression", bands)
    raw = outcome.diagnostics["raw_interval_quality"]
    assert raw == pytest.approx(1 / 1.7, abs=1e-12)
    assert W_ACC * 0.5 + W_CAL * raw == pytest.approx(0.35 + 0.3 / 1.7, abs=1e-12)
    assert W_ACC * 0.5 + W_CAL * raw > 0.526
    assert outcome.diagnostics["interval_quality"] == 0.5


def test_interval_credit_capped_at_point_credit(tmp_path: pathlib.Path) -> None:
    """Prediction leg 0.6 and a band whose raw interval quality is 0.9: the leg is cut to 0.6."""
    # Every point is 4/9 high: MAE 4/9 against the naive 2/3, so the leg is (2/3) / (2/3 + 4/9) = 0.6.
    points = tuple(y + 4 / 9 for y in TRUTH)
    # Width 2/3 and covering every row: IS = 2/3 = naive_IS / 9, so the raw quality is 0.9.
    bands = tuple((y - 1 / 9, y + 5 / 9) for y in TRUTH)
    assert _mean_is(bands) == pytest.approx(NAIVE_IS / 9, abs=1e-12)
    outcome = _score(tmp_path, "regression", bands, points=points)
    assert outcome.diagnostics["predictive_quality"] == pytest.approx(0.6, abs=1e-12)
    assert outcome.diagnostics["raw_interval_quality"] == pytest.approx(0.9, abs=1e-12)
    assert outcome.diagnostics["interval_quality"] == pytest.approx(0.6, abs=1e-12)
    assert outcome.score == pytest.approx(W_ACC * 0.6 + W_CAL * 0.6, abs=1e-12)


def test_cap_never_raises_a_bad_band(tmp_path: pathlib.Path) -> None:
    """Prediction leg 0.8 and a band that misses every row (raw 0.3): the leg stays 0.3."""
    # Every point is 1/6 high: (2/3) / (2/3 + 1/6) = 0.8.
    points = tuple(y + 1 / 6 for y in TRUTH)
    # Width 4 and 0.5 above every row: IS = 4 + 20 x 0.5 = 14 = naive_IS x 7/3, so raw = 0.3.
    bands = tuple((y + 0.5, y + 4.5) for y in TRUTH)
    assert _mean_is(bands) == pytest.approx(NAIVE_IS * 7 / 3, abs=1e-12)
    outcome = _score(tmp_path, "regression", bands, points=points)
    assert outcome.diagnostics["predictive_quality"] == pytest.approx(0.8, abs=1e-12)
    assert outcome.diagnostics["raw_interval_quality"] == pytest.approx(0.3, abs=1e-12)
    assert outcome.diagnostics["interval_quality"] == pytest.approx(0.3, abs=1e-12)
    assert outcome.score == pytest.approx(W_ACC * 0.8 + W_CAL * 0.3, abs=1e-12)


@pytest.mark.parametrize("target_type", TARGET_TYPES)
def test_bad_band_with_naive_points_still_costs(
    tmp_path: pathlib.Path, target_type: str
) -> None:
    """Naive points with a band that misses every row: the leg is the raw value, below 0.5, as in
    5.2.0."""
    bands = ((5.0, 5.5),) * 3  # misses y = 1, 2, 3 by 4, 3, 2
    outcome = _score(tmp_path, target_type, bands)
    raw = NAIVE_IS / (NAIVE_IS + _mean_is(bands))
    assert raw < 0.5
    assert outcome.diagnostics["raw_interval_quality"] == pytest.approx(raw, abs=1e-12)
    assert (
        outcome.diagnostics["interval_quality"]
        == outcome.diagnostics["raw_interval_quality"]
    )
    assert outcome.score == pytest.approx(W_ACC * 0.5 + W_CAL * raw, abs=1e-12)


def test_cap_is_continuous_at_half(tmp_path: pathlib.Path) -> None:
    """A prediction leg a hair above 0.5 unlocks only that hair of interval credit, not the full
    band credit (the hard-gate alternative would jump to the raw value here)."""
    eps = 8e-9  # SYN-A's point moves 8e-9 toward y: the leg is 2 / (4 - eps), about 0.5 + 1e-9
    points = (NAIVE_POINT - eps, NAIVE_POINT, NAIVE_POINT)
    outcome = _score(tmp_path, "regression", _scaled(0.7), points=points)
    pq = outcome.diagnostics["predictive_quality"]
    assert 0.5 < pq <= 0.5 + 2e-9
    assert outcome.diagnostics["raw_interval_quality"] > 0.58
    assert outcome.diagnostics["interval_quality"] == pytest.approx(pq, abs=1e-15)
    assert outcome.diagnostics["interval_quality"] <= 0.5 + 2e-9


@pytest.mark.parametrize(
    ("labels", "leg"),
    [
        pytest.param(NAIVE_LABELS, 0.5, id="naive-labels"),
        pytest.param(("beat", "miss", "miss"), 1.0, id="perfect-labels"),
        pytest.param(("miss", "beat", "beat"), 0.0, id="wrong-labels"),
    ],
)
@pytest.mark.parametrize("kind", ["interval-leg-false", "pure-label"])
def test_no_interval_leg_units_unaffected(
    tmp_path: pathlib.Path, labels: tuple[str, ...], leg: float, kind: str
) -> None:
    """Without an interval leg there is nothing to cap: the unit scores its prediction leg alone,
    whatever band the answer carries, exactly as in 5.2.0."""
    outcome = _score(
        tmp_path,
        "classification",
        _scaled(0.3),
        labels=labels,
        interval_leg=kind != "interval-leg-false",
        numeric_truth=kind != "pure-label",
    )
    assert outcome.diagnostics["interval_quality"] is None
    assert outcome.diagnostics["raw_interval_quality"] is None
    assert outcome.diagnostics["predictive_quality"] == pytest.approx(leg)
    assert outcome.score == pytest.approx(leg, abs=1e-15)


def test_cap_reads_the_anchored_prediction_leg(tmp_path: pathlib.Path) -> None:
    """The cap is keyed to the ANCHORED prediction leg, not the raw quality. Here the naive labels
    are right on two rows of three (raw accuracy 2/3, above 0.5), and the answer copies them with a
    band that would earn raw interval quality 1 / 1.7. Keyed to the raw quality the interval part
    could rise to 2/3 and the naive labels would beat the naive rule; keyed to the anchored leg
    (0.5 for the naive labels) it cannot pass 0.5."""
    naive_labels = ("beat", "miss", "beat")
    outcome = _score(
        tmp_path,
        "classification",
        _scaled(0.7),
        labels=naive_labels,
        naive_labels=naive_labels,
    )
    assert outcome.diagnostics["raw_predictive_quality"] == pytest.approx(2 / 3)
    assert outcome.diagnostics["predictive_quality"] == 0.5
    assert outcome.diagnostics["raw_interval_quality"] == pytest.approx(
        1 / 1.7, abs=1e-12
    )
    assert outcome.diagnostics["interval_quality"] == 0.5
    assert outcome.score <= 0.5 + 1e-12
