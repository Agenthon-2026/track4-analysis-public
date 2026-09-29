"""The interval leg is an interval score measured against the unit's DECLARED naive interval.

Adopted as 5.1.0 (2026-09-24): per roster row

    IS = (hi - lo) + (2/alpha) * (lo - y)+ + (2/alpha) * (y - hi)+,   alpha = 1 - interval_level

averaged over the roster exactly as MAE is. Interval quality ``iq = naive_IS / (naive_IS + IS)``
lies in (0, 1] and is 0.5 when the interval scores equal to the naive rule's. The composite is
``w_a * quality + w_c * iq``: both legs are ratios in [0, 1], so the worst case W is 0.

Replaced rule (5.0.0): ``w_a * quality - w_c * |coverage - interval_level|``. Under it an interval
wide enough to cover everything paid only ``w_c * (1 - level)``, less than a sharp naive band that
missed some rows; the first test below is that inversion.

The naive interval is organizer material. A naive rule whose interval score is zero, or which has
a zero-width interval on any row, is refused as an organizer fault, like a missing naive file.
A pure-label unit (no numeric truth) has no interval leg and needs no naive file.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Sequence
from typing import Any

import pytest

from qfbench2_track_analysis import scoring as S
from qfbench2_track_analysis.alignment import EntityRoster, align_predictions
from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import UnitOutcome, score_unit

from .synthetic import SUPPORTING_TEXT, StubJudge, answer_for, build_unit, outcome_for

W_ACC, W_CAL = 0.7, 0.3
TRUTH = (1.0, 2.0, 3.0)  # outcome_for's y, in roster order SYN-A, SYN-B, SYN-C
ENTITIES = ("SYN-A", "SYN-B", "SYN-C")
NAIVE_BAND = (1.5, 2.5)  # flat naive interval: covers y=2 only


def _is_row(lo: float, hi: float, y: float, alpha: float) -> float:
    """The ticket's formula, written out independently of the scorer."""
    return (
        (hi - lo) + (2.0 / alpha) * max(lo - y, 0.0) + (2.0 / alpha) * max(y - hi, 0.0)
    )


def _mean_is(
    bands: Sequence[tuple[float, float]], ys: Sequence[float], alpha: float
) -> float:
    return sum(_is_row(lo, hi, y, alpha) for (lo, hi), y in zip(bands, ys)) / len(ys)


def _answer(
    bands: Sequence[tuple[float, float]],
    *,
    points: Sequence[float] = (2.0, 2.0, 2.0),
    target_type: str = "regression",
    level: float = 0.90,
    baseline_id: str | None = None,
) -> dict[str, Any]:
    answer = answer_for(interval_level=level)
    answer["target_type"] = target_type
    for row, (lo, hi), point in zip(answer["entity_predictions"], bands, points):
        row["interval"] = {"level": level, "lo": lo, "hi": hi}
        row["point_forecast"] = point
        row["label"] = "beat" if row["entity_id"] == "SYN-A" else "miss"
    if baseline_id is not None:
        answer["notes"] = {"baseline_id": baseline_id}
    return answer


NAIVE_BANDS = (NAIVE_BAND,) * 3
NAIVE = _answer(NAIVE_BANDS, baseline_id="flat-two-band")
NAIVE_IS = _mean_is(NAIVE_BANDS, TRUTH, 0.1)


def _unit(
    tmp_path: pathlib.Path,
    *,
    target_type: str = "regression",
    naive: dict[str, Any] | None = NAIVE,
    level: float = 0.90,
    ys: Sequence[float | None] | None = TRUTH,
) -> pathlib.Path:
    unit = build_unit(tmp_path, target_type=target_type, interval_level=level)
    outcome = outcome_for()
    for row, y in zip(outcome["outcomes"], ys or (None, None, None)):
        row["y"] = y
    (unit / "reference").mkdir(exist_ok=True)
    (unit / "reference" / "outcome.json").write_text(
        json.dumps(outcome), encoding="utf-8"
    )
    if naive is not None:
        (unit / "reference" / "naive_answer.json").write_text(
            json.dumps(naive, indent=1) + "\n", encoding="utf-8"
        )
    return unit


def _score(
    tmp_path: pathlib.Path, unit: pathlib.Path, answer: dict[str, Any]
) -> UnitOutcome:
    out = tmp_path / "res"
    out.mkdir(parents=True, exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out},
        judge=StubJudge((SUPPORTING_TEXT,)),
        judge_provenance=provenance,
    )


def _iq(bands: Sequence[tuple[float, float]], *, alpha: float = 0.1) -> float:
    own = _mean_is(bands, TRUTH, alpha)
    return NAIVE_IS / (NAIVE_IS + own)


WIDE = ((-10.0, 14.0),) * 3  # covers every row, width 24
SHARP_HIT = tuple((y - 0.01, y + 0.01) for y in TRUTH)
SHARP_MISS = tuple((y + 2.0, y + 2.02) for y in TRUTH)  # misses every row by 2


def test_a_wide_always_covering_interval_scores_well_below_the_naive_band(
    tmp_path: pathlib.Path,
) -> None:
    """The inversion the old rule had: covering everything beat the sharper naive band."""
    unit = _unit(tmp_path)
    wide = _score(tmp_path / "w", unit, _answer(WIDE))
    naive = _score(tmp_path / "n", unit, _answer(NAIVE_BANDS))
    assert wide.state == naive.state == "participant_success"
    # Same point forecasts, so the accuracy leg is equal; only the interval leg differs.
    assert (
        wide.diagnostics["predictive_quality"]
        == naive.diagnostics["predictive_quality"]
    )
    assert wide.score < naive.score
    assert wide.diagnostics["interval_quality"] == pytest.approx(_iq(WIDE), abs=1e-12)
    assert wide.diagnostics["interval_quality"] < 0.3


def test_answering_the_naive_rule_scores_exactly_one_half_on_both_legs(
    tmp_path: pathlib.Path,
) -> None:
    unit = _unit(tmp_path)
    outcome = _score(tmp_path, unit, NAIVE)
    assert outcome.diagnostics["predictive_quality"] == 0.5
    assert outcome.diagnostics["interval_quality"] == 0.5
    assert outcome.score == pytest.approx(0.5, abs=1e-15)


def test_a_sharp_hit_scores_near_one(tmp_path: pathlib.Path) -> None:
    unit = _unit(tmp_path)
    outcome = _score(tmp_path, unit, _answer(SHARP_HIT))
    iq = outcome.diagnostics["interval_quality"]
    assert iq == pytest.approx(_iq(SHARP_HIT), abs=1e-12)
    assert iq > 0.99
    assert outcome.score == pytest.approx(
        W_ACC * outcome.diagnostics["predictive_quality"] + W_CAL * iq, abs=1e-12
    )


def test_a_sharp_miss_scores_below_the_wide_interval(tmp_path: pathlib.Path) -> None:
    """The miss penalty carries the 2/alpha factor: missing by 2 costs 40, more than width 24."""
    unit = _unit(tmp_path)
    miss = _score(tmp_path / "m", unit, _answer(SHARP_MISS))
    wide = _score(tmp_path / "w", unit, _answer(WIDE))
    assert miss.diagnostics["interval_quality"] == pytest.approx(
        _iq(SHARP_MISS), abs=1e-12
    )
    assert miss.diagnostics["interval_quality"] < wide.diagnostics["interval_quality"]
    assert miss.score < wide.score


def test_one_wild_miss_drags_iq_toward_zero_but_never_below(
    tmp_path: pathlib.Path,
) -> None:
    unit = _unit(tmp_path)
    bands = ((1000.0, 1000.02),) + SHARP_HIT[1:]
    outcome = _score(tmp_path, unit, _answer(bands, points=(1000.0, 2.0, 3.0)))
    iq = outcome.diagnostics["interval_quality"]
    assert iq == pytest.approx(_iq(bands), abs=1e-12)
    assert 0.0 < iq < 0.01
    assert outcome.score >= 0.0
    assert outcome.score == pytest.approx(
        W_ACC * outcome.diagnostics["predictive_quality"] + W_CAL * iq, abs=1e-12
    )


@pytest.mark.parametrize(
    "bands",
    [
        pytest.param(tuple((y, y) for y in TRUTH), id="naive-IS-zero"),
        pytest.param(((1.5, 2.5), (2.0, 2.0), (1.5, 2.5)), id="one-zero-width-row"),
    ],
)
def test_a_degenerate_naive_interval_is_an_organizer_fault(
    tmp_path: pathlib.Path, bands: tuple[tuple[float, float], ...]
) -> None:
    unit = _unit(tmp_path, naive=_answer(bands))
    with pytest.raises(T4OrganizerFault, match="naive"):
        _score(tmp_path, unit, _answer(SHARP_HIT))


@pytest.mark.parametrize("target_type", ["regression", "classification"])
def test_a_naive_answer_missing_an_interval_on_a_row_is_an_organizer_fault(
    tmp_path: pathlib.Path, target_type: str
) -> None:
    """Regression already aligned its naive file under 5.0.0; classification did not read one."""
    broken = _answer(NAIVE_BANDS, target_type=target_type)
    del broken["entity_predictions"][1]["interval"]
    unit = _unit(tmp_path, target_type=target_type, naive=broken)
    with pytest.raises(T4OrganizerFault, match="naive_answer.json"):
        _score(tmp_path, unit, _answer(SHARP_HIT, target_type=target_type))


def test_the_interval_level_sets_alpha() -> None:
    """At an 80 % unit alpha = 0.2, so a miss costs 10x its distance, not 20x.

    Driven through `_composite` with trusted 0.80 parameters: the published answer schema pins
    `interval.level` to 0.90, so an 80 % unit cannot be reached through `score_unit` today.
    """
    params = S.ScoringParams("regression", 0.80, 0.8, 0.5, (W_ACC, W_CAL))
    roster = EntityRoster(entity_ids=ENTITIES)
    naive = align_predictions(
        _answer(NAIVE_BANDS, level=0.80),
        roster,
        target_type="regression",
        interval_level=0.80,
    )
    own = align_predictions(
        _answer(SHARP_MISS, level=0.80),
        roster,
        target_type="regression",
        interval_level=0.80,
    )
    parts = S._composite(own, outcome_for(), params, roster, naive_aligned=naive)
    naive_is = _mean_is(NAIVE_BANDS, TRUTH, 0.2)
    own_is = _mean_is(SHARP_MISS, TRUTH, 0.2)
    assert parts["interval_quality"] == pytest.approx(
        naive_is / (naive_is + own_is), abs=1e-12
    )
    assert parts["interval_score"] == pytest.approx(own_is, abs=1e-12)
    assert parts["naive_interval_score"] == pytest.approx(naive_is, abs=1e-12)
    # and not the 90 % value
    assert parts["interval_score"] != pytest.approx(_mean_is(SHARP_MISS, TRUTH, 0.1))


def test_rows_are_aligned_by_entity_id_not_by_file_order(
    tmp_path: pathlib.Path,
) -> None:
    """Both files list their rows in reverse roster order, with a different band on each row."""
    naive_bands = ((0.0, 2.0), (1.0, 3.0), (2.5, 3.5))  # roster order A, B, C
    own_bands = ((0.9, 1.1), (2.5, 2.6), (0.0, 1.0))  # hit A, miss B high, miss C low
    naive = _answer(naive_bands)
    own = _answer(own_bands)
    naive["entity_predictions"].reverse()
    own["entity_predictions"].reverse()
    unit = _unit(tmp_path, naive=naive)
    outcome = _score(tmp_path, unit, own)
    n_is = _mean_is(naive_bands, TRUTH, 0.1)
    o_is = _mean_is(own_bands, TRUTH, 0.1)
    assert outcome.diagnostics["interval_quality"] == pytest.approx(
        n_is / (n_is + o_is), abs=1e-12
    )


def test_the_worst_case_is_zero() -> None:
    assert S.worst_case_score((0.7, 0.3), 0.90) == 0.0
    assert S.DOMAIN_MIN == 0.0
    assert S.clip_to_domain(-5.0) == 0.0
    assert S.scorer_identity()["metric_domain"] == {"min": 0.0, "max": 1.0}


def test_a_schema_invalid_submission_scores_the_new_worst_case(
    tmp_path: pathlib.Path,
) -> None:
    unit = _unit(tmp_path)
    answer = _answer(SHARP_HIT)
    del answer["entity_predictions"][0]["interval"]
    outcome = _score(tmp_path, unit, answer)
    assert outcome.state == "participant_failure"
    assert outcome.score == 0.0


def test_a_pure_label_unit_needs_its_naive_rule_from_5_2_0(
    tmp_path: pathlib.Path,
) -> None:
    """5.2.0: a pure-label unit's prediction leg is anchored to its declared naive rule, so the
    file is required; without an interval leg the composite is that leg alone ("R")."""
    unit = _unit(tmp_path, target_type="classification", naive=None, ys=None)
    with pytest.raises(T4OrganizerFault, match="naive_answer.json"):
        _score(tmp_path, unit, _answer(WIDE, target_type="classification"))


def test_a_classification_unit_with_numeric_truth_scores_the_interval_leg(
    tmp_path: pathlib.Path,
) -> None:
    """A label unit whose outcome also carries a numeric 0/1-style y. The leg applies."""
    naive = _answer(NAIVE_BANDS, target_type="classification")
    unit = _unit(tmp_path, target_type="classification", naive=naive)
    outcome = _score(tmp_path, unit, _answer(SHARP_HIT, target_type="classification"))
    iq = outcome.diagnostics["interval_quality"]
    assert iq == pytest.approx(_iq(SHARP_HIT), abs=1e-12)
    assert outcome.score == pytest.approx(W_ACC * 1.0 + W_CAL * iq, abs=1e-12)


def test_a_classification_unit_with_numeric_truth_and_no_naive_rule_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    unit = _unit(tmp_path, target_type="classification", naive=None)
    with pytest.raises(T4OrganizerFault, match="naive_answer.json"):
        _score(tmp_path, unit, _answer(SHARP_HIT, target_type="classification"))


def test_the_scorer_version_is_the_adopted_release() -> None:
    # 5.1.1 = 5.1.0 plus the two citation guards (test_citation_span_guards.py); 5.1.2 = the
    # review fixes (test_scorer_5_1_2_fixes.py); 5.1.3 = the narrowed own-value exemption
    # (test_own_value_exemption.py); 5.2.0 = the per-claim faithfulness penalty
    # (test_claim_penalty.py).
    assert S.SCORER_VERSION == "5.2.0"


def test_the_interval_score_is_the_toolkits_and_faults_as_the_tracks() -> None:
    """The formula lives in the toolkit; the track name delegates to it and keeps its fault class."""
    from qfbench2_common.scoring import faithfulness as F

    lo = [b[0] for b in SHARP_MISS]
    hi = [b[1] for b in SHARP_MISS]
    ours = S.mean_interval_score(lo, hi, list(TRUTH), 0.9)
    assert ours == F.mean_interval_score(lo, hi, list(TRUTH), 0.9)
    assert ours == pytest.approx(_mean_is(SHARP_MISS, TRUTH, 0.1), abs=1e-12)
    for bad in (([], [], []), ([0.0], [1.0, 2.0], [0.5])):
        with pytest.raises(T4OrganizerFault, match="one lo, hi and y"):
            S.mean_interval_score(*bad, 0.9)
