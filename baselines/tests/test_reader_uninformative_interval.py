"""The interval the baseline emits when it found no usable evidence.

Measured on the ten public-dev units (2026-09-12): on all five regression units the agent
extracted nothing, so every point forecast was the 0.0 fallback and every band was the 0.05
floor -- an assertion of 90% confidence in a range the agent has no reason to believe. Coverage
was 2 of 38. The composite scores an interval on |coverage - level| with no width penalty, so
asserting a narrow range buys nothing and costs almost the whole calibration weight.
"""

from __future__ import annotations

import math

from baselines.baseline_agent.reader import predict_entity


def test_without_evidence_the_interval_does_not_assert_a_narrow_range() -> None:
    """THE DEFECT: no evidence, yet a +/-0.05 interval around a meaningless 0.0."""
    entity = {"entity_id": "UST2Y", "start_yield_pct": 3.59, "maturity_years": 2}
    pred = predict_entity(entity, "no numbers here")
    band = (pred["hi"] - pred["lo"]) / 2
    assert band > 10.0, f"still asserting a narrow interval: +/-{band}"


def test_the_uninformative_width_scales_with_the_entitys_own_magnitudes() -> None:
    """A basis-point unit and a ratio unit must not get the same width."""
    small = predict_entity({"latest_published_mom_pct": 0.18}, "")
    large = predict_entity({"prior_year_q_eps": 4.33}, "")
    assert (large["hi"] - large["lo"]) > (small["hi"] - small["lo"])


def test_an_entity_with_no_numbers_still_yields_a_valid_interval() -> None:
    pred = predict_entity({"entity_id": "X", "name": "no numeric field"}, "")
    assert math.isfinite(pred["lo"]) and math.isfinite(pred["hi"])
    assert pred["lo"] <= pred["point_forecast"] <= pred["hi"]


def test_a_grounded_prediction_keeps_its_evidence_scaled_interval() -> None:
    """CONTROL: where the agent DID extract a number, the interval is unchanged."""
    pred = predict_entity(
        {"consensus_eps": 1.0, "threshold_pct": 0.05}, "earnings per share of 1.50"
    )
    assert (pred["lo"], pred["hi"]) == (1.35, 1.65)


def test_declared_magnitudes_are_read_literally_without_unit_conversion() -> None:
    """KNOWN LIMITATION, recorded rather than left implicit.

    `t4-fomc-curve` entities declare `unit: "bps_change"` while their only numeric field is a
    yield in PERCENT (3.59). The width is therefore derived from 3.59 and not from 359, and those
    two units reached 2 of 6 coverage in the 2026-09-12 dev check where the other three reached
    7/7, 11/11 and 6/8. Converting would require knowing that this family's target is a hundredth
    of its declared field -- per-family knowledge this agent deliberately does not carry, and a
    multiplier chosen to close the gap would be fitted on outcomes that are not published.
    """
    entity = {"start_yield_pct": 3.59, "maturity_years": 2, "unit": "bps_change"}
    band = (predict_entity(entity, "")["hi"] - predict_entity(entity, "")["lo"]) / 2
    assert band == 35.9
