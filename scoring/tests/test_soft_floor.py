"""The soft-floor faithfulness factor (scorer 5.2.0, adopted 2026-09-29).

    factor = 1 - F / (F + min(T, 3E))        (to the power penalty_k = 1)

F = false claims, T = every other claim (N - F), E = the roster's entity count; no claims -> 1.
Each false claim costs a share of the unit; other claims beyond 3 x E in total do not dilute that cost; a
unit with no false claims is not penalised; neutral claims neither earn nor cost
beyond that cap. The reviewer's properties (padding-review/REVIEW.md, soft-floor addendum) are
asserted here on an exhaustive small grid.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qfbench2_track_analysis.scoring import (
    PADDING_CAP_PER_ENTITY,
    ClaimReport,
    ClaimVerdict,
)


def report(false: int, true: int, *, contradicted: int = 0) -> ClaimReport:
    """`false` unanchored claims, `contradicted` judge-false claims and `true` neutral ones."""
    return ClaimReport(
        tuple(ClaimVerdict("E", f"f{i}", "unanchored", 0.0) for i in range(false))
        + tuple(
            ClaimVerdict("E", f"c{i}", "contradicted", 0.95)
            for i in range(contradicted)
        )
        + tuple(ClaimVerdict("E", f"t{i}", "neutral", 0.0) for i in range(true))
    )


def soft(f: int, t: int, e: int) -> float:
    return report(f, t).penalty_factor(1.0, entity_count=e)


def test_the_cap_is_three_claims_per_entity() -> None:
    assert PADDING_CAP_PER_ENTITY == 3


def test_no_claims_is_exactly_one() -> None:
    assert ClaimReport(()).penalty_factor(1.0, entity_count=4) == 1.0


@pytest.mark.parametrize("e", [1, 2, 5, 10])
@pytest.mark.parametrize("t", [0, 1, 7, 30, 200])
def test_no_false_claims_is_exactly_one(e: int, t: int) -> None:
    assert soft(0, t, e) == 1.0


@pytest.mark.parametrize("e", [1, 3, 10])
@pytest.mark.parametrize("f", [1, 2, 9])
def test_every_claim_false_is_zero(e: int, f: int) -> None:
    assert soft(f, 0, e) == 0.0


@pytest.mark.parametrize("e", [1, 2, 5, 10])
def test_padding_never_dilutes_past_the_bound(e: int) -> None:
    """The adaptive attack: whatever the number of neutral claims, a false claim keeps its share
    of at least 1/(F + 3E)."""
    for f in range(1, 12):
        bound = 1 - f / (f + 3 * e)
        for t in range(0, 6 * e + 40):
            assert soft(f, t, e) <= bound + 1e-15
        assert soft(f, 3 * e, e) == pytest.approx(bound)
        assert soft(f, 3 * e + 500, e) == pytest.approx(
            bound
        )  # beyond the cap: no change


@pytest.mark.parametrize("e", [1, 2, 5, 10])
def test_up_to_the_cap_it_is_the_old_share(e: int) -> None:
    for f in range(0, 10):
        for t in range(0, 3 * e + 1):
            n = f + t
            old = 1.0 if n == 0 else 1 - f / n
            assert soft(f, t, e) == pytest.approx(old)


@pytest.mark.parametrize("e", [1, 2, 5])
def test_monotone(e: int) -> None:
    for f in range(0, 10):
        for t in range(0, 4 * e + 3):
            assert soft(f + 1, t, e) <= soft(
                f, t, e
            )  # one more false claim never helps
            assert soft(f, t + 1, e) >= soft(
                f, t, e
            )  # one more non-false claim never hurts
            if f:
                assert soft(f - 1, t + 1, e) >= soft(
                    f, t, e
                )  # making a false claim true never hurts


def test_the_smoke_path_counts_contradicted_claims_as_not_false() -> None:
    r = report(1, 2, contradicted=3)  # F = 4 with the judge, F = 1 without it
    assert r.penalty_factor(1.0, entity_count=1) == pytest.approx(1 - 4 / (4 + 2))
    # smoke: T = N - F_smoke = 5, capped at 3E = 3
    assert r.penalty_factor(1.0, entity_count=1, judge_verdicts=False) == pytest.approx(
        1 - 1 / (1 + 3)
    )


def test_the_entity_count_is_required_and_positive() -> None:
    with pytest.raises(TypeError):
        report(1, 1).penalty_factor(1.0)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        report(1, 1).penalty_factor(1.0, entity_count=0)


def test_the_smoke_path_applies_the_soft_floor_with_the_roster_count(
    tmp_path: Path,
) -> None:
    """The smoke path (a non-rankable local run) charges the deterministic reasons with the same
    soft floor and the same E as production: one unanchored claim among 22, spread over three
    entities, keeps 1 - 1/(1 + min(21, 3 x 3)) = 0.9 of the score (E = 1 would give 0.75)."""
    import json

    from qfbench2_track_analysis.judge_factory import build_smoke_judge
    from qfbench2_track_analysis.scoring import score_unit
    from scoring.tests.synthetic import answer_for, build_unit

    roster = ("SYN-A", "SYN-B", "SYN-C")
    unit = build_unit(tmp_path, entities=roster, with_outcome=True)
    answer = answer_for(entities=roster)
    base = answer["entity_predictions"][0]["claims"][0]
    for i, row in enumerate(answer["entity_predictions"]):
        row["claims"] = [
            {
                **base,
                "claim": f"Synthetic Issuer {row['entity_id'][-1]} published a report, note {n}.",
            }
            for n in "abcdefg"
        ]
        if i == 0:
            row["claims"].append(
                {**base, "claim": "Revenue was $5.9 billion."}
            )  # unanchored: false
    out = tmp_path / "out"
    out.mkdir()
    (out / "answer.json").write_text(json.dumps(answer))
    judge, provenance = build_smoke_judge()
    outcome = score_unit(
        {"unit_dir": str(unit), "output_dir": str(out), "_enforce_faithfulness": False},
        judge=judge,
        judge_provenance=provenance,
    )
    assert outcome.diagnostics["claim_count"] == 22
    assert outcome.diagnostics["unanchored_claim_count"] == 1
    assert outcome.diagnostics["faithfulness_factor"] == pytest.approx(1 - 1 / (1 + 9))
    assert outcome.score == pytest.approx(
        0.9 * outcome.diagnostics["composite_before_penalty"]
    )
