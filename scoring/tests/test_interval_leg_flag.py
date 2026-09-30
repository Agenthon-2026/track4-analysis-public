"""A classification unit may DECLARE its interval leg off; nothing turns it off silently.

Adopted 2026-09-24. Some classification units carry a numeric ``y`` that is only a 0/1
encoding of the label (for example, a yes/no event recorded as 1 or 0). For them an interval around a label code
scores nothing meaningful, so they are scored on the label only: ``composite = w_a * quality``,
the way a pure-label unit is. The choice is a per-unit flag in the unit's card,

    [scoring.params]
    interval_leg = false

resolved through ``ScoringParams`` like every other scoring parameter. It is never inferred from
the data: a classification unit whose truth happens to be 0/1 and which does NOT declare the flag
keeps the interval leg. The interval is still required by the answer schema either way.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from qfbench2_track_analysis import scoring as S
from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import ScoringParams, UnitOutcome, score_unit

from .synthetic import SUPPORTING_TEXT, StubJudge, answer_for, build_unit, outcome_for

W_ACC, W_CAL = 0.7, 0.3
LABELS = ("beat", "miss", "miss")  # outcome_for's true labels
ZERO_ONE = (1.0, 0.0, 0.0)  # a 0/1 encoding of those labels


def _unit(
    tmp_path: pathlib.Path,
    *,
    flag: str | None,
    target_type: str = "classification",
    naive: bool = True,
) -> pathlib.Path:
    unit = build_unit(
        tmp_path, target_type=target_type, with_outcome=False, with_naive=naive
    )
    outcome = outcome_for()
    for row, y in zip(outcome["outcomes"], ZERO_ONE):
        row["y"] = y
    (unit / "reference").mkdir(exist_ok=True)
    (unit / "reference" / "outcome.json").write_text(
        json.dumps(outcome), encoding="utf-8"
    )
    if flag is not None:
        card = unit / "card.toml"
        text = card.read_text(encoding="utf-8")
        anchor = "tau_citation           = 0.5\n"
        assert text.count(anchor) == 1
        card.write_text(
            text.replace(anchor, anchor + f"interval_leg           = {flag}\n"),
            encoding="utf-8",
        )
    return unit


def _answer(target_type: str = "classification") -> dict[str, Any]:
    answer = answer_for(
        lo=-0.05, hi=0.05
    )  # a sharp band at 0: misses the one y = 1 row
    answer["target_type"] = target_type
    for row, label in zip(answer["entity_predictions"], LABELS):
        row["label"] = label
    return answer


def _score(tmp_path: pathlib.Path, unit: pathlib.Path, **ctx: Any) -> UnitOutcome:
    out = tmp_path / "res"
    out.mkdir(parents=True, exist_ok=True)
    (out / "answer.json").write_text(json.dumps(_answer()), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out, **ctx},
        judge=StubJudge((SUPPORTING_TEXT,)),
        judge_provenance=provenance,
    )


def test_a_declared_exemption_scores_the_label_only(tmp_path: pathlib.Path) -> None:
    unit = _unit(tmp_path, flag="false")
    outcome = _score(tmp_path, unit)
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["predictive_quality"] == 1.0
    assert outcome.diagnostics["interval_quality"] is None
    assert outcome.diagnostics["interval_coverage"] is None
    # 5.2.0: no interval leg -> the composite is the prediction leg alone, not W_ACC x it.
    assert outcome.score == pytest.approx(1.0)


def test_a_zero_one_truth_unit_without_the_flag_still_gets_the_leg(
    tmp_path: pathlib.Path,
) -> None:
    """The exemption is a declared choice, never inferred from 0/1-looking truth."""
    unit = _unit(tmp_path, flag=None)
    outcome = _score(tmp_path, unit)
    iq = outcome.diagnostics["interval_quality"]
    assert iq is not None and 0.0 < iq < 1.0
    assert outcome.score == pytest.approx(W_ACC * 1.0 + W_CAL * iq, abs=1e-12)
    assert outcome.score != pytest.approx(W_ACC * 1.0)


def test_an_explicit_true_is_the_default(tmp_path: pathlib.Path) -> None:
    flagged = _score(tmp_path / "t", _unit(tmp_path / "t", flag="true"))
    default = _score(tmp_path / "d", _unit(tmp_path / "d", flag=None))
    assert flagged.score == default.score
    assert flagged.diagnostics["interval_quality"] is not None


def test_an_exempt_unit_still_needs_its_naive_rule(tmp_path: pathlib.Path) -> None:
    """5.2.0: the prediction leg of every type is anchored to the declared naive rule."""
    unit = _unit(tmp_path, flag="false", naive=False)
    with pytest.raises(T4OrganizerFault, match="naive_answer.json"):
        _score(tmp_path, unit)


@pytest.mark.parametrize("target_type", ["regression", "ranking"])
def test_the_exemption_is_refused_off_classification(
    tmp_path: pathlib.Path, target_type: str
) -> None:
    """A numeric forecast's interval is always scored; only a label code may opt out."""
    unit = _unit(tmp_path, flag="false", target_type=target_type)
    with pytest.raises(T4OrganizerFault, match="interval_leg"):
        _score(tmp_path, unit)


@pytest.mark.parametrize("bad", ['"false"', "0", '"no"'])
def test_a_non_boolean_flag_is_refused(tmp_path: pathlib.Path, bad: str) -> None:
    unit = _unit(tmp_path, flag=bad)
    with pytest.raises(T4OrganizerFault, match="interval_leg"):
        _score(tmp_path, unit)


def test_a_card_that_disagrees_with_the_trusted_params_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """With trusted params supplied, the mounted card is only a cross-check."""
    unit = _unit(tmp_path, flag="false")
    trusted = {
        "target_type": "classification",
        "interval_level": 0.90,
        "faithfulness_threshold": 0.80,
        "tau_citation": 0.5,
        "composite_weights": (0.7, 0.3),
    }
    with pytest.raises(T4OrganizerFault, match="interval_leg"):
        _score(tmp_path, unit, scoring_params=trusted)
    agreeing = _score(
        tmp_path / "ok", unit, scoring_params={**trusted, "interval_leg": False}
    )
    assert agreeing.score == pytest.approx(1.0)  # 5.2.0 "R": the prediction leg alone


def test_scoring_params_default_keeps_the_leg() -> None:
    params = ScoringParams.from_sources(trusted=None, card_params=None)
    assert params.interval_leg is True


def test_a_missing_naive_file_message_names_the_actual_target_type(
    tmp_path: pathlib.Path,
) -> None:
    """The 'no readable naive file' refusal used to say 'regression unit' for every type."""
    unit = _unit(tmp_path, flag=None, naive=False)
    (unit / "reference" / "naive_answer.json").mkdir()  # a directory, not a file
    with pytest.raises(T4OrganizerFault, match="classification unit") as excinfo:
        _score(tmp_path, unit)
    assert not str(excinfo.value).startswith("regression unit")


def test_the_scorer_version_is_adopted() -> None:
    # 5.1.1 = 5.1.0 plus the two citation guards (test_citation_span_guards.py); 5.1.2 = the
    # review fixes (test_scorer_5_1_2_fixes.py); 5.1.3 = the narrowed own-value exemption
    # (test_own_value_exemption.py); 5.2.0 = the per-claim faithfulness penalty
    # (test_claim_penalty.py); 5.2.1 = the interval leg capped by the prediction leg
    # (test_interval_cap.py).
    assert S.SCORER_VERSION == "5.2.1"
