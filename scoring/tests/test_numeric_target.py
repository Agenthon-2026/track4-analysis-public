"""The numeric-target rule, pinned at VERDICT level (mirror of the sealed scorer's behaviour).

The rule (three-way since 2026-08-22): every roster row numeric -> ordinary unit;
no row numeric -> pure-label unit with no calibration leg; mixed -> refused as an organizer
fault, as is a non-finite or non-numeric target. `test_pure_label_units.py` pins the same rule
at the `_true_vectors` helper level; this module pins what a participant actually receives from
`score_unit` — the composite, the nullable `interval_coverage`, and the fault propagation —
because the original defect was end-to-end: `_score` read ``float(out.get("y", 0.0) or 0.0)``
and scored every interval on a pure-label unit against an invented target of zero.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any

import pytest

from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import UnitOutcome, score_unit

from .synthetic import SUPPORTING_TEXT, StubJudge, answer_for, build_unit, outcome_for

_TRUE_LABELS = (
    "beat",
    "miss",
    "miss",
)  # what outcome_for resolves for the three entities
W_ACC, W_CAL, LEVEL = (
    0.7,
    0.3,
    0.90,
)  # the synthetic card's composite_weights / interval_level


def _answer(labels: tuple[str, str, str]) -> dict[str, Any]:
    """`answer_for` with per-entity labels instead of one label for every row."""
    answer = answer_for()
    for row, label in zip(answer["entity_predictions"], labels):
        row["label"] = label
    return answer


def _score(
    tmp_path: pathlib.Path, ys: tuple[object, ...], answer: dict[str, Any]
) -> UnitOutcome:
    unit = build_unit(tmp_path, with_outcome=False, with_naive=True)
    outcome = outcome_for()
    for row, y in zip(outcome["outcomes"], ys):
        row["y"] = y
    (unit / "reference").mkdir(exist_ok=True)
    (unit / "reference" / "outcome.json").write_text(
        json.dumps(outcome), encoding="utf-8"
    )
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out},
        judge=StubJudge((SUPPORTING_TEXT,)),
        judge_provenance=provenance,
    )


def test_numeric_roster_scores_with_calibration_over_the_full_roster(
    tmp_path: pathlib.Path,
) -> None:
    """POSITIVE CONTROL: an ordinary numeric unit keeps its calibration leg end to end."""
    outcome = _score(tmp_path, (1.0, 2.0, 3.0), _answer(_TRUE_LABELS))
    assert outcome.state == "participant_success"
    # answer_for's default interval [0.5, 3.5] covers y = 1.0, 2.0 and 3.0 -> coverage 1.0
    assert outcome.diagnostics["interval_coverage"] == pytest.approx(1.0)
    # 5.1.0: the interval leg is w_c * iq; the answer's band equals the naive band -> iq 0.5.
    assert outcome.diagnostics["interval_quality"] == 0.5
    assert outcome.score == pytest.approx(W_ACC * 1.0 + W_CAL * 0.5)


def test_pure_label_roster_has_no_calibration_leg(tmp_path: pathlib.Path) -> None:
    """No numeric target anywhere -> no interval leg, coverage reported as None.

    None rather than 0.0, so a reader can tell "not applicable" from "measured, and it was
    zero". From 5.2.0 the composite is the anchored prediction leg alone, so a perfect
    answer scores 1.0 (the old w_a = 0.7 ceiling is gone).
    """
    outcome = _score(tmp_path, (None, None, None), _answer(_TRUE_LABELS))
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["interval_coverage"] is None
    assert outcome.score == pytest.approx(1.0)


def test_pure_label_wrong_label_is_still_scored(tmp_path: pathlib.Path) -> None:
    """A wrong prediction on a pure-label unit is a LOW SCORE, not a failure state."""
    outcome = _score(tmp_path, (None, None, None), _answer(("inline", "beat", "beat")))
    assert outcome.state == "participant_success"
    assert outcome.score == pytest.approx(0.0)


def test_mixed_roster_is_refused_as_organizer_fault(tmp_path: pathlib.Path) -> None:
    """Some rows numeric, some not: refuse rather than shrink the denominator or invent a y.

    The fault must PROPAGATE out of `score_unit` — the frozen C1 policy says an organizer
    fault aborts the evaluation and never becomes a participant zero.
    """
    with pytest.raises(T4OrganizerFault, match="numeric target for 2 of 3"):
        _score(tmp_path, (1.0, 2.0, None), _answer(_TRUE_LABELS))


def test_non_finite_target_is_refused(tmp_path: pathlib.Path) -> None:
    """NaN in the ANSWER KEY: scored as-is, `lo <= nan <= hi` is False and the participant
    pays for a defect in reference material. The frozen rule makes it an organizer fault."""
    with pytest.raises(T4OrganizerFault, match="nonfinite numeric target"):
        _score(tmp_path, (math.nan, 2.0, 3.0), _answer(_TRUE_LABELS))


def test_non_numeric_target_is_refused(tmp_path: pathlib.Path) -> None:
    """ "1.25" as a JSON string used to read as a 0.700 composite from a typo."""
    with pytest.raises(T4OrganizerFault, match="neither a number nor absent"):
        _score(tmp_path, ("1.25", "2.0", "3.0"), _answer(_TRUE_LABELS))
