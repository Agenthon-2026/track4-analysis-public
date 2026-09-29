"""A unit whose outcome carries no numeric target has no calibration leg.

The sealed scorer applies the same rule (`final_scorer.py` types `interval_cov` as
`float | None` and says so at its module docstring). The public scorer did not, and the remediation
made it worse in two directions at once: a NULL target was silently coerced to `0.0` — so coverage
was measured against an invented number — while a non-numeric one raised an organizer fault, which
would have refused a unit anatomy the sealed side scores happily.

Three cases, and the middle one is the reason a single boolean will not do.
"""

from __future__ import annotations

import pytest

from qfbench2_track_analysis.alignment import EntityRoster
from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.scoring import _true_vectors

ROSTER = EntityRoster(entity_ids=("E1", "E2", "E3"))


def _outcome(*ys: object) -> dict[str, object]:
    return {
        "outcomes": [
            {"entity_id": eid, "true_label": "up", "y": y}
            for eid, y in zip(ROSTER.entity_ids, ys)
        ]
    }


def test_every_row_numeric_keeps_the_calibration_leg() -> None:
    """POSITIVE CONTROL. The ordinary unit is unaffected by any of this."""
    labels, values = _true_vectors(
        _outcome(1.0, 2.0, 3.0), ROSTER, target_type="regression"
    )
    assert labels == ["up", "up", "up"]
    assert values == [1.0, 2.0, 3.0]


def test_no_row_numeric_is_a_pure_label_unit_not_an_error() -> None:
    """`None` for the value vector means: drop the calibration leg, do not abort.

    `target_type="classification"` here is load-bearing, not incidental: only a classification
    unit may be pure-label, and the same outcome under `regression` or `ranking`
    is an organizer fault (`test_target_type_needs_numeric_truth.py`). Do not generalise this
    call back to a numeric target type.

    Before the fix this path returned `[0.0, 0.0, 0.0]` and coverage was computed against zeros —
    a measured-looking number produced from nothing.
    """
    labels, values = _true_vectors(
        _outcome(None, None, None), ROSTER, target_type="classification"
    )
    assert labels == ["up", "up", "up"]
    assert values is None


def test_a_mixed_outcome_is_an_organizer_fault() -> None:
    """Neither answer is defensible, so refuse rather than pick one.

    Scoring coverage over only the numeric rows shrinks the denominator, which is forbidden;
    counting an absent target as a miss invents a result. A file where some entities carry a
    numeric target and others do not is malformed, and that is an organizer problem.
    """
    with pytest.raises(T4OrganizerFault, match="numeric target for 2 of 3"):
        _true_vectors(_outcome(1.0, 2.0, None), ROSTER, target_type="regression")


@pytest.mark.parametrize("bad", ["1.0", True, [], {}])
def test_a_non_numeric_target_is_still_refused(bad: object) -> None:
    """Absent means pure-label; a string or a bool means malformed. Do not guess between them."""
    with pytest.raises(T4OrganizerFault, match="neither a number nor absent"):
        _true_vectors(_outcome(bad, bad, bad), ROSTER, target_type="regression")


def test_a_nonfinite_target_is_an_organizer_fault() -> None:
    """A NaN in the ANSWER KEY must not be charged to the participant.

    NaN is numeric by type and useless as a target. Scored as-is, `lo <= nan <= hi` is False, so
    the row counts as a coverage miss and the submission pays for a defect in reference material.
    The frozen rule sends a nonfinite value in organizer data to an organizer fault.
    """
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(T4OrganizerFault, match="nonfinite numeric target"):
            _true_vectors(_outcome(bad, 2.0, 3.0), ROSTER, target_type="regression")
