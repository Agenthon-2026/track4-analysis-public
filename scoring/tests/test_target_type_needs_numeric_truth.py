"""A regression or ranking unit whose outcome carries no numeric truth is an organizer fault.

The defect. `_true_vectors` decides "pure-label unit" from the OUTCOME
alone — no row carries a numeric `y`, so the calibration leg is dropped — and never consults the
declared `target_type`. A regression or ranking unit whose reference lost its `y` values is
therefore accepted as pure-label, and the effect is to RAISE what a worthless submission scores,
because the miscoverage penalty it would otherwise pay disappears.

Measured end to end on a synthetic four-entity unit, submitting an all-zero forecast with
zero-width intervals — an answer with no content at all:

    target_type   outcome has y   composite   predictive_quality
    regression    yes             -0.2700     0.000
    regression    NO              +0.0000     0.000
    ranking       yes             +0.0800     0.500
    ranking       NO              +0.3500     0.500

Both no-`y` rows beat the pre-committed worst value W = -0.27, and the ranking case more than
quadruples. `predictive_quality` returning 0.5 for a ranking unit with nothing to rank against is
CORRECT and documented ("a unit with fewer than two rows scores 0.5 — neither rewarded nor
punished for saying nothing"); the bug is that such a unit is reachable at all when the task
declares itself a ranking. The fix is deliberately scoped to the target-type cross-check ALONE,
here in Track 4's own code. The shared `predictive_quality` n<2 rule is not touched, and once a
ranking unit with non-numeric truth is refused, the ranking metric never sees that vector.

The guard must stay narrow in the other direction too: a `classification` unit whose outcome
carries no numeric `y` is the legitimate pure-label anatomy, and it must keep being accepted.
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


@pytest.mark.parametrize("target_type", ["regression", "ranking"])
def test_numeric_target_type_with_no_numeric_truth_is_refused(target_type: str) -> None:
    """THE GUARD. A unit that declares a numeric target must carry one."""
    with pytest.raises(T4OrganizerFault) as excinfo:
        _true_vectors(_outcome(None, None, None), ROSTER, target_type=target_type)
    message = str(excinfo.value)
    assert target_type in message
    assert "numeric" in message


def test_classification_with_no_numeric_truth_is_still_a_pure_label_unit() -> None:
    """POSITIVE CONTROL, and the reason this is a cross-check and not a blanket refusal.

    The legitimate pure-label anatomy must keep working: the calibration leg is dropped, the unit
    is scored on accuracy alone, and nothing is refused. A direction-of-change unit whose target
    is a class and whose outcome carries no `y` is exactly this shape.
    """
    labels, values = _true_vectors(
        _outcome(None, None, None), ROSTER, target_type="classification"
    )
    assert labels == ["up", "up", "up"]
    assert values is None


@pytest.mark.parametrize("target_type", ["classification", "regression", "ranking"])
def test_a_fully_numeric_outcome_is_unaffected_for_every_target_type(
    target_type: str,
) -> None:
    """POSITIVE CONTROL. The guard must not touch the ordinary unit under any target type."""
    labels, values = _true_vectors(
        _outcome(1.0, 2.0, 3.0), ROSTER, target_type=target_type
    )
    assert labels == ["up", "up", "up"]
    assert values == [1.0, 2.0, 3.0]


def test_the_mixed_outcome_fault_still_fires_first() -> None:
    """A mixed outcome stays a fault about mixing, not about the target type."""
    with pytest.raises(T4OrganizerFault) as excinfo:
        _true_vectors(_outcome(1.0, None, 3.0), ROSTER, target_type="regression")
    assert "pure-label" in str(excinfo.value)
