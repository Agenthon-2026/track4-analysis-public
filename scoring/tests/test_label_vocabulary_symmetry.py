"""The two roster paths agree about what a label vocabulary is (#20, follow-on to #18).

`EntityRoster.labels` is built from two sources: `EntityRoster.from_task` reads `task.json` on the
local path, and `plan_adapter.labels_from_entry` reads the signed C1 entry on the platform path.
#18 made the plan path strict -- two or more unique, non-empty, unpadded strings, classification
only -- and left the local path as it was: no count, no uniqueness, no whitespace, no target type,
and a fall-through to `labels = None` that SKIPS the label check when the shape is wrong.

That is the same defect #18 closed, pointed the other way. `task.json` is organizer material too,
so a malformed vocabulary there is an organizer fault, not a check that quietly does not run.

These tests hand the SAME vocabulary to BOTH paths and assert they reach the same verdict. That is
the point of the file: the symmetry is a property of the code, not of two reviews agreeing.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from qfbench2_common.contracts import RosterEntry
from qfbench2_common.contracts.fixtures import load_fixture

from qfbench2_track_analysis.alignment import EntityRoster
from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.plan_adapter import labels_from_entry

FIXTURE = "c1/analysis_final.expanded.json"

#: Every shape #18 refuses on the plan side. Each row is a control: before this change the local
#: path accepted the first four and silently skipped the check on the last four.
MALFORMED: tuple[object, ...] = (
    ["up"],
    [],
    ["up", "up"],
    ["up", ""],
    ["up", " down"],
    ["up", 3],
    "up,down",
    None,
)


def _entry(*, target_type: str, labels: object = ...) -> RosterEntry:
    """A C1 roster entry carrying this vocabulary, built without the hub parser."""
    raw = copy.deepcopy(load_fixture(FIXTURE))["roster"]["expected_units"][0]
    params = dict(raw["scoring_params"])
    params["target_type"] = target_type
    if labels is not ...:
        params["labels"] = labels
    return RosterEntry(
        unit_handle=raw["unit_handle"],
        entity_roster=raw["entity_roster"],
        scoring_params=params,
    )


def _task(*, target_type: str, labels: object = ...) -> dict[str, Any]:
    """The same vocabulary as a unit's `task.json` would carry it."""
    target: dict[str, Any] = {"type": target_type}
    if labels is not ...:
        target["labels"] = labels
    return {
        "entities": [{"entity_id": "e1"}, {"entity_id": "e2"}],
        "target": target,
    }


@pytest.mark.parametrize("labels", MALFORMED, ids=repr)
def test_a_malformed_vocabulary_is_refused_on_both_paths(labels: object) -> None:
    """THE SYMMETRY TEST. One vocabulary, two builders, one verdict."""
    with pytest.raises(T4OrganizerFault):
        labels_from_entry(_entry(target_type="classification", labels=labels))
    with pytest.raises(T4OrganizerFault):
        EntityRoster.from_task(_task(target_type="classification", labels=labels))


@pytest.mark.parametrize("target_type", ["regression", "ranking"])
def test_a_vocabulary_on_a_unit_that_has_none_is_refused_on_both_paths(
    target_type: str,
) -> None:
    with pytest.raises(T4OrganizerFault, match="only classification units"):
        labels_from_entry(_entry(target_type=target_type, labels=["up", "down"]))
    with pytest.raises(T4OrganizerFault, match="only classification units"):
        EntityRoster.from_task(_task(target_type=target_type, labels=["up", "down"]))


def test_a_well_formed_vocabulary_is_accepted_on_both_paths() -> None:
    """POSITIVE CONTROL: the rule refuses malformed vocabularies, not vocabularies."""
    assert labels_from_entry(
        _entry(target_type="classification", labels=["up", "down"])
    ) == ("up", "down")
    assert EntityRoster.from_task(
        _task(target_type="classification", labels=["up", "down"])
    ).labels == ("up", "down")


def test_both_paths_refuse_a_malformed_vocabulary_with_the_same_sentence() -> None:
    """One validator, one error text -- so the two cannot drift into two rules again."""
    with pytest.raises(T4OrganizerFault) as from_entry:
        labels_from_entry(_entry(target_type="classification", labels=["up"]))
    with pytest.raises(T4OrganizerFault) as from_task:
        EntityRoster.from_task(_task(target_type="classification", labels=["up"]))
    shared = "two or more unique non-empty strings"
    assert shared in str(from_entry.value)
    assert shared in str(from_task.value)


def test_a_task_that_declares_no_vocabulary_is_unchanged() -> None:
    """The six regression and ranking units: no `labels` key, no check, no fault.

    This is the pre-existing behaviour and it is the one the change must NOT touch, so it is
    recorded rather than left to inference.
    """
    assert EntityRoster.from_task(_task(target_type="regression")).labels is None
    assert EntityRoster.from_task(_task(target_type="classification")).labels is None
    assert labels_from_entry(_entry(target_type="classification")) is None


def test_a_task_with_no_target_at_all_is_unchanged() -> None:
    """`target` is optional on the local path today; absence is not a malformed vocabulary."""
    task = {"entities": [{"entity_id": "e1"}]}
    assert EntityRoster.from_task(task).labels is None
