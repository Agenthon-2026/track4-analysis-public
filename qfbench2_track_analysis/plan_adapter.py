"""Adapt a C1 roster entry into the trusted roster and scoring parameters Track 4 scores with.

C1 1.1.0 grants Track 4 exactly what it asked for: ``expected_units[].entity_roster`` (ordered ids
plus count and digest) and ``expected_units[].scoring_params``. This module is the consumer side —
the one place a signed plan becomes an :class:`~qfbench2_track_analysis.alignment.EntityRoster` and
a :class:`~qfbench2_track_analysis.scoring.ScoringParams`, so no other module has to know the
plan's field names.

It fails closed on two places where the C1 *shape* used to be looser than Track 4's semantics. As
of C1 1.2.0 it is not: ``TARGET_TYPES`` and ``COMPOSITE_WEIGHT_KEYS`` are closed in the hub's
parser, and the golden fixture is re-minted onto real members of the enum. **The checks below
stay** — the hub validates a *document* at parse time, this validates an *entry*, and a caller
holding a :class:`~qfbench2_common.contracts.RosterEntry` from anywhere other than a freshly parsed
plan reaches only this one. ``scoring/tests/test_plan_contract.py`` asserts both layers refuse, so
neither can be removed on the strength of the other.

* ``scoring_params.target_type`` is a **closed three-value enum** (``classification | regression |
  ranking``), identical to the hub's. An unrecognized value is refused rather than defaulted,
  which is the whole point of T4-3: the pre-fix scorer fell back to label accuracy.
* ``scoring_params.composite_weights`` names exactly ``accuracy`` and ``calibration``, summing
  to 1, because ``W = -w_calibration * interval_level`` is the frozen worst case and a plan whose
  weights do not sum to 1 silently moves the domain the leaderboard is clipped into.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qfbench2_common.contracts import EvaluationPlan, RosterEntry

from .alignment import TARGET_TYPES, EntityRoster, validated_label_vocabulary
from .codes import T4OrganizerFault

__all__ = [
    "WEIGHT_KEYS",
    "entity_roster_from_entry",
    "labels_from_entry",
    "scoring_params_from_entry",
    "trusted_inputs_for",
]

#: The two weight names Track 4's composite is defined over, in composite order.
WEIGHT_KEYS = ("accuracy", "calibration")


def entity_roster_from_entry(entry: RosterEntry) -> EntityRoster:
    """The expanded roster's ordered entity ids. A count-and-digest-only entry cannot be scored."""
    raw = entry.entity_roster
    if not isinstance(raw, Mapping):
        raise T4OrganizerFault(
            f"C1 entry {entry.unit_handle!r} carries no entity_roster; Track 4's denominator is "
            "per entity and a unit-level roster does not reach it"
        )
    ids = raw.get("entity_ids")
    if not isinstance(ids, list) or not ids:
        raise T4OrganizerFault(
            f"C1 entry {entry.unit_handle!r} carries only the entity_roster commitment "
            "(count + digest). Scoring needs the expanded form; the public commitment is not a "
            "roster a scorer can iterate."
        )
    count = raw.get("count")
    if count != len(ids):
        raise T4OrganizerFault(
            f"C1 entry {entry.unit_handle!r}: entity_roster.count disagrees with entity_ids"
        )
    return EntityRoster(
        entity_ids=tuple(str(x) for x in ids), labels=labels_from_entry(entry)
    )


def labels_from_entry(entry: RosterEntry) -> tuple[str, ...] | None:
    """The unit's label vocabulary from the signed plan (C1 1.3.0, `scoring_params.labels`).

    This is the platform half of `Agenthon2026#123` / private #89 F1. `align_predictions` refuses
    a submitted label outside `roster.labels`; the local path fills that field from `task.json`
    (`EntityRoster.from_task`), the platform path built the roster from the plan entry, and the
    entry did not carry the vocabulary -- so the check ran locally and silently did not run on
    the platform. Same submission, two verdicts.

    A 1.2.0 plan still carries no `labels`; then this returns None and the check stays skipped for
    that unit, exactly as before -- the ambiguity is the plan's, and the hub parser removes it for
    1.3.0 documents (classification => required, regression/ranking => forbidden). What is present
    is validated here too: a caller holding a `RosterEntry` from anywhere but a freshly parsed
    plan reaches only this layer.
    """
    params = entry.scoring_params
    if not isinstance(params, Mapping) or "labels" not in params:
        return None
    return validated_label_vocabulary(
        params["labels"],
        params.get("target_type"),
        source=f"C1 entry {entry.unit_handle!r}: scoring_params.labels",
    )


def scoring_params_from_entry(entry: RosterEntry) -> dict[str, Any]:
    """The trusted per-unit scoring parameters, in the shape `ScoringParams.from_sources` takes."""
    raw = entry.scoring_params
    if not isinstance(raw, Mapping):
        raise T4OrganizerFault(
            f"C1 entry {entry.unit_handle!r} carries no scoring_params; reading them from the "
            "mounted card instead would take them from the directory the participant's container "
            "was given"
        )
    target_type = raw.get("target_type")
    if target_type not in TARGET_TYPES:
        raise T4OrganizerFault(
            f"C1 entry {entry.unit_handle!r} declares target_type {target_type!r}, which is not "
            f"one of Track 4's {list(TARGET_TYPES)}. An unrecognized target type is refused: the "
            "pre-fix scorer fell back to label accuracy, so a plan typo silently changed which "
            "metric the leaderboard reported."
        )
    weights = raw.get("composite_weights")
    if not isinstance(weights, Mapping) or set(weights) != set(WEIGHT_KEYS):
        raise T4OrganizerFault(
            f"C1 entry {entry.unit_handle!r}: composite_weights must name exactly "
            f"{list(WEIGHT_KEYS)}"
        )
    pair = (float(weights["accuracy"]), float(weights["calibration"]))
    if abs(sum(pair) - 1.0) > 1e-9:
        raise T4OrganizerFault(
            f"C1 entry {entry.unit_handle!r}: composite_weights must sum to 1; W is "
            "-w_calibration * interval_level and a different sum moves the frozen domain"
        )
    return {
        "target_type": str(target_type),
        "interval_level": float(raw["interval_level"]),
        "faithfulness_threshold": float(raw["faithfulness_threshold"]),
        "tau_citation": float(raw["tau_citation"]),
        "composite_weights": pair,
    }


def trusted_inputs_for(
    plan: EvaluationPlan, unit_handle: str
) -> tuple[EntityRoster, dict[str, Any]]:
    """Look one unit up in the signed plan. An unknown handle is an organizer fault, not a skip."""
    if plan.track != "analysis":
        raise T4OrganizerFault(f"plan is for track {plan.track!r}, not analysis")
    for entry in plan.expected_units:
        if entry.unit_handle == unit_handle:
            return entity_roster_from_entry(entry), scoring_params_from_entry(entry)
    raise T4OrganizerFault(
        f"unit handle {unit_handle!r} is not in the C1 roster. Scoring a unit the plan does not "
        "commit to would put a row on the leaderboard that no denominator accounts for."
    )
