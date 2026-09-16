"""Contract tests against the hub's frozen C1 — the plan is the denominator and the parameters.

Three things are pinned here.

**The frozen numbers agree.** ``plan.failure_score_for(code)`` and ``plan.clip(score)`` are the two
functions the freeze says every track calls; this asserts Track 4's local ``DOMAIN_MIN`` and
``clip_to_domain`` produce exactly the same values, so there is no second opinion about ``W``.

**The shipped golden fixture is scoreable.** It used to carry ``target_type:
"point_and_interval"``, which is not a Track-4 target type, so the adapter refused it. C1 1.2.0
closes ``scoring_params.target_type`` to ``classification | regression | ranking`` and closes
``composite_weights`` to exactly ``accuracy`` + ``calibration`` summing to 1, and the fixture is
re-minted (unit 1 classification, unit 2 regression). The negative assertion that recorded the gap
is deleted; the positive one below replaces it.

**Both layers refuse a loose plan, and the test says so.** The refusal now lives in two places: the
hub's C1 parser rejects the document at construction, and Track 4's ``plan_adapter`` rejects the
entry at adaptation. They are not redundant — the adapter is what a caller reaches through when it
is handed a ``RosterEntry`` from anywhere other than a freshly parsed plan — so each negative
control below exercises **both**, and neither can be deleted on the strength of the other.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from qfbench2_common.contracts import (
    COMPOSITE_WEIGHT_KEYS,
    TARGET_TYPES as HUB_TARGET_TYPES,
    ContractError,
    EvaluationPlan,
    RosterEntry,
    sign_payload,
)
from qfbench2_common.contracts.fixtures import DEV_KEY_ID, DEV_SEED, load_fixture

from qfbench2_track_analysis.alignment import TARGET_TYPES, align_predictions
from qfbench2_track_analysis.codes import (
    T4OrganizerFault,
    T4ParticipantFailure,
    T4Reason,
)
from qfbench2_track_analysis.plan_adapter import (
    WEIGHT_KEYS,
    entity_roster_from_entry,
    labels_from_entry,
    scoring_params_from_entry,
    trusted_inputs_for,
)
from qfbench2_track_analysis.scoring import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    ScoringParams,
    clip_to_domain,
)

from .synthetic import answer_for  # noqa: E402

FIXTURE = "c1/analysis_final.expanded.json"


def _plan(**overrides: object) -> EvaluationPlan:
    """A variant of the golden C1, re-signed with the published dev key.

    Re-signing rather than patching `payload_digest` by hand: the plan verifies its own envelope
    against the canonical body at parse time, so a fixture edited without re-signing is refused —
    which is the binding working, not an obstacle to route around.
    """
    return EvaluationPlan(_raw(**overrides))


def _raw(**overrides: object) -> dict[str, Any]:
    raw = copy.deepcopy(load_fixture(FIXTURE))
    for entry in raw["roster"]["expected_units"]:
        entry["scoring_params"].update(overrides)
    return _resign(raw)


def _entry(**overrides: object) -> RosterEntry:
    """The fixture's first roster entry as a `RosterEntry`, built WITHOUT the C1 parser.

    The point is to reach `plan_adapter` with parameters the parser would have refused. A caller
    holding a `RosterEntry` has not necessarily just parsed a plan, and Track 4's own refusal is
    the layer that covers that caller.
    """
    raw = copy.deepcopy(load_fixture(FIXTURE))["roster"]["expected_units"][0]
    params = dict(raw["scoring_params"])
    params.update(overrides)
    return RosterEntry(
        unit_handle=raw["unit_handle"],
        entity_roster=raw["entity_roster"],
        scoring_params=params,
    )


def _resign(raw: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in raw.items() if k != "signature"}
    envelope = sign_payload(
        body, seed=DEV_SEED, key_id=DEV_KEY_ID, signed_at=raw["signature"]["signed_at"]
    )
    raw["signature"] = {
        "alg": envelope.alg,
        "key_id": envelope.key_id,
        "payload_digest": envelope.payload_digest,
        "signed_at": envelope.signed_at,
        "signature": envelope.signature,
    }
    return raw


# --- the frozen numbers -------------------------------------------------------------------------
def test_the_plans_worst_case_is_track_fours_domain_minimum() -> None:
    plan = EvaluationPlan(load_fixture(FIXTURE))
    assert plan.failure_score_for("schema_invalid") == pytest.approx(DOMAIN_MIN)
    assert plan.failure_score_for("no_output") == pytest.approx(DOMAIN_MIN)


def test_plan_clip_and_local_clip_agree() -> None:
    plan = EvaluationPlan(load_fixture(FIXTURE))
    for value in (-9.0, -0.27, 0.0, 0.5, 1.0, 9.0):
        assert plan.clip(value) == pytest.approx(clip_to_domain(value))
    assert plan.clip(9.0) == pytest.approx(DOMAIN_MAX)


def test_every_expected_unit_stays_in_the_denominator() -> None:
    plan = EvaluationPlan(load_fixture(FIXTURE))
    assert plan.denominator == len(plan.expected_handles)
    # No public failure code is allowed to remove a unit from the roster.
    for code in (
        "no_output",
        "malformed_output",
        "schema_invalid",
        "incomplete_output",
        "cutoff_violation",
        "domain_gate_failed",
    ):
        assert plan.failure_score_for(code) == pytest.approx(DOMAIN_MIN)


# --- the two enums are ONE enum -----------------------------------------------------------------
def test_track_fours_target_types_are_exactly_c1s() -> None:
    """C1 1.2.0 closed this field to the set Track 4 already implemented. Pin them together.

    If the hub ever grows a fourth target type, this fails here rather than in the adapter, where
    the symptom would be a plan the scorer refuses for a reason the plan believes is legal.
    """
    assert tuple(TARGET_TYPES) == tuple(HUB_TARGET_TYPES)


def test_track_fours_weight_names_are_exactly_c1s() -> None:
    assert tuple(WEIGHT_KEYS) == tuple(COMPOSITE_WEIGHT_KEYS)


# --- the adapter ---------------------------------------------------------------------------------
def test_a_well_formed_analysis_entry_adapts() -> None:
    plan = _plan(target_type="classification")
    handle = plan.expected_handles[0]
    roster, params = trusted_inputs_for(plan, handle)
    assert roster.count == 2
    resolved = ScoringParams.from_sources(trusted=params, card_params=None)
    assert resolved.target_type == "classification"
    assert resolved.worst_case == pytest.approx(DOMAIN_MIN)


def test_the_shipped_c1_analysis_fixture_is_scoreable() -> None:
    """Replaces `..._is_not_yet_scoreable`, deleted when C1 1.2.0 re-minted the fixture.

    Every unit in the shipped golden plan adapts, unmodified and unresigned, to a roster and a
    parameter set Track 4 can score. This is the assertion the old one was a placeholder for.
    """
    plan = EvaluationPlan(load_fixture(FIXTURE))
    seen = []
    for handle in plan.expected_handles:
        roster, params = trusted_inputs_for(plan, handle)
        assert roster.count == len(roster.entity_ids) == 2
        resolved = ScoringParams.from_sources(trusted=params, card_params=None)
        assert resolved.target_type in TARGET_TYPES
        assert resolved.worst_case == pytest.approx(DOMAIN_MIN)
        seen.append(resolved.target_type)
    # The re-minted fixture deliberately shows that the enum has members rather than one value.
    assert seen == ["classification", "regression"]


def test_an_unknown_handle_is_an_organizer_fault_not_a_skip() -> None:
    plan = _plan(target_type="classification")
    with pytest.raises(T4OrganizerFault, match="not in the C1 roster"):
        trusted_inputs_for(plan, "u-00000000")


# --- negative controls, asserted at BOTH layers ---------------------------------------------------
# Each of these was impossible to state as a *refusal* while C1 accepted any string and any key
# set: the loose document was legal and only Track 4 objected. Now the hub refuses the document and
# Track 4 refuses the entry, and both halves are asserted so that removing either one reddens a
# test instead of quietly leaving one gate.
def test_an_unknown_target_type_is_refused_by_c1_and_by_the_adapter() -> None:
    with pytest.raises(ContractError, match="target_type"):
        EvaluationPlan(_raw(target_type="point_and_interval"))
    with pytest.raises(T4OrganizerFault, match="not one of Track 4's"):
        scoring_params_from_entry(_entry(target_type="point_and_interval"))


def test_weights_that_do_not_sum_to_one_are_refused_by_c1_and_by_the_adapter() -> None:
    bad = {"accuracy": 0.7, "calibration": 0.5}
    with pytest.raises(ContractError, match="not 1"):
        EvaluationPlan(_raw(target_type="classification", composite_weights=bad))
    with pytest.raises(T4OrganizerFault, match="sum to 1"):
        scoring_params_from_entry(
            _entry(target_type="classification", composite_weights=bad)
        )


def test_unexpected_weight_names_are_refused_by_c1_and_by_the_adapter() -> None:
    bad = {"w_a": 0.7, "w_c": 0.3}
    with pytest.raises(ContractError, match="must name exactly"):
        EvaluationPlan(_raw(target_type="classification", composite_weights=bad))
    with pytest.raises(T4OrganizerFault, match="must name exactly"):
        scoring_params_from_entry(
            _entry(target_type="classification", composite_weights=bad)
        )


def test_a_public_commitment_cannot_be_scored_against() -> None:
    """The public commitment carries counts and digests. Iterating it as a roster is the A01 shape."""
    raw = copy.deepcopy(load_fixture(FIXTURE))
    raw["roster"].pop("expected_units")
    plan = EvaluationPlan(_resign(raw))
    assert plan.is_public_commitment is True
    with pytest.raises(Exception):
        trusted_inputs_for(plan, "u-644dc0d6eda4da5f")


# ------------------------------------------------- C1 1.3.0: the plan entry carries the vocabulary
#
# `Agenthon2026#123` / private #89 F1. `align_predictions` refuses a label outside
# `roster.labels`; the local path fills that field from task.json, the platform path built the
# roster from the plan entry, and the entry carried no vocabulary -- so four of the ten deployed
# Development units were LABEL_INVALID locally and admissible on the platform. The entry now
# carries `scoring_params.labels` and the adapter hands it to the roster.


def test_a_classification_entry_carries_its_labels_into_the_platform_roster() -> None:
    entry = _entry(target_type="classification", labels=["beat", "miss", "inline"])
    assert labels_from_entry(entry) == ("beat", "miss", "inline")
    assert entity_roster_from_entry(entry).labels == ("beat", "miss", "inline")


def test_the_platform_path_now_refuses_a_label_outside_the_vocabulary() -> None:
    """The verdict the local path always gave, reproduced from the signed entry alone."""
    entry = _entry(target_type="classification", labels=["up", "down"])
    roster = entity_roster_from_entry(entry)
    answer = answer_for(roster.entity_ids, label="sideways")
    with pytest.raises(T4ParticipantFailure) as caught:
        align_predictions(
            answer, roster, target_type="classification", interval_level=0.9
        )
    assert caught.value.reason is T4Reason.LABEL_INVALID


def test_a_label_inside_the_vocabulary_aligns() -> None:
    """POSITIVE CONTROL: the vocabulary refuses outsiders, not members."""
    entry = _entry(target_type="classification", labels=["up", "down"])
    roster = entity_roster_from_entry(entry)
    answer = answer_for(roster.entity_ids, label="down")
    aligned = align_predictions(
        answer, roster, target_type="classification", interval_level=0.9
    )
    assert set(aligned.pred_labels) == {"down"}


def test_an_entry_without_labels_keeps_the_pre_1_3_0_behaviour() -> None:
    """A 1.2.0 plan: no vocabulary, no check -- the ambiguity is the document's, and the hub
    parser removes it for 1.3.0 documents. Recorded so the skip is a known state, not a surprise."""
    entry = _entry(target_type="classification")
    assert entry.scoring_params is not None and "labels" not in entry.scoring_params
    assert entity_roster_from_entry(entry).labels is None


@pytest.mark.parametrize("target", ["regression", "ranking"])
def test_labels_on_a_unit_that_has_no_labels_are_an_organizer_fault(
    target: str,
) -> None:
    entry = _entry(target_type=target, labels=["up", "down"])
    with pytest.raises(T4OrganizerFault, match="only classification units"):
        labels_from_entry(entry)


@pytest.mark.parametrize(
    "labels",
    [["up"], [], ["up", "up"], ["up", ""], ["up", " down"], ["up", 3], "up,down", None],
)
def test_a_malformed_vocabulary_on_a_signed_entry_is_an_organizer_fault(
    labels: object,
) -> None:
    entry = _entry(target_type="classification", labels=labels)
    with pytest.raises(T4OrganizerFault, match="two or more unique"):
        labels_from_entry(entry)
