"""Track 4 official scorer: one implementation, mandatory judge, fixed roster, no perfect defaults.

This module is the **single** Track-4 scoring implementation. The public official verifier, the
public smoke preview and the sealed private final scorer all reach the composite through
:func:`score_unit` here, so the three cannot drift the way three copies did — one that never built
a judge, one that fell back to a lexical stub on any exception, and one that had no schema gate at
all, disagreeing on cutoff polarity, target-type handling and interval validation.

What changed, and what each change closes:

* **The judge is mandatory in every rankable factory.** :func:`build_verifier` constructs the
  pinned production ensemble or raises an organizer fault. There is no ``ctx.get("judge")``, no
  skipped faithfulness block, and above all no ``ctx.get("_faithfulness", 1.0)`` — a missing judge
  used to mean *perfect* faithfulness. :func:`build_smoke_verifier` is a separately named factory
  that stamps ``judge_mode="smoke"`` and ``rankable=False``.
* **Evidence is bound to the prediction.** The entailment hypothesis is derived from the submitted
  label/forecast/rank/interval, not from participant-authored prose (see ``hypothesis.py``).
* **The roster is the denominator.** Alignment is driven by the trusted entity roster in trusted
  order; missing, duplicate and unknown ids fail before any metric runs (see ``alignment.py``).
* **Citations resolve through a trusted dictionary.** No participant string is ever interpolated
  into a path, and an unresolved, undated or post-cutoff citation fails closed (see ``corpus.py``).
* **An inadmissible unit scores W, not ``None``.** ``None`` is what removed a unit from the
  aggregate; the frozen policy is a pre-committed worst value that stays in the denominator, and
  real scores are clipped into the same domain so failing can never beat participating.
* **Public detail is enum code plus counts.** No raw exception strings; the previous gate returned
  ``{"reason": str(e)}`` and produced a ~1.5 KB detail embedding the whole answer schema.

Domain and worst case (frozen): ``desc``, domain ``[-0.27, 1.0]``,
``W = 0*w_a - w_c*interval_level = -0.27`` at the default weights.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from qfbench2_common.failure_labels import FailureLabel
from qfbench2_common.scoring import faithfulness as F
from qfbench2_common.taskcard import schema_path
from qfbench2_common.verifier import GateResult, HierarchicalVerifier

from .alignment import AlignedPredictions, EntityRoster, TARGET_TYPES, align_predictions
from .codes import T4OrganizerFault, T4ParticipantFailure, T4Reason
from .corpus import CorpusIndex, parse_iso_date
from .hypothesis import HypothesisSpec, prediction_claims
from .judge_factory import JudgeProvenance, build_production_judge, build_smoke_judge

__all__ = [
    "DOMAIN_MAX",
    "DOMAIN_MIN",
    "LEADERBOARD_SORT",
    "SCORER_VERSION",
    "ScoringParams",
    "UnitOutcome",
    "build_smoke_verifier",
    "build_verifier",
    "clip_to_domain",
    "score_unit",
    "worst_case_score",
]

logger = logging.getLogger(__name__)

LEADERBOARD_SORT = "desc"

#: The frozen Track-4 metric domain. `W` is the minimum because the direction is `desc`.
DOMAIN_MIN = -0.27
DOMAIN_MAX = 1.0

#: Bumped whenever the composite, the gates or the evidence semantics change. Recorded in
#: provenance so a leaderboard can be attributed to an implementation rather than to a repo state.
SCORER_VERSION = "3.1.0"

_SCHEMA = schema_path("analysis.schema.json")


# --------------------------------------------------------------------------- #
# Trusted per-unit scoring parameters                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ScoringParams:
    """The per-unit scoring parameters, from the trusted side.

    These used to be read out of ``card.toml`` inside the unit directory — the same directory that
    is bind-mounted into the participant's container. They are contract data (C1
    ``expected_units[].*``) and the mounted card is only ever a cross-check: a card-derived value
    that disagrees with the plan is refused rather than preferred.
    """

    target_type: str
    interval_level: float
    faithfulness_threshold: float
    tau_citation: float
    composite_weights: tuple[float, float]

    def __post_init__(self) -> None:
        if self.target_type not in TARGET_TYPES:
            raise T4OrganizerFault(
                f"target_type {self.target_type!r} is not one of {list(TARGET_TYPES)}; an unknown "
                "target type is refused, never defaulted to classification"
            )
        if not 0.0 < self.interval_level < 1.0:
            raise T4OrganizerFault("interval_level must be in (0, 1)")
        if not 0.0 < self.faithfulness_threshold <= 1.0:
            raise T4OrganizerFault("faithfulness_threshold must be in (0, 1]")
        if not 0.0 <= self.tau_citation < 1.0:
            raise T4OrganizerFault("tau_citation must be in [0, 1)")
        w_a, w_c = self.composite_weights
        if abs((w_a + w_c) - 1.0) > 1e-9:
            raise T4OrganizerFault("composite_weights must sum to 1")

    @property
    def worst_case(self) -> float:
        """`W` for this unit: zero predictive quality and no valid coverage."""
        return worst_case_score(self.composite_weights, self.interval_level)

    @classmethod
    def from_sources(
        cls,
        *,
        trusted: Mapping[str, Any] | None,
        card_params: Mapping[str, Any] | None,
    ) -> ScoringParams:
        """Build from the trusted plan values, refusing a mounted-card value that disagrees."""
        base: dict[str, Any] = {
            "target_type": "classification",
            "interval_level": 0.90,
            "faithfulness_threshold": 0.80,
            "tau_citation": 0.5,
            "composite_weights": (0.7, 0.3),
        }
        if trusted:
            for key in base:
                if key in trusted:
                    base[key] = trusted[key]
        base["composite_weights"] = tuple(float(x) for x in base["composite_weights"])
        params = cls(
            target_type=str(base["target_type"]),
            interval_level=float(base["interval_level"]),
            faithfulness_threshold=float(base["faithfulness_threshold"]),
            tau_citation=float(base["tau_citation"]),
            composite_weights=(
                base["composite_weights"][0],
                base["composite_weights"][1],
            ),
        )
        if trusted and card_params:
            _refuse_card_disagreement(params, card_params)
        if not trusted and card_params:
            # No plan supplied (public practice path): the card is all there is, and it is
            # organizer-authored. It is still validated, and it is still never the source in a
            # sealed run, where `trusted` is always present.
            merged = dict(base)
            for key in (
                "target_type",
                "interval_level",
                "faithfulness_threshold",
                "tau_citation",
            ):
                if key in card_params:
                    merged[key] = card_params[key]
            if "composite_weights" in card_params:
                merged["composite_weights"] = tuple(
                    float(x) for x in card_params["composite_weights"]
                )
            params = cls(
                target_type=str(merged["target_type"]),
                interval_level=float(merged["interval_level"]),
                faithfulness_threshold=float(merged["faithfulness_threshold"]),
                tau_citation=float(merged["tau_citation"]),
                composite_weights=(
                    float(merged["composite_weights"][0]),
                    float(merged["composite_weights"][1]),
                ),
            )
        return params


def _refuse_card_disagreement(
    params: ScoringParams, card_params: Mapping[str, Any]
) -> None:
    checks: list[tuple[str, Any, Any]] = []
    for key in (
        "target_type",
        "interval_level",
        "faithfulness_threshold",
        "tau_citation",
    ):
        if key in card_params:
            checks.append((key, card_params[key], getattr(params, key)))
    if "composite_weights" in card_params:
        checks.append(
            (
                "composite_weights",
                tuple(float(x) for x in card_params["composite_weights"]),
                params.composite_weights,
            )
        )
    for key, from_card, from_plan in checks:
        same = (
            abs(float(from_card) - float(from_plan)) <= 1e-12
            if isinstance(from_plan, float)
            else from_card == from_plan
        )
        if isinstance(from_plan, tuple):
            same = tuple(float(x) for x in from_card) == tuple(
                float(x) for x in from_plan
            )
        if not same:
            raise T4OrganizerFault(
                f"the mounted card's {key} disagrees with the trusted plan. The card sits in the "
                "directory that was bind-mounted into the participant's container, so the plan "
                "wins and the disagreement is a staging fault rather than an override."
            )


def worst_case_score(
    composite_weights: tuple[float, float], interval_level: float
) -> float:
    """`W` = ``0*w_a - w_c*interval_level`` — the worst attainable composite for a unit."""
    return -float(composite_weights[1]) * float(interval_level)


def clip_to_domain(score: float) -> float:
    """Clamp a real score into the frozen domain, so failure is never better than participating."""
    return max(DOMAIN_MIN, min(DOMAIN_MAX, float(score)))


# --------------------------------------------------------------------------- #
# Result                                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class UnitOutcome:
    """One unit's outcome. `score` is ALWAYS a float in the frozen domain — never ``None``."""

    state: str
    score: float
    rankable: bool
    failure_code: str | None
    detail: dict[str, Any]
    labels: tuple[FailureLabel, ...]
    judge: dict[str, Any]
    #: Operator-only. Never serialized into a participant-visible artifact.
    diagnostics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Trusted context hydration                                                     #
# --------------------------------------------------------------------------- #
def _read_json(path: pathlib.Path, *, what: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise T4OrganizerFault(f"{what} is missing or is a link: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise T4OrganizerFault(f"{what} is unreadable or not JSON") from exc


def hydrate(ctx: dict[str, Any]) -> None:
    """Populate the trusted half of the context from `unit_dir` and any plan the caller supplied.

    Everything read here is organizer material. The participant's bytes are read in ``g0`` and
    nowhere else, and they are never used to decide which organizer file to open.
    """
    if ctx.get("_hydrated"):
        return
    unit = pathlib.Path(ctx["unit_dir"])

    card_path = unit / "card.toml"
    if "card" not in ctx:
        if card_path.is_symlink() or not card_path.is_file():
            raise T4OrganizerFault(f"unit {unit.name!r} has no readable card.toml")
        ctx["card"] = tomllib.loads(card_path.read_text(encoding="utf-8"))
    card_params = (ctx["card"].get("scoring", {}) or {}).get("params", {}) or {}

    task = _read_json(unit / "task.json", what="task.json")
    if not isinstance(task, Mapping):
        raise T4OrganizerFault("task.json is not a JSON object")
    ctx["_task"] = task

    # A signed C1 plan, when the caller has one, is the ONLY source of the roster and the scoring
    # parameters. It outranks both the mounted card and task.json, because both of those live in
    # the directory that was bind-mounted into the participant's container.
    plan = ctx.get("plan")
    trusted: Mapping[str, Any] | None = ctx.get("scoring_params")
    plan_roster: EntityRoster | None = None
    if plan is not None:
        from .plan_adapter import trusted_inputs_for

        handle = ctx.get("unit_handle")
        if not isinstance(handle, str) or not handle:
            raise T4OrganizerFault(
                "a C1 plan was supplied without a unit_handle; a plan cannot be applied to a unit "
                "whose roster entry has not been identified"
            )
        plan_roster, plan_params = trusted_inputs_for(plan, handle)
        if trusted is not None and dict(trusted) != plan_params:
            raise T4OrganizerFault(
                "the caller-supplied scoring_params disagree with the signed plan entry"
            )
        trusted = plan_params

    ctx["_params"] = ScoringParams.from_sources(
        trusted=trusted, card_params=card_params
    )

    roster = ctx.get("entity_roster")
    if plan_roster is not None:
        ctx["_roster"] = plan_roster
    elif isinstance(roster, EntityRoster):
        ctx["_roster"] = roster
    elif isinstance(roster, (list, tuple)) and roster:
        ctx["_roster"] = EntityRoster(entity_ids=tuple(str(x) for x in roster))
    else:
        ctx["_roster"] = EntityRoster.from_task(task)

    cutoff_raw = task.get("cutoff_date")
    if cutoff_raw is None:
        raise T4OrganizerFault(
            f"unit {unit.name!r} declares no cutoff_date. The embargo gate cannot be evaluated "
            "without one, and a unit that cannot be embargo-checked is an organizer fault, not a "
            "participant failure."
        )
    ctx["_cutoff"] = parse_iso_date(
        cutoff_raw, field="task.json.cutoff_date", fault="organizer"
    )

    ctx["_corpus"] = CorpusIndex.from_unit(unit)
    ctx["_hypothesis_spec"] = HypothesisSpec.from_task(
        task,
        target_type=ctx["_params"].target_type,
        interval_level=ctx["_params"].interval_level,
    )

    if "realized" not in ctx:
        outcome_path = unit / "reference" / "outcome.json"
        ctx["realized"] = (
            _read_json(outcome_path, what="reference/outcome.json")
            if outcome_path.exists()
            else None
        )
    ctx["_hydrated"] = True


# --------------------------------------------------------------------------- #
# Gates                                                                         #
# --------------------------------------------------------------------------- #
def _load_answer(output_dir: pathlib.Path) -> Mapping[str, Any]:
    path = pathlib.Path(output_dir) / "answer.json"
    if path.is_symlink():
        raise T4ParticipantFailure(
            T4Reason.MALFORMED_ANSWER, "answer.json is a link", invalid_row_count=1
        )
    if not path.is_file():
        raise T4ParticipantFailure(
            T4Reason.NO_ANSWER, "answer.json is missing", missing_count=1
        )
    try:
        answer = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T4ParticipantFailure(
            T4Reason.MALFORMED_ANSWER,
            "answer.json is not valid JSON",
            invalid_row_count=1,
        ) from exc
    if not isinstance(answer, Mapping):
        raise T4ParticipantFailure(
            T4Reason.MALFORMED_ANSWER,
            "answer.json is not a JSON object",
            invalid_row_count=1,
        )
    return answer


def _g0_integrity(ctx: dict[str, Any]) -> GateResult:
    hydrate(ctx)
    ctx["_answer"] = _load_answer(ctx["output_dir"])
    return GateResult(True)


def _g1_schema(ctx: dict[str, Any]) -> GateResult:
    """Validate against the shared analysis schema. A missing validator is an ORGANIZER fault.

    The previous implementation swallowed ``ModuleNotFoundError`` and passed. The interval-level
    check is the only thing on the public path that catches a wrong level, so that swallow silently
    removed a gate; global rule 7 says a missing dependency must fail, not report green.
    """
    try:
        import jsonschema
    except (
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - jsonschema is a declared dependency
        raise T4OrganizerFault(
            "jsonschema is not importable in the scoring environment, so the schema gate cannot "
            "run. A gate that cannot run is an organizer fault; it is never a pass."
        ) from exc
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors = list(validator.iter_errors(ctx["_answer"]))
    if errors:
        raise T4ParticipantFailure(
            T4Reason.SCHEMA_INVALID,
            "answer.json does not validate against the published analysis schema",
            invalid_row_count=len(errors),
        )
    return GateResult(True)


def _g2_cutoff_resource(ctx: dict[str, Any]) -> GateResult:
    """Bind the answer to the trusted task. Closed-resource is enforced at docker run."""
    declared = ctx["_answer"].get("task_id")
    expected = ctx["_task"].get("task_id")
    if isinstance(expected, str) and expected and declared != expected:
        raise T4ParticipantFailure(
            T4Reason.TASK_ID_MISMATCH,
            "answer.json names a different task_id than the unit it was produced for",
            invalid_row_count=1,
        )
    return GateResult(True)


def _g3_domain_semantics(ctx: dict[str, Any]) -> GateResult:
    """Exact roster + numeric contract, then embargo, then prediction-bound faithfulness."""
    params: ScoringParams = ctx["_params"]
    aligned = align_predictions(
        ctx["_answer"],
        ctx["_roster"],
        target_type=params.target_type,
        interval_level=params.interval_level,
    )
    ctx["_aligned"] = aligned

    corpus: CorpusIndex = ctx["_corpus"]
    report = corpus.embargo_report(aligned.all_citations(), ctx["_cutoff"])
    ctx["_embargo"] = report
    if not report.clean:
        reason = (
            T4Reason.CITATION_POST_CUTOFF
            if report.post_cutoff
            else T4Reason.CITATION_UNRESOLVED
            if (report.unresolved or report.malformed)
            else T4Reason.CITATION_UNDATED
        )
        raise T4ParticipantFailure(
            reason,
            "one or more citations are unresolved, undated or post-cutoff",
            violation_count=report.violation_count,
            observed_count=report.checked,
        )

    judge = ctx.get("judge")
    if judge is None:
        raise T4OrganizerFault(
            "no NLI judge is present in the scoring context. The faithfulness gate is the whole "
            "point of Track 4; skipping it and defaulting faithfulness to 1.0 is the defect this "
            "scorer exists to remove."
        )
    claims = prediction_claims(aligned, ctx["_hypothesis_spec"])
    phi = F.citation_faithfulness(
        claims, corpus.lookup(), judge, tau=params.tau_citation
    )
    ctx["_faithfulness"] = float(phi)
    if not ctx.get("_enforce_faithfulness", True):
        # SMOKE ONLY, and never reachable from a rankable factory. The lexical proxy judge
        # produces a number on a different scale from the pinned NLI ensemble, so applying the
        # production threshold to it would give a local preview that is wrong in both directions:
        # it rejects honest work and it would pass work the real judge refuses. The figure is
        # still computed — the plumbing is exercised — and the verdict records that the gate was
        # NOT applied, so a smoke "admissible" can never be read as a production one.
        ctx["_faithfulness_gate_applied"] = False
        return GateResult(True)
    ctx["_faithfulness_gate_applied"] = True
    if phi < params.faithfulness_threshold:
        raise T4ParticipantFailure(
            T4Reason.EVIDENCE_UNSUPPORTED,
            "too few predictions are supported by their own cited evidence",
            observed_count=aligned.count,
            violation_count=max(0, round((1.0 - phi) * aligned.count)),
        )
    return GateResult(True)


# --------------------------------------------------------------------------- #
# Composite                                                                     #
# --------------------------------------------------------------------------- #
def _true_vectors(
    realized: Mapping[str, Any], roster: EntityRoster
) -> tuple[list[str], list[float] | None]:
    """Truth in TRUSTED ROSTER ORDER. The reference must cover the roster exactly."""
    outcomes = realized.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise T4OrganizerFault("the resolved outcome carries no outcomes[]")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in outcomes:
        if not isinstance(row, Mapping) or not isinstance(row.get("entity_id"), str):
            raise T4OrganizerFault("a resolved outcome row has no string entity_id")
        if row["entity_id"] in by_id:
            raise T4OrganizerFault("the resolved outcome repeats an entity_id")
        by_id[row["entity_id"]] = row
    missing = [eid for eid in roster.entity_ids if eid not in by_id]
    if missing:
        raise T4OrganizerFault(
            f"the resolved outcome is missing {len(missing)} roster entities; an incomplete "
            "reference is an organizer fault and must abort rather than shrink the denominator"
        )
    labels: list[str] = []
    values: list[float] = []
    numeric_rows = 0
    for entity_id in roster.entity_ids:
        row = by_id[entity_id]
        label = row.get("true_label", row.get("direction", ""))
        labels.append(label if isinstance(label, str) else "")
        raw = row.get("y", row.get("true_value"))
        if raw is None:
            values.append(math.nan)
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise T4OrganizerFault(
                "the resolved outcome for one roster entity has a target that is neither a "
                "number nor absent. Absent means a pure-label unit; a string or a bool means a "
                "malformed outcome file, and guessing which was meant is how a reference defect "
                "becomes a score."
            )
        value = float(raw)
        if not math.isfinite(value):
            # Reference data, not participant data: the frozen rule sends a nonfinite value in
            # organizer material to an organizer fault. Left alone it is worse than useless --
            # `lo <= nan <= hi` is False, so a NaN target scores as a coverage MISS and the
            # participant is charged for a defect in the answer key.
            raise T4OrganizerFault(
                "the resolved outcome carries a nonfinite numeric target. A NaN or infinity in "
                "reference material is an organizer fault; scored as-is it would count as a "
                "coverage miss and bill the participant for it."
            )
        values.append(value)
        numeric_rows += 1

    # Three cases, and the middle one is the reason this is not a one-liner.
    #
    #   every row numeric  -> ordinary unit, calibration leg applies
    #   NO row numeric     -> pure-label unit. The calibration leg is DROPPED, per the issue #22
    #                         ruling of 2026-08-11 that the sealed scorer already implements
    #                         (final_scorer.py: `interval_cov: float | None`). This code used to
    #                         raise instead, which would have refused a unit anatomy the sealed
    #                         side scores happily -- a public/private divergence in the direction
    #                         that aborts a real evaluation.
    #   MIXED              -> organizer fault. Not a legitimate anatomy: coverage over a partial
    #                         set either shrinks the denominator (forbidden) or scores an absent
    #                         target as a miss (arbitrary). An outcome file where some entities
    #                         carry a numeric target and others do not is malformed.
    if numeric_rows and numeric_rows != len(values):
        raise T4OrganizerFault(
            f"the resolved outcome carries a numeric target for {numeric_rows} of "
            f"{len(values)} roster entities. A unit is numeric or it is pure-label; a mixed "
            "outcome file cannot be scored without either shrinking the denominator or "
            "inventing a target."
        )
    return labels, (values if numeric_rows else None)


def _composite_value(parts: Mapping[str, float | None]) -> float:
    """The composite, narrowed to a float.

    Only `interval_coverage` is nullable in a `_composite` result -- it is None exactly for a
    pure-label unit, where the calibration leg does not apply. The composite itself is always a
    number, and a None here would mean `_composite` returned something it has no branch for, so
    this raises rather than letting `clip_to_domain` receive a None it would crash on later.
    """
    value = parts["composite"]
    if value is None:
        raise T4OrganizerFault(
            "the scorer produced no composite for a unit it reported as scored"
        )
    return float(value)


def _composite(
    aligned: AlignedPredictions,
    realized: Mapping[str, Any],
    params: ScoringParams,
    roster: EntityRoster,
) -> dict[str, float | None]:
    true_labels, true_values = _true_vectors(realized, roster)
    quality = F.predictive_quality(
        params.target_type,
        list(aligned.pred_labels),
        true_labels,
        list(aligned.pred_values),
        true_values if true_values is not None else [],
    )
    w_a, w_c = params.composite_weights

    if true_values is None:
        # Pure-label unit: no calibration leg (issue #22). Reported as None rather than 0.0, so a
        # reader can tell "not applicable" from "measured, and it was zero" -- the same distinction
        # `scored` draws in the shared aggregate.
        return {
            "predictive_quality": float(quality),
            "interval_coverage": None,
            "composite": float(w_a * quality),
        }

    # Coverage over EVERY roster row. Alignment already guaranteed every lo/hi/level is finite and
    # ordered, so there is no row to drop and no denominator to shrink.
    covered = sum(
        1 for lo, hi, y in zip(aligned.lo, aligned.hi, true_values) if lo <= y <= hi
    )
    coverage = covered / roster.count
    composite = w_a * quality - w_c * abs(coverage - params.interval_level)
    return {
        "predictive_quality": float(quality),
        "interval_coverage": float(coverage),
        "composite": float(composite),
    }


# --------------------------------------------------------------------------- #
# The one scoring entrypoint                                                    #
# --------------------------------------------------------------------------- #
def score_unit(
    ctx: dict[str, Any],
    *,
    judge: Any,
    judge_provenance: JudgeProvenance,
    require_outcome: bool = True,
) -> UnitOutcome:
    """Score one unit end to end. The single implementation behind every Track-4 entrypoint.

    Organizer faults **propagate**. They are not converted into a participant zero and they are not
    swallowed into a smaller denominator; the caller is expected to abort the whole evaluation, per
    the frozen C1 ``organizer_failure`` policy.
    """
    ctx = dict(ctx)
    ctx["judge"] = judge
    hydrate(ctx)
    params: ScoringParams = ctx["_params"]
    judge_mapping = judge_provenance.to_mapping()
    judge_provenance.to_judge_record()  # parse through C4 so a shape drift fails here

    try:
        for gate in (
            _g0_integrity,
            _g1_schema,
            _g2_cutoff_resource,
            _g3_domain_semantics,
        ):
            result = gate(ctx)
            if (
                not result.passed
            ):  # pragma: no cover - gates raise rather than return False
                raise T4ParticipantFailure(T4Reason.SCHEMA_INVALID, "gate refused")
    except T4ParticipantFailure as failure:
        logger.info(
            "unit %s: participant failure (%s)",
            ctx.get("unit_handle", "<unit>"),
            failure.reason,
        )
        return UnitOutcome(
            state="participant_failure",
            score=clip_to_domain(params.worst_case),
            rankable=judge_provenance.rankable,
            failure_code=failure.code.value,
            detail=failure.public_detail(),
            labels=(failure.label,),
            judge=judge_mapping,
            diagnostics={"reason": failure.reason.value},
        )

    realized = ctx.get("realized")
    if realized is None:
        if require_outcome:
            raise T4OrganizerFault(
                "no resolved outcome is mounted for this unit. A reference failure is an organizer "
                "fault; it produces no participant score and must abort the evaluation."
            )
        return UnitOutcome(
            state="unrankable",
            score=clip_to_domain(params.worst_case),
            rankable=False,
            failure_code=None,
            detail={},
            labels=(),
            judge=judge_mapping,
            diagnostics={
                "note": "no resolved outcome (public practice unit); admissibility only",
                "faithfulness": ctx.get("_faithfulness"),
            },
        )

    parts = _composite(ctx["_aligned"], realized, params, ctx["_roster"])
    return UnitOutcome(
        state="participant_success",
        score=clip_to_domain(_composite_value(parts)),
        rankable=judge_provenance.rankable,
        failure_code=None,
        detail={},
        labels=(),
        judge=judge_mapping,
        diagnostics={
            "faithfulness": ctx.get("_faithfulness"),
            "expected_entity_count": ctx["_roster"].count,
            "graded_entity_count": ctx["_aligned"].count,
            **parts,
        },
    )


# --------------------------------------------------------------------------- #
# Factories                                                                     #
# --------------------------------------------------------------------------- #
def _verifier(
    ctx: dict[str, Any], provenance: JudgeProvenance, *, require_outcome: bool
) -> HierarchicalVerifier:
    """Adapt `score_unit` to the shared `HierarchicalVerifier` gate/score contract."""
    gates = [
        ("g0_integrity", _wrap(_g0_integrity)),
        ("g1_schema", _wrap(_g1_schema)),
        ("g2_cutoff_resource", _wrap(_g2_cutoff_resource)),
        ("g3_domain_semantics", _wrap(_g3_domain_semantics)),
    ]

    def _scorer(scoring_ctx: dict[str, Any]) -> dict[str, Any]:
        params: ScoringParams = scoring_ctx["_params"]
        realized = scoring_ctx.get("realized")
        if realized is None:
            if require_outcome:
                raise T4OrganizerFault(
                    "no resolved outcome is mounted for this unit; a reference failure aborts"
                )
            return {
                "score": None,
                "rankable": False,
                "judge_mode": provenance.judge_mode,
                "faithfulness": scoring_ctx.get("_faithfulness"),
                "faithfulness_gate_applied": scoring_ctx.get(
                    "_faithfulness_gate_applied", True
                ),
                "note": "no resolved outcome (public practice unit)",
            }
        parts = _composite(
            scoring_ctx["_aligned"], realized, params, scoring_ctx["_roster"]
        )
        return {
            "score": clip_to_domain(_composite_value(parts)),
            "rankable": provenance.rankable,
            "judge_mode": provenance.judge_mode,
            "faithfulness": scoring_ctx.get("_faithfulness"),
            "faithfulness_gate_applied": scoring_ctx.get(
                "_faithfulness_gate_applied", True
            ),
            "expected_entity_count": scoring_ctx["_roster"].count,
            "graded_entity_count": scoring_ctx["_aligned"].count,
            **parts,
        }

    return HierarchicalVerifier(gates, _scorer)


def _wrap(
    gate: Callable[[dict[str, Any]], GateResult],
) -> Callable[[dict[str, Any]], GateResult]:
    """Turn a raising gate into a `GateResult`, with a REDACTED detail.

    The public projection is enum code plus integer counts. Nothing here can emit a string, which
    is what makes it impossible for a validator's exception text to reach a participant artifact.
    """

    def _run(ctx: dict[str, Any]) -> GateResult:
        try:
            return gate(ctx)
        except T4ParticipantFailure as failure:
            return GateResult(False, failure.label, failure.public_detail())

    return _run


def build_verifier(ctx: dict[str, Any]) -> HierarchicalVerifier:
    """THE official, rankable factory. Constructs the pinned production judge or refuses.

    Raising here is the intended behaviour: a missing or unloadable judge is an organizer fault,
    and the frozen C1 organizer-failure policy is to abort the whole evaluation rather than emit a
    partial leaderboard. Use :func:`build_smoke_verifier` for a local, explicitly non-rankable run.
    """
    judge, provenance = build_production_judge()
    ctx["judge"] = judge
    ctx["judge_provenance"] = provenance
    ctx["scorer_version"] = SCORER_VERSION
    return _verifier(ctx, provenance, require_outcome=True)


def build_smoke_verifier(ctx: dict[str, Any]) -> HierarchicalVerifier:
    """The separately named non-rankable factory: lexical judge, `judge_mode="smoke"`.

    Nothing in the production path falls back to this, and no environment variable selects it. It
    exists so a participant can preview admissibility locally without model weights, and every
    artifact it produces is stamped ``rankable=False``.
    """
    judge, provenance = build_smoke_judge()
    ctx["judge"] = judge
    ctx["judge_provenance"] = provenance
    ctx["scorer_version"] = SCORER_VERSION
    # The lexical proxy is not on the production scale, so its number is reported and not gated
    # on. This key is set by NO other factory, and `build_verifier` never touches it.
    ctx["_enforce_faithfulness"] = False
    return _verifier(ctx, provenance, require_outcome=False)


def scorer_identity() -> dict[str, Any]:
    """The provenance block every entrypoint stamps onto its output."""
    return {
        "scorer_package": "qfbench2_track_analysis.scoring",
        "scorer_version": SCORER_VERSION,
        "leaderboard_sort": LEADERBOARD_SORT,
        "metric_domain": {"min": DOMAIN_MIN, "max": DOMAIN_MAX},
    }
