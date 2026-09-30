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
* **Faithfulness is claim by claim.** The judge reads the cited passage against the
  participant's *claim*; from 5.2.0 a claim the evidence demonstrably contradicts, or that
  misattributes its evidence, is false and the unit's score is multiplied by the share of claims
  that are not (see the 5.2.0 note below). A document-level entity check and an exact-code
  numeric backstop run in front of the judge (``corpus.py``, ``numeric.py``).
  Asking instead whether the passage entails the submitted *prediction* (``hypothesis.py``) is
  kept as a recorded diagnostic (``prediction_relevance``) and is never enforced: a pre-cutoff
  passage cannot entail a post-cutoff forecast, so that question has no right answer and no
  threshold on it both admits honest work and refuses an irrelevant passage of the same
  document.
* **The roster is the denominator.** Alignment is driven by the trusted entity roster in trusted
  order; missing, duplicate and unknown ids fail before any metric runs (see ``alignment.py``).
* **Citations resolve through a trusted dictionary.** No participant string is ever interpolated
  into a path, and an unresolved, undated or post-cutoff citation fails closed (see ``corpus.py``).
* **An inadmissible unit scores W, not ``None``.** ``None`` is what removed a unit from the
  aggregate; the frozen policy is a pre-committed worst value that stays in the denominator, and
  real scores are clipped into the same domain so failing can never beat participating.
* **Public detail is enum code plus counts.** No raw exception strings; the previous gate returned
  ``{"reason": str(e)}`` and produced a ~1.5 KB detail embedding the whole answer schema.

Domain and worst case: ``desc``, domain ``[0.0, 1.0]``, ``W = 0*w_a + 0*w_c = 0``.

5.1.0 (adopted 2026-09-24) replaces the calibration leg
``- w_c*|coverage - interval_level|`` with an interval-score ratio against the unit's declared naive
interval: per row ``IS = (hi - lo) + (2/alpha)(lo - y)+ + (2/alpha)(y - hi)+`` with
``alpha = 1 - interval_level``, averaged over the roster; ``iq = naive_IS / (naive_IS + IS)``; the
composite is ``w_a*quality + w_c*iq``. Both legs are ratios in [0, 1], so the worst case moved from
the 5.0.0 value ``W = -w_c*interval_level = -0.27`` to 0. ``IS`` is the Gneiting-Raftery interval
score. A classification unit whose numeric truth is only a 0/1
label code may declare ``[scoring.params] interval_leg = false`` and is then scored on the label
alone (``w_a*quality``), as a pure-label unit is; the flag is never inferred from the data.

5.1.1 adds two citation guards to the gate and changes nothing else. The offsets guard: a citation whose offsets
are not a real slice of its document (negative start, end past the document, start at or after
end) names no passage, where slicing used to clamp it silently. The judged-window guard: every cited passage is cut to
the text the NLI judge actually reads (its tokenizer window, 512 tokens with the claim), and the
numeric backstop reads that same cut (`judged_passage`), so a figure the judge never saw can no
longer anchor a claim. Judge inputs and scores are unchanged for passages inside the window.
The claim's own token length is not capped yet (next version).

5.1.2 fixes review findings and changes no weight or threshold. The figure reader no longer
raises on an unknown suffix ("12kn" is 12), no longer reads a capitalised word such as
"Decreased" or "Margin" as a month, and no longer blanks a 4-digit amount as a year when a
currency sign, a decimal part or a scale word says it is an amount ("$2030 million"). The local
check builds the same windowed ensemble as the gate, and a model ensemble or rankable judge
without a window is an organizer fault rather than a silent read of the whole passage. A
whole-number float offset (``120.0``, which the schema admits) is read as its integer; any other
non-integer offset is refused as CITATION_MALFORMED instead of being dropped.

5.1.3 narrows the numeric backstop's own-value exemption (decided 2026-09-28). A claim
figure is exempt from appearing in the cited passage only when it EXACTLY equals a value the unit
scores: the point forecast on a regression or ranking unit, the interval bounds only when the
interval leg is actually scored (declared AND numeric truth, the one predicate
`_interval_leg_scored` the composite also uses, so a pure-label unit's bounds are never exempt),
never the rank. Before, any submitted value (rank and unscored bounds included) exempted every
figure it matched after scale re-basing and rounding. 5.2.1 (rule "every") allows the scale steps
again, exactly and without rounding, and refuses a figure whose written direction contradicts
the value's sign (`numeric._is_own`).

The interval score itself now comes from the toolkit
(``qfbench2_common.scoring.faithfulness.mean_interval_score``); the formula is unchanged, and
``scoring.mean_interval_score`` stays as a re-export that raises ``T4OrganizerFault``.

5.2.0 (decided 2026-09-28) turns faithfulness from an 80% admission gate
into a per-claim PENALTY. A claim is FALSE when it cites a document not labelled for its entity
(before: a whole-unit CITATION_WRONG_ENTITY refusal), when one of its citations is out of range
(the offsets guard), when its text is malformed, when it is ``unanchored`` (the unchanged numeric backstop: it
states figures and none is in the passage the judge reads), or when the NLI ensemble's three-way
probability of CONTRADICTION for (judged passage, claim) exceeds `contradiction_bar` (mean over
members, max over the claim's passages). Every other claim is NEUTRAL: no credit, no penalty. The
unit scores ``composite * factor ** penalty_k`` with the SOFT FLOOR ``factor = 1 - F / (F + min(T,
PADDING_CAP_PER_ENTITY * E))`` (F false claims, T = every other claim, E the roster's entity count;
adopted 2026-09-29): each false claim costs a share of the unit, and other claims beyond
3 x E in total (a cap over the whole unit, not per entity) do not dilute that share. A unit with no claims, or no false claims, keeps factor 1.
Nothing about faithfulness refuses a unit any more; only structural errors (schema, roster,
embargo, malformed payload, organizer faults) do. The judge's ``entail`` and its window are
untouched; contradiction is a separate member method (`contradiction`) on the same pipeline
forward pass. ``penalty_k`` = 1 and ``contradiction_bar`` = 0.9 are fixed scorer constants (adopted
2026-09-29); a card or plan that names either is refused.
``faithfulness_threshold`` is retired: the legacy value 0.80 that every shipped card and plan
carries is mapped EXPLICITLY to the 5.2.0 defaults (recorded as ``faithfulness_rule`` in the
diagnostics), and any other value is refused as an organizer fault rather than read silently.

5.2.1 (decided 2026-09-29) caps the interval leg by the prediction leg: the interval part can
score above 0.5 only as far as the point forecast beats the naive rule, that is
``interval_quality = min(naive_IS / (naive_IS + IS), max(0.5, quality))``. Before, an answer that
copied the naive rule's points and narrowed its band scored above the naive rule with no
information, because the declared naive bands can be wider than the outcomes needed. Below 0.5
nothing changes, so a band that misses still costs. The uncapped value is recorded as
``raw_interval_quality`` in the diagnostics. The same version reads equivalent number forms in
the claim check (`numeric`: a fraction of a point, a number in words, glued suffixes), stops
reading dates, index bases and rule numbers as amounts, and makes the own-value exemption
direction- and scale-aware, float-noise free, and inclusive of the "±" half-width and the unit's
interval level.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pathlib
import statistics
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Any

from qfbench2_common.contracts import OrganizerFault
from qfbench2_common.failure_labels import FailureLabel
from qfbench2_common.scoring import faithfulness as F
from qfbench2_common.taskcard import schema_path
from qfbench2_common.verifier import GateResult, HierarchicalVerifier

from .alignment import AlignedPredictions, EntityRoster, TARGET_TYPES, align_predictions
from .codes import T4OrganizerFault, T4ParticipantFailure, T4Reason
from .corpus import CorpusIndex, parse_iso_date
from .hypothesis import HypothesisSpec, prediction_claims
from .judge_factory import JudgeProvenance, build_production_judge, build_smoke_judge
from .numeric import FIGURE_RULES, claim_number_status, verbatim_quote

__all__ = [
    "CLAIM_MAX_JUDGE_TOKENS",
    "DOMAIN_MAX",
    "DOMAIN_MIN",
    "FIGURE_SPAN_CAP",
    "LEADERBOARD_SORT",
    "SCORER_VERSION",
    "ClaimReport",
    "ClaimVerdict",
    "ScoringParams",
    "UnitOutcome",
    "build_smoke_verifier",
    "build_verifier",
    "clip_to_domain",
    "judged_passage",
    "score_unit",
    "unit_entity_names",
    "worst_case_score",
]

logger = logging.getLogger(__name__)

LEADERBOARD_SORT = "desc"

#: The frozen Track-4 metric domain. `W` is the minimum because the direction is `desc`.
DOMAIN_MIN = 0.0
DOMAIN_MAX = 1.0

#: Bumped whenever the composite, the gates or the evidence semantics change. Recorded in
#: provenance so a leaderboard can be attributed to an implementation rather than to a repo state.
SCORER_VERSION = "5.2.1"

#: Which figures of a claim its cited spans must carry. "every" (the whole cited span) is the
#: 5.2.0 rule; "any" (the judge window) is the earlier rule. A switch for the measurement.
FIGURE_RULE = "every"

#: A cited span longer than this many characters anchors no figure (the published
#: per-citation cap of the reasoning contract, where a longer citation is refused). A
#: whole-document citation no longer lets a figure that occurs anywhere in a long filing anchor a
#: claim by coincidence: a claim with figures cites the passage that states them. A verbatim quote
#: is still recognised in a span of any length.
FIGURE_SPAN_CAP = 8000

#: The longest claim the judge reads, in judge tokens. Above it the claim leaves too little
#: of the 512-token window for its passage, and past ~509 tokens the pipeline's "too short"
#: fallback would read premise and claim untruncated, beyond the window (bypassing the judged-window guard). Such a
#: claim is malformed (false), like one over `CLAIM_TEXT_MAX_CHARS`.
CLAIM_MAX_JUDGE_TOKENS = 400


def unit_entity_names(task: Mapping[str, Any]) -> tuple[str, ...]:
    """The unit's own entity names and tickers: each `task.json` `entities` row's
    `name`, `entity_id` and any `*ticker` field. A number written as part of one of them
    ("Phillips 66", "S&P 500", "3M") is not a figure (`numeric.name_fragments`)."""
    names: list[str] = []
    for row in task.get("entities") or ():
        if not isinstance(row, Mapping):
            continue
        for key, value in row.items():
            if isinstance(value, str) and (
                key in ("name", "entity_id") or key.endswith("ticker")
            ):
                names.append(value)
    return tuple(dict.fromkeys(names))


_SCHEMA = schema_path("analysis.schema.json")

#: The one retired admission threshold 5.2.0 still reads. Every shipped card and every C1
#: plan carries ``faithfulness_threshold = 0.80`` (the hub's C1 schema still requires the key), and
#: 5.2.0 maps exactly that value to the per-claim penalty with its default parameters. Any other
#: value was written for the retired 80% gate and has no meaning under the penalty, so it is
#: refused as an organizer fault instead of being ignored.
LEGACY_FAITHFULNESS_THRESHOLD = 0.80

#: The name recorded in every outcome's diagnostics for the faithfulness rule that scored it (the soft
#: floor with a cap of 3 x E non-false claims in total, adopted 2026-09-29; earlier 5.2.0 builds recorded
#: "claim_penalty/5.2.0" for the plain share).
FAITHFULNESS_RULE = "claim_penalty_soft_floor_3e/5.2.0"

#: 5.2.0 defaults, chosen from measurements on the development runs: k = 1 and an
#: ensemble three-way P(contradiction) bar of 0.9.
DEFAULT_PENALTY_K = 1.0
DEFAULT_CONTRADICTION_BAR = 0.9
#: The soft floor's padding cap (adopted 2026-09-29): at most this many non-false claims
#: per roster entity count toward diluting the false ones, ``factor = 1 - F / (F + min(T, 3E))``.
PADDING_CAP_PER_ENTITY = 3
#: Fixed scorer constants in 5.2.0 (decided 2026-09-29): a card or plan naming either is refused.
FIXED_PENALTY_KEYS = ("penalty_k", "contradiction_bar")


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
    #: False only on a classification unit that DECLARES its interval leg off
    #: (`[scoring.params] interval_leg = false`): its numeric truth is a 0/1 label code, so it is
    #: scored on the label alone, like a pure-label unit. Never inferred from the data.
    interval_leg: bool = True
    #: 5.2.0: the exponent of the per-claim penalty, ``score = composite * factor ** k`` (the soft floor,
    #: `ClaimReport.penalty_factor`).
    penalty_k: float = DEFAULT_PENALTY_K
    #: 5.2.0: a judged claim is false when the ensemble's three-way P(contradiction) exceeds this.
    contradiction_bar: float = DEFAULT_CONTRADICTION_BAR

    def __post_init__(self) -> None:
        if self.target_type not in TARGET_TYPES:
            raise T4OrganizerFault(
                f"target_type {self.target_type!r} is not one of {list(TARGET_TYPES)}; an unknown "
                "target type is refused, never defaulted to classification"
            )
        if not 0.0 < self.interval_level < 1.0:
            raise T4OrganizerFault("interval_level must be in (0, 1)")
        if abs(self.faithfulness_threshold - LEGACY_FAITHFULNESS_THRESHOLD) > 1e-12:
            raise T4OrganizerFault(
                f"faithfulness_threshold = {self.faithfulness_threshold!r} is not the legacy value "
                f"{LEGACY_FAITHFULNESS_THRESHOLD}. Scorer 5.2.0 retired the 80% admission gate for "
                "a per-claim penalty (penalty_k, contradiction_bar); only the legacy 0.80 that "
                "every shipped card and plan carries is mapped to it, and any other value was "
                "written for a rule that no longer exists, so it is refused rather than ignored"
            )
        for name in ("penalty_k", "contradiction_bar"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise T4OrganizerFault(f"{name} must be a number")
            if not math.isfinite(value):
                raise T4OrganizerFault(f"{name} must be finite")
        if not self.penalty_k > 0.0:
            raise T4OrganizerFault("penalty_k must be > 0")
        if not 0.0 < self.contradiction_bar < 1.0:
            raise T4OrganizerFault("contradiction_bar must be in (0, 1)")
        if not 0.0 <= self.tau_citation < 1.0:
            raise T4OrganizerFault("tau_citation must be in [0, 1)")
        w_a, w_c = self.composite_weights
        if abs((w_a + w_c) - 1.0) > 1e-9:
            raise T4OrganizerFault("composite_weights must sum to 1")
        if not isinstance(self.interval_leg, bool):
            raise T4OrganizerFault(
                "interval_leg must be a boolean (true or false); a string or a number is refused "
                "rather than read as truthy"
            )
        if not self.interval_leg and self.target_type != "classification":
            raise T4OrganizerFault(
                f"interval_leg = false is declared on a {self.target_type} unit. Only a "
                "classification unit whose numeric truth is a label code may opt out of the "
                "interval leg; a numeric forecast's interval is always scored."
            )

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
        """Build from the trusted plan values, refusing a mounted-card value that disagrees.

        `penalty_k` and `contradiction_bar` are FIXED scorer constants in 5.2.0 (decided
        2026-09-29): no card or plan may set them, and one that tries is refused rather than read.
        """
        for source, what in ((trusted, "plan / scoring_params"), (card_params, "card")):
            named = sorted(k for k in FIXED_PENALTY_KEYS if source and k in source)
            if named:
                raise T4OrganizerFault(
                    f"the {what} sets {named}; scorer 5.2.0 fixes penalty_k = {DEFAULT_PENALTY_K} and "
                    f"contradiction_bar = {DEFAULT_CONTRADICTION_BAR} as scorer constants, not unit parameters"
                )
        base: dict[str, Any] = {
            "target_type": "classification",
            "interval_level": 0.90,
            "faithfulness_threshold": 0.80,
            "tau_citation": 0.5,
            "composite_weights": (0.7, 0.3),
            "interval_leg": True,
            "penalty_k": DEFAULT_PENALTY_K,
            "contradiction_bar": DEFAULT_CONTRADICTION_BAR,
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
            interval_leg=base["interval_leg"],
            penalty_k=_number(base["penalty_k"], "penalty_k"),
            contradiction_bar=_number(base["contradiction_bar"], "contradiction_bar"),
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
                "interval_leg",
                "penalty_k",
                "contradiction_bar",
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
                interval_leg=merged["interval_leg"],
                penalty_k=_number(merged["penalty_k"], "penalty_k"),
                contradiction_bar=_number(
                    merged["contradiction_bar"], "contradiction_bar"
                ),
            )
        return params


def _number(value: Any, name: str) -> float:
    """A penalty parameter as a float; a bool or a non-number is an organizer fault, never truthy."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise T4OrganizerFault(
            f"{name} must be a number; {type(value).__name__} is refused rather than coerced"
        )
    return float(value)


def _refuse_card_disagreement(
    params: ScoringParams, card_params: Mapping[str, Any]
) -> None:
    checks: list[tuple[str, Any, Any]] = []
    for key in (
        "target_type",
        "interval_level",
        "faithfulness_threshold",
        "tau_citation",
        "penalty_k",
        "contradiction_bar",
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
    if "interval_leg" in card_params:
        from_card = card_params["interval_leg"]
        if not isinstance(from_card, bool) or from_card != params.interval_leg:
            raise T4OrganizerFault(
                "the mounted card's interval_leg disagrees with the trusted plan (or is not a "
                "boolean); the plan wins and the disagreement is a staging fault"
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
    """`W` = ``0*w_a + 0*w_c`` = 0 — the worst attainable composite for a unit.

    Both composite legs are ratios in [0, 1] (5.1.0), so neither the weights nor the interval
    level move the worst case. The arguments are kept so every caller's signature is unchanged.
    """
    del composite_weights, interval_level
    return 0.0


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

    # The task table is citable as `doc_id: "task"` (5.2.x).
    ctx["_corpus"] = CorpusIndex.from_unit(unit).with_task_table(task, ctx["_cutoff"])
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
    _hydrate_naive(ctx, unit)
    ctx["_hydrated"] = True


NAIVE_ANSWER_FILE = "naive_answer.json"


def _hydrate_naive(ctx: dict[str, Any], unit: pathlib.Path) -> None:
    """Read and align the unit's DECLARED naive rule, the zero-skill anchor of the soft ratios.

    ``reference/naive_answer.json`` is a full answer file in the analysis schema. It is organizer
    material, aligned to the trusted roster exactly as the participant's answer is, so a defect in
    it is an organizer fault and never a participant failure. A regression unit needs its point
    forecasts (accuracy leg); every unit with numeric truth needs its intervals (interval leg,
    5.1.0). A regression unit with a resolved outcome and no naive file is refused here. Any
    other unit's file is read when present; whether it was REQUIRED depends on the outcome having
    numeric truth, which `_composite` decides and refuses on. A practice unit with no outcome
    scores nothing and so needs no naive rule.
    """
    ctx["_naive_answer"] = None
    ctx["_naive_aligned"] = None
    ctx["_naive_provenance"] = None
    params: ScoringParams = ctx["_params"]
    path = unit / "reference" / NAIVE_ANSWER_FILE
    if not path.is_symlink() and not path.exists():
        # 5.2.0: every unit type is anchored to its declared naive rule, so a unit with a
        # resolved outcome needs the file whatever its target type.
        if ctx.get("realized") is None:
            return
    if path.is_symlink() or not path.is_file():
        raise T4OrganizerFault(
            f"{params.target_type} unit {unit.name!r} has no readable "
            f"reference/{NAIVE_ANSWER_FILE} (missing, a link, or not a regular file). The declared "
            "naive rule is a scoring parameter of every unit with a resolved outcome (5.2.0: the "
            "prediction leg of every target type is anchored to it); scoring without it is an "
            "organizer fault, never a fallback to another baseline."
        )
    try:
        blob = path.read_bytes()
        answer = json.loads(blob.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T4OrganizerFault(
            f"reference/{NAIVE_ANSWER_FILE} is unreadable or not JSON"
        ) from exc
    if not isinstance(answer, Mapping):
        raise T4OrganizerFault(f"reference/{NAIVE_ANSWER_FILE} is not a JSON object")
    try:
        aligned = align_predictions(
            answer,
            ctx["_roster"],
            target_type=params.target_type,
            interval_level=params.interval_level,
        )
    except T4ParticipantFailure as failure:
        raise T4OrganizerFault(
            f"reference/{NAIVE_ANSWER_FILE} fails alignment to the trusted roster "
            f"({failure.reason.value}); the unit's naive rule is organizer material, so this is "
            "an organizer fault"
        ) from failure
    notes = answer.get("notes")
    baseline_id = notes.get("baseline_id") if isinstance(notes, Mapping) else None
    ctx["_naive_answer"] = answer
    ctx["_naive_aligned"] = aligned
    ctx["_naive_provenance"] = {
        "naive_answer_file": NAIVE_ANSWER_FILE,
        "naive_answer_sha256": hashlib.sha256(blob).hexdigest(),
        "naive_baseline_id": baseline_id if isinstance(baseline_id, str) else None,
    }


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


def _entity_bound_citations(corpus: CorpusIndex, aligned: AlignedPredictions) -> int:
    """Count the citations whose document is not about the entity that cites it.

    5.2.0: a wrong-entity citation no longer refuses the unit. It makes the CLAIM that carries it
    false (`evaluate_claims`), which costs that claim's share of the unit through the per-claim
    penalty. This function keeps the organizer-fault checks below (an unlabelled manifest, a
    labelled manifest that forgot a cited document) and returns the number of wrong-entity
    citations for the review queue.

    Why this exists: an NLI model has no notion of which company a passage is about. Read a
    claim about company A against a passage from company B's filing and the model returns
    entailment whenever the wording matches -- the sentences are about the same kind of thing,
    and an extracted span usually does not repeat the name of the company whose filing it came
    from. The cited DOCUMENT does carry that identity, and the organizer labels every corpus
    document in the trusted manifest with the roster entities it is about, or as `shared`
    (market-wide). So the correspondence is checked here, deterministically, at the document
    level and BEFORE any judge call: a prediction for entity E may cite a document only if the
    manifest lists E on it or marks it shared; a claim citing any other document is false.

    A manifest with no labels at all is an organizer fault -- an unlabelled corpus cannot be
    silently admitted. A document with `entity_ids: []` is about someone off the roster (a peer,
    a benchmark) and admits nobody: peer citations are refused.

    This does not make shared-corpus units entity-specific: where every document is `shared` by
    design (macro series, FOMC material), every citation passes here and the judge is all there
    is. That limit is documented, not hidden.
    """
    if not corpus.labelled:
        raise T4OrganizerFault(
            "the unit manifest carries no entity labels (`entity_ids` / `shared`) on any corpus "
            "document, so the entity check cannot run. An unlabelled corpus is an organizer "
            "fault, never a silent pass."
        )
    gaps = set(corpus.unlabelled_doc_ids)
    checked = 0
    wrong = 0
    for entity_id, citations in zip(aligned.entity_ids, aligned.citations_by_entity):
        for citation in citations:
            checked += 1
            doc = corpus.resolve(citation.get("doc_id"))
            if doc.doc_id in gaps:
                # The manifest is labelled but forgot this document. That is the organizer's gap,
                # and refusing the participant for it would blame them for it.
                raise T4OrganizerFault(
                    f"the unit manifest labels its corpus but carries no `entity_ids` / `shared` "
                    f"for {len(gaps)} cited-or-citable document(s); an unlabelled document in a "
                    "labelled manifest is an organizer fault, never a participant refusal"
                )
            if not doc.admits_citation(entity_id, citation):
                wrong += 1
    del checked
    return wrong


def _g3_domain_semantics(ctx: dict[str, Any]) -> GateResult:
    """Exact roster + numeric contract, then embargo, then the document-level entity labels, then
    the per-claim faithfulness verdicts (5.2.0: a penalty, never a refusal). The
    prediction-relevance diagnostic is computed last and recorded, never enforced."""
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

    # Organizer faults (unlabelled manifest) raise here; wrong-entity citations are per-claim
    # falsehoods from 5.2.0, counted for the review queue and penalised in `evaluate_claims`.
    ctx["_wrong_entity_citations"] = _entity_bound_citations(corpus, aligned)

    judge = ctx.get("judge")
    if judge is None:
        raise T4OrganizerFault(
            "no NLI judge is present in the scoring context. The faithfulness check is the whole "
            "point of Track 4; skipping it and defaulting faithfulness to 1.0 is the defect this "
            "scorer exists to remove."
        )
    lookup = corpus.lookup()
    claims = evaluate_claims(
        aligned,
        lookup,
        judge,
        target_type=params.target_type,
        interval_scored=claim_interval_scored(ctx),
        contradiction_bar=params.contradiction_bar,
        entity_admits=_entity_admits(corpus),
        require_window=bool(ctx.get("_require_judge_window", False)),
        entity_names=unit_entity_names(ctx["_task"]),
        interval_level=params.interval_level,
    )
    ctx["_claim_report"] = claims
    ctx["_faithfulness"] = float(claims.faithfulness)
    # RECORDED, NEVER ENFORCED. Does the passage entail the prediction itself, asked in three
    # phrasings and aggregated by median. It goes to the review queue beside the claim verdicts
    # so an operator can see a submission whose claims are accurate about passages that say
    # nothing about its forecast; nothing acts on it because a pre-cutoff passage cannot entail
    # a post-cutoff forecast, so the score cannot tell honestly cited evidence from an unrelated
    # passage of the same document. "Does the evidence support the forecast" is reasoning
    # grading's question.
    ctx["_prediction_relevance"] = float(
        _paraphrase_faithfulness(
            prediction_claims(aligned, ctx["_hypothesis_spec"]),
            lookup,
            judge,
            tau=params.tau_citation,
        )
    )
    if not ctx.get("_enforce_faithfulness", True):
        # SMOKE ONLY, and never reachable from a rankable factory. The lexical proxy judge is not
        # on the production scale, so its contradiction verdicts are reported and NOT charged;
        # the deterministic reasons (wrong entity, out of range, malformed, unanchored) are
        # organizer data and exact code, so they are charged as in production. The verdict
        # records that the judge part was not applied, so a smoke score is never a production one.
        ctx["_faithfulness_gate_applied"] = False
        ctx["_faithfulness_factor"] = claims.penalty_factor(
            params.penalty_k, entity_count=ctx["_roster"].count, judge_verdicts=False
        )
        return GateResult(True)
    # A non-rankable judge that cannot answer the contradiction question (a served preview) charged only the
    # deterministic reasons; the verdict says so, so it is never read as a production one.
    ctx["_faithfulness_gate_applied"] = claims.contradiction_applied
    ctx["_faithfulness_factor"] = claims.penalty_factor(
        params.penalty_k, entity_count=ctx["_roster"].count
    )
    return GateResult(True)


def _entity_admits(corpus: CorpusIndex) -> Callable[[str, Any], bool]:
    """Whether the manifest lets `entity_id` cite `doc_id` (its label, or `shared`).

    Only reached after the embargo check resolved every cited doc_id and `_entity_bound_citations`
    refused an unlabelled document as an organizer fault, so `resolve` cannot fail here.
    """

    def admits(entity_id: str, cite: Any) -> bool:
        # The whole citation, so a task-table span is bound to the entity's own row
        return bool(corpus.resolve(cite.get("doc_id")).admits_citation(entity_id, cite))

    return admits


# --------------------------------------------------------------------------- #
# Claim-by-claim faithfulness (the gate)                                        #
# --------------------------------------------------------------------------- #


#: Why a claim is false (5.2.0). A claim can carry several; any one makes it false.
FALSE_REASONS = (
    "wrong_entity",
    "out_of_range",
    "malformed",
    "unanchored",
    "contradicted",
)


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    """One claim's outcome (5.2.0: a penalty verdict, never an admission vote).

    `status` is the figure/NLI stage: ``malformed`` (no usable claim text; nothing else is
    checked), ``unanchored`` (the claim states figures and none of them appears in any passage
    the judge reads of its citations; the judge is not asked), ``contradicted`` (the ensemble's
    three-way P(contradiction) for some judged passage exceeds the unit's `contradiction_bar`),
    or ``neutral`` (anything else: no credit, no penalty). `score` is that best
    P(contradiction), 0.0 whenever the judge was not asked. `wrong_entity` and `out_of_range`
    are citation facts about the claim: it cites a document the manifest does not label for its
    entity, or a slice that is not in its document (the offsets guard). The claim is FALSE when any of the four
    applies (`reasons`).
    """

    entity_id: str
    text: str
    status: str
    score: float
    wrong_entity: bool = False
    out_of_range: bool = False

    @property
    def judged(self) -> bool:
        return self.status in ("contradicted", "neutral")

    @property
    def reasons(self) -> tuple[str, ...]:
        """Every reason this claim is false, in `FALSE_REASONS` order; empty for a neutral claim."""
        out: list[str] = []
        if self.wrong_entity:
            out.append("wrong_entity")
        if self.out_of_range:
            out.append("out_of_range")
        if self.status in ("malformed", "unanchored", "contradicted"):
            out.append(self.status)
        return tuple(out)

    @property
    def false(self) -> bool:
        return bool(self.reasons)


@dataclass(frozen=True, slots=True)
class ClaimReport:
    """Every claim's verdict, in roster order, and the penalty the unit takes for the false ones.

    Two citation counts ride along for the review queue: citations whose offsets were out of
    range and so named no passage (the offsets guard), and passages cut to the judge's window
    (the judged-window guard).
    """

    verdicts: tuple[ClaimVerdict, ...]
    out_of_range_citation_count: int = 0
    window_cut_citation_count: int = 0
    #: False when the judge could not answer the contradiction question (a served or stand-in judge that exposes
    #: only `entail`, in a NON-rankable preview): no claim was put to it, only the deterministic reasons were
    #: checked, and the verdicts are not a production reading. A rankable run refuses instead (organizer fault).
    contradiction_applied: bool = True

    @property
    def claim_count(self) -> int:
        return len(self.verdicts)

    @property
    def false_count(self) -> int:
        return sum(1 for v in self.verdicts if v.false)

    @property
    def unanchored_count(self) -> int:
        return sum(1 for v in self.verdicts if v.status == "unanchored")

    @property
    def malformed_count(self) -> int:
        return sum(1 for v in self.verdicts if v.status == "malformed")

    @property
    def contradicted_count(self) -> int:
        return sum(1 for v in self.verdicts if v.status == "contradicted")

    @property
    def wrong_entity_claim_count(self) -> int:
        return sum(1 for v in self.verdicts if v.wrong_entity)

    @property
    def out_of_range_claim_count(self) -> int:
        return sum(1 for v in self.verdicts if v.out_of_range)

    @property
    def faithfulness(self) -> float:
        """Share of claims that are NOT false. 1.0 when there are no claims: nothing was stated,
        so nothing false was stated (the penalty factor is 1)."""
        if not self.verdicts:
            return 1.0
        return 1.0 - self.false_count / len(self.verdicts)

    def penalty_factor(
        self, k: float, *, entity_count: int, judge_verdicts: bool = True
    ) -> float:
        """The multiplier on the unit's composite, the SOFT FLOOR (adopted 2026-09-29)::

            (1 - F / (F + min(T, PADDING_CAP_PER_ENTITY * E))) ** k

        F = false claims, T = every other claim (``claims - F``), E = `entity_count`, the roster's
        entity count. 1.0 with no claims or no false claims; 0.0 when every claim is false. While T is
        at most ``3E`` (a cap over the whole unit, not per entity) it equals the plain share
        ``1 - F/claims``; beyond that extra claims do not dilute the false ones, so each false claim costs at least
        ``1/(F + 3E)`` of the unit.

        `judge_verdicts=False` counts only the deterministic reasons (wrong entity, out of range,
        malformed, unanchored) and ignores ``contradicted``: the smoke path, whose lexical judge
        is not on the production scale, still charges what organizer data and exact code decide.
        T is then the claims that path does not count false.
        """
        if (
            isinstance(entity_count, bool)
            or not isinstance(entity_count, int)
            or entity_count < 1
        ):
            raise ValueError(
                f"entity_count must be a positive roster count, got {entity_count!r}"
            )
        if not self.verdicts:
            return 1.0
        false = sum(
            1
            for v in self.verdicts
            if (
                v.reasons
                if judge_verdicts
                else tuple(r for r in v.reasons if r != "contradicted")
            )
        )
        if not false:
            return 1.0
        rest = len(self.verdicts) - false
        diluting = min(rest, PADDING_CAP_PER_ENTITY * entity_count)
        return float((1.0 - false / (false + diluting)) ** k)

    @property
    def unanchored_claims(self) -> tuple[dict[str, str], ...]:
        """The claims the numeric backstop found unanchored, each with its entity and its own
        text. Recorded for the operator's review queue; the text is the participant's own."""
        return tuple(
            {"entity_id": v.entity_id, "claim": v.text}
            for v in self.verdicts
            if v.status == "unanchored"
        )

    @property
    def false_claims(self) -> tuple[dict[str, Any], ...]:
        """Every false claim with its entity, its own text and its reasons (operator-only)."""
        return tuple(
            {"entity_id": v.entity_id, "claim": v.text, "reasons": list(v.reasons)}
            for v in self.verdicts
            if v.false
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "faithfulness_rule": FAITHFULNESS_RULE,
            "contradiction_applied": self.contradiction_applied,
            "claim_count": self.claim_count,
            "false_claim_count": self.false_count,
            "unanchored_claim_count": self.unanchored_count,
            "malformed_claim_count": self.malformed_count,
            "contradicted_claim_count": self.contradicted_count,
            "wrong_entity_claim_count": self.wrong_entity_claim_count,
            "out_of_range_claim_count": self.out_of_range_claim_count,
            "out_of_range_citation_count": self.out_of_range_citation_count,
            "window_cut_citation_count": self.window_cut_citation_count,
            "unanchored_claims": list(self.unanchored_claims),
            "false_claims": list(self.false_claims),
        }


def _span_text(
    cite: Mapping[str, Any],
    corpus_lookup: Callable[[str], Mapping[str, Any]],
) -> tuple[str | None, bool]:
    """The passage one citation names, or None when it names none; and whether the reason was
    out-of-range offsets.

    A citation names no passage when a field is missing, its document is not in the corpus, or
    (5.1.1, the offsets guard) its offsets are not a real slice of the document: `span_start < 0`,
    `span_end > len(text)`, or `span_start >= span_end`. Python slicing would clamp such offsets
    silently -- `[0, N)` past the end read the whole document, a negative start read its tail --
    so they are refused here instead, and the claim is judged on its other citations only, the
    same treatment as a citation of an unknown document. The caller counts them
    (`ClaimReport.out_of_range_citation_count`); the unit is not refused for them by itself.
    """
    doc_id = cite.get("doc_id")
    start, end = cite.get("span_start"), cite.get("span_end")
    if doc_id is None or start is None or end is None:
        return None, False
    try:
        document = corpus_lookup(doc_id)
    except (KeyError, LookupError, TypeError):
        return None, False
    text = str(document.get("text", ""))
    offsets_are_ints = all(
        isinstance(v, int) and not isinstance(v, bool) for v in (start, end)
    )
    if not (offsets_are_ints and 0 <= start < end <= len(text)):
        return None, True
    return text[start:end], False


def judged_passage(
    judge: Any, premise: str, hypothesis: str, *, require_window: bool = False
) -> str:
    """The text of `premise` the judge actually reads beside `hypothesis` (5.1.1, the judged-window guard).

    THE ONE SHARED CUT. The gate hands this text both to the numeric backstop and to the judge,
    so the two cannot disagree about what a passage says: a figure the judge never saw cannot
    anchor a claim. The production ensemble reads at most its tokenizer window (512 tokens,
    premise and claim together; the pipeline truncates the premise only) and says where through
    `judged_premise` (`faithfulness.judge.judged_premise_text`). A passage already inside the
    window comes back unchanged, so the judge's input and score are exactly what they were
    before 5.1.1; a longer one comes back as the prefix the pipeline would have kept, which the
    judge scores identically to the whole passage.

    A judge with no `judged_premise` (the smoke judge, test stand-ins) reads its whole premise,
    so the whole premise is returned -- but only where that is true (5.1.2). An ensemble whose
    members read a window (they expose `judged_premise`, as the DeBERTa and served members do)
    but which does not itself expose one -- the plain hub `EnsembleNLIJudge` -- and any judge in
    a rankable run (`require_window`, set from the judge provenance) without `judged_premise`
    are organizer faults, never a silent read of the whole passage, which would let a figure
    the judge never saw anchor a claim (the judged-window guard failing open).
    """
    window = getattr(judge, "judged_premise", None)
    if window is None:
        members = getattr(judge, "_judges", ())
        if require_window or any(hasattr(m, "judged_premise") for m in members):
            raise T4OrganizerFault(
                "the NLI judge does not say which text of a passage it reads (no "
                "judged_premise), so the figure check cannot read the same window as the judge. "
                "Build the judge with the windowed ensemble (faithfulness.judge.build_judge or "
                "judge_factory.build_production_judge)."
            )
        return premise
    return str(window(premise, hypothesis))


def _cited_spans(
    citations: tuple[Mapping[str, Any], ...],
    corpus_lookup: Callable[[str], Mapping[str, Any]],
) -> tuple[list[str], int]:
    """The non-empty passages a claim cites, read narrowly: `span_start`/`span_end` into the
    document's `text`, plus how many citations had out-of-range offsets (see `_span_text`). See
    `_paraphrase_faithfulness` for why this parser is narrower than the shared primitive's and
    why nothing it refuses can reach here."""
    spans: list[str] = []
    out_of_range = 0
    for cite in citations:
        premise, bad_offsets = _span_text(cite, corpus_lookup)
        out_of_range += bad_offsets
        if premise:
            spans.append(premise)
    return spans, out_of_range


def _interval_leg_scored(params: ScoringParams, *, numeric_truth: bool | None) -> bool:
    """Whether this unit scores the interval leg: ONE predicate, read by `_composite` (which scores
    it) and by the claim check's own-value exemption (`claim_interval_scored`), so the two cannot
    disagree about which bounds count.

    The leg is scored when the unit declares it (`interval_leg`, true unless a classification
    unit's card says false) AND the resolved outcome carries numeric truth; a pure-label unit never
    scores it. `numeric_truth=None` means no outcome is mounted (a public practice unit, or a
    participant's local check), so numeric truth is unknown: the declared flag is all there is.
    A regression or ranking unit always has numeric truth (`_true_vectors` refuses one without),
    so only a pure-label classification unit that leaves `interval_leg` at its default can read
    differently with and without its outcome; declaring `interval_leg = false` removes that.
    """
    if numeric_truth is None:
        return params.interval_leg
    return numeric_truth and params.interval_leg


def claim_interval_scored(ctx: Mapping[str, Any]) -> bool:
    """`_interval_leg_scored` for a hydrated context, as the gate and the local check both call it.

    Numeric truth is read with `_true_vectors`, the reader `_composite` scores with, so a
    malformed outcome is the same organizer fault here as there (it is now raised at the gate).
    """
    params: ScoringParams = ctx["_params"]
    realized = ctx.get("realized")
    if realized is None:
        return _interval_leg_scored(params, numeric_truth=None)
    _, true_values = _true_vectors(
        realized, ctx["_roster"], target_type=params.target_type
    )
    return _interval_leg_scored(params, numeric_truth=true_values is not None)


def _scored_own_values(
    aligned: AlignedPredictions, index: int, *, target_type: str, interval_scored: bool
) -> tuple[float, ...]:
    """The entity's submitted values this unit SCORES, the only ones a claim may state without
    the cited passage carrying them (5.1.3, decided 2026-09-28).

    The point forecast on a regression or ranking unit (ranking quality reads `point_forecast`);
    the interval bounds only when the interval leg is actually scored (`interval_scored`, from
    `claim_interval_scored`: declared AND numeric truth); never the rank, and never an unscored
    bound or an unscored point value. An unscored value costs the participant nothing to set, so
    exempting it would let any fabricated amount skip the figure check.
    """
    if target_type not in TARGET_TYPES:
        raise T4OrganizerFault(
            f"target_type {target_type!r} is not one of {list(TARGET_TYPES)}; the claim check "
            "cannot tell which submitted values are scored"
        )
    own: list[float] = []
    if target_type in ("regression", "ranking"):
        own.append(aligned.pred_values[index])
    if interval_scored:
        own.extend((aligned.lo[index], aligned.hi[index]))
    return tuple(own)


def contradiction_supported(judge: Any) -> bool:
    """Can `judge` answer the three-way contradiction question? It needs a `contradiction` method, and an ensemble
    (`_judges`) needs it on every member: the served member (`faithfulness.judge.ServedNLIJudge`) exposes only the
    two-way `entail` its endpoint returns, so a served ensemble cannot."""
    if getattr(judge, "contradiction", None) is None:
        return False
    members = getattr(judge, "_judges", None)
    if members is None:
        return True
    return bool(members) and all(
        getattr(m, "contradiction", None) is not None for m in members
    )


def judged_contradiction(
    judge: Any, premise: str, hypothesis: str, *, require_window: bool = False
) -> float:
    """The ensemble's three-way P(contradiction) for (premise, hypothesis) (5.2.0).

    The production ensemble exposes `contradiction` (each member's softmax over entailment,
    neutral and contradiction from the SAME pipeline forward pass `entail` runs, averaged over
    members); `entail` and its two-way normalisation are unchanged. A judge without
    `contradiction` (the smoke judge, test stand-ins) never finds a contradiction -- but only
    where that is allowed: in a rankable run (`require_window`), or for an ensemble whose members
    do expose it, a missing method is an organizer fault, never a silent "no contradiction".
    """
    method = getattr(judge, "contradiction", None)
    if method is None:
        members = getattr(judge, "_judges", ())
        if require_window or any(hasattr(m, "contradiction") for m in members):
            raise T4OrganizerFault(
                "the NLI judge exposes no three-way contradiction probability (no "
                "`contradiction`), so the 5.2.0 faithfulness penalty cannot be computed. Build the "
                "judge with judge_factory.build_production_judge or faithfulness.judge.build_judge."
            )
        return 0.0
    value = float(method(premise, hypothesis))
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise T4OrganizerFault(
            "the NLI judge returned a contradiction probability outside [0, 1]"
        )
    return value


def _claim_too_long(judge: Any, text: str, *, require_window: bool = False) -> bool:
    """Is the claim longer than `CLAIM_MAX_JUDGE_TOKENS` judge tokens?

    Counted with the judge's own tokenizer (`claim_tokens`). A judge that cannot count (the
    smoke judge, test stand-ins) applies no guard, but only where that is allowed: a rankable run
    (`require_window`) or an ensemble whose members can count must count."""
    count = getattr(judge, "claim_tokens", None)
    if count is None:
        members = getattr(judge, "_judges", ())
        if require_window or any(hasattr(m, "claim_tokens") for m in members):
            raise T4OrganizerFault(
                "the NLI judge cannot count claim tokens (no `claim_tokens`), so the claim "
                "length guard cannot be applied. Build the judge with "
                "judge_factory.build_production_judge or faithfulness.judge.build_judge."
            )
        return False
    return int(count(text)) > CLAIM_MAX_JUDGE_TOKENS


def evaluate_claims(
    aligned: AlignedPredictions,
    corpus_lookup: Callable[[str], Mapping[str, Any]],
    judge: Any,
    *,
    target_type: str,
    interval_scored: bool,
    contradiction_bar: float,
    entity_admits: Callable[[str, Any], bool] | None = None,
    require_window: bool = False,
    figure_rule: str | None = None,
    entity_names: Sequence[str] = (),
    interval_level: float | None = None,
) -> ClaimReport:
    """The per-claim faithfulness verdicts (5.2.0): which claims are demonstrably false.

    Per claim, in roster order:

    1. Citation facts. ``wrong_entity``: a citation names a document the manifest does not label
       for the claim's entity nor mark shared (`entity_admits`; None means every document is
       admitted, for callers that ran no entity check). ``out_of_range``: a citation's offsets
       are not a real slice of its document (the offsets guard, `_span_text`); the claim is still read on its
       other citations.
    2. A claim with no usable text (empty, or over `CLAIM_TEXT_MAX_CHARS`) is ``malformed``.
    3. The numeric backstop (`numeric.claim_number_status`, unchanged in 5.2.0): a claim whose
       figures appear in none of the passages it cites is ``unanchored``. The participant's own
       SCORED values for the entity (`_scored_own_values`) are not figures the passage is
       expected to carry.
    4. Otherwise the judge is asked, with each cited passage as premise and the claim text as
       hypothesis, for its three-way P(contradiction) (`judged_contradiction`); the best over
       the claim's passages above `contradiction_bar` makes it ``contradicted``, anything else
       is ``neutral``.

    A cited passage is what the judge reads of it: every passage is cut to the judge's window
    before either check reads it (the judged-window guard, `judged_passage`). A claim is FALSE if 1 finds anything or
    its status is malformed, unanchored or contradicted. Generic or content-free text is neither
    entailed nor contradicted, so it is neutral: it earns nothing here and costs nothing; whether
    evidence supports the forecast is reasoning grading's question.
    """
    applied = contradiction_supported(judge)
    if not applied and require_window:
        raise T4OrganizerFault(
            "the rankable NLI judge cannot answer the three-way contradiction question (a member exposes no "
            "`contradiction`), so the 5.2.0 faithfulness penalty cannot be computed. Build it with "
            "judge_factory.build_production_judge."
        )
    verdicts: list[ClaimVerdict] = []
    out_of_range = window_cut = 0
    for index, entity_id in enumerate(aligned.entity_ids):
        submitted = _scored_own_values(
            aligned, index, target_type=target_type, interval_scored=interval_scored
        )
        # 5.2.1: a "±" half-width of the own interval is exempt only when the interval is scored.
        intervals = ((aligned.lo[index], aligned.hi[index]),) if interval_scored else ()
        for claim in aligned.claims_by_entity[index]:
            wrong = entity_admits is not None and any(
                not entity_admits(entity_id, cite) for cite in claim.citations
            )
            if not claim.malformed and _claim_too_long(
                judge, claim.text, require_window=require_window
            ):
                # Counted as malformed (false); the judge never reads it.
                claim = _dc_replace(claim, malformed=True)
            if claim.malformed:
                bad_here = sum(
                    _span_text(cite, corpus_lookup)[1] for cite in claim.citations
                )
                out_of_range += bad_here
                verdicts.append(
                    ClaimVerdict(
                        entity_id, claim.text, "malformed", 0.0, wrong, bad_here > 0
                    )
                )
                continue
            whole, bad = _cited_spans(claim.citations, corpus_lookup)
            out_of_range += bad
            # One cut, read by both checks below: the figure check sees what the judge sees.
            spans = [
                judged_passage(judge, span, claim.text, require_window=require_window)
                for span in whole
            ]
            window_cut += sum(1 for a, b in zip(whole, spans, strict=True) if a != b)
            rule = FIGURE_RULE if figure_rule is None else figure_rule
            if rule not in FIGURE_RULES:
                raise T4OrganizerFault(
                    f"figure_rule {rule!r} is not one of {list(FIGURE_RULES)}"
                )
            if rule == "every":
                # The every-figure rule: EVERY figure, read against the WHOLE cited span (not the judge window)
                status = claim_number_status(
                    claim.text,
                    whole,
                    submitted=submitted,
                    rule="every",
                    names=entity_names,
                    span_cap=FIGURE_SPAN_CAP,
                    own_intervals=intervals,
                    own_levels=(interval_level,),
                )
            else:
                status = claim_number_status(claim.text, spans, submitted=submitted)
            if status == "unanchored":
                verdicts.append(
                    ClaimVerdict(
                        entity_id, claim.text, "unanchored", 0.0, wrong, bad > 0
                    )
                )
                continue
            best = 0.0
            if verbatim_quote(claim.text, whole):
                # A word-for-word quote of a passage it cites states what that
                # passage states; the judge is not asked (measured: participant verbatim quotes
                # reach P(contradiction) 0.883 on long table quotes, the closest class to the bar).
                spans = []
            for premise in spans if applied else ():
                if not premise.strip() or not claim.text.strip():
                    continue
                best = max(
                    best,
                    judged_contradiction(
                        judge, premise, claim.text, require_window=require_window
                    ),
                )
            verdicts.append(
                ClaimVerdict(
                    entity_id,
                    claim.text,
                    "contradicted" if best > contradiction_bar else "neutral",
                    best,
                    wrong,
                    bad > 0,
                )
            )
    return ClaimReport(
        tuple(verdicts),
        out_of_range_citation_count=out_of_range,
        window_cut_citation_count=window_cut,
        contradiction_applied=applied,
    )


# --------------------------------------------------------------------------- #
# Composite                                                                     #
# --------------------------------------------------------------------------- #


def _paraphrase_faithfulness(
    claims: list[dict[str, Any]],
    corpus_lookup: Callable[[str], Mapping[str, Any]],
    judge: Any,
    *,
    tau: float,
) -> float:
    """Fraction of predictions supported, aggregating over PHRASINGS by median.

    THIS IS THE `prediction_relevance` DIAGNOSTIC AND NOT THE GATE. The gate is
    `evaluate_claims`. Everything below describes how the diagnostic is computed.

    A prediction is supported when the MEDIAN, across the phrasings of that same prediction, of
    the best entailment over its own cited spans exceeds `tau`.

    Why the median and why at all. An NLI cross-encoder's entailment probability moves with
    wording even when the content asserted is identical, so a claim sitting near `tau` can cross
    it on the rewording alone: a single phrasing is a noisy instrument at exactly the margin
    where admissibility is decided. The judge already ensembles over two MODELS for the same
    reason; this ensembles over phrasing, which was not damped at all.

    MEDIAN, not max: max is mechanically >= any single phrasing and would raise admissions for a
    reason unrelated to grounding. The point is not to admit more, it is to stop admissibility
    swinging on which wording the organizers happened to pick.

    Aggregated PER CLAIM rather than per unit, because the noise is per claim: a unit-level
    median over three faithfulness fractions can hide one claim being lifted by one phrasing and
    another being sunk by a different one.

    The Hub still owns the primitive. Entailment is `judge.entail`, unchanged; Track 4 owns only
    the semantics of what is asserted and how the phrasings are combined, which is the ownership
    split in global rule 3.

    This reads the citation and the document itself rather than calling
    `F.citation_faithfulness`, because that primitive aggregates to a unit fraction and the median
    has to be taken per claim, before any thresholding. The consequence is that Track 4 now parses
    a citation span for the first time, and it parses NARROWLY: `span_start`/`span_end` and a
    document with a `text` key. The shared primitive is more tolerant -- it also accepts a
    `span: [s, e]` citation and a document that is a bare string or is `spans`-shaped -- so the two
    would disagree if such an input ever reached here.

    None can, and both exclusions are ENFORCED rather than merely observed:
      * `span: [s, e]` is refused by `g1_schema`, which runs before this gate: analysis.schema.json
        requires `span_start` and `span_end` on every citation, and validation reports both as
        missing. `test_the_published_schema_still_requires_span_start_and_span_end` fails if that
        `required` list is ever relaxed, because relaxing it would silently change scores here --
        such a citation would reach this code and support nothing, where before it scored normally.
      * A non-object document is refused at corpus load with an organizer fault ("is not a JSON
        object"), so `document` is always a Mapping, and `CorpusIndex` reads a document's body
        from its `text` key; nothing in this repository authors a `spans`-shaped document.
    """
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        texts = claim.get("texts") or [claim["text"]]
        per_phrasing: list[float] = []
        for text in texts:
            best = 0.0
            for cite in claim.get("citations", ()):
                premise, _ = _span_text(cite, corpus_lookup)
                if premise:
                    best = max(best, float(judge.entail(premise, text)))
            per_phrasing.append(best)
        supported += int(statistics.median(per_phrasing) > tau)
    return supported / len(claims)


def _true_vectors(
    realized: Mapping[str, Any], roster: EntityRoster, *, target_type: str
) -> tuple[list[str], list[float] | None]:
    """Truth in TRUSTED ROSTER ORDER. The reference must cover the roster exactly.

    `target_type` is REQUIRED and has no default on purpose: it is the trusted declaration this
    function cross-checks the reference against, and a default would let a caller opt out of the
    check by omission — which is how the gap below existed in the first place.
    """
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
    #   NO row numeric     -> pure-label unit. The calibration leg is DROPPED: there is no
    #                         numeric truth for an interval to cover, and the sealed scorer
    #                         applies the same rule
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
    # The fourth case, and the one that was missing: "pure-label" was decided from the OUTCOME
    # alone. A `regression` or `ranking` unit whose reference lost its `y` values was therefore
    # accepted as pure-label, and dropping the calibration leg RAISES what a worthless submission
    # scores. Measured on a synthetic four-entity unit with an all-zero forecast and zero-width
    # intervals (under 5.0.0): regression -0.2700 -> +0.0000, ranking +0.0800 -> +0.3500, both
    # above that version's worst value -0.27. The pre-committed worst value is W = 0.0 from 5.1.0.
    #
    # The cross-check lives here and nowhere else. The shared
    # `predictive_quality` rule that a ranking with fewer than two comparable rows scores 0.5 is
    # CORRECT and documented for genuinely small rosters; it is untouched, and once this refuses
    # a numeric-target unit with non-numeric truth, the ranking metric never receives that vector.
    if not numeric_rows and target_type in ("regression", "ranking"):
        raise T4OrganizerFault(
            f"the unit declares target_type {target_type!r} but its resolved outcome carries no "
            f"numeric target on any of {len(values)} roster entities. That is a reference "
            "defect, not a pure-label unit: scored as pure-label the calibration leg is dropped, "
            "which pays a worthless submission more than the same submission on a correct "
            "reference. Only `classification` may be pure-label."
        )
    return labels, (values if numeric_rows else None)


def _apply_faithfulness_penalty(
    parts: dict[str, float | None], ctx: Mapping[str, Any]
) -> dict[str, float | None]:
    """The unit's composite times its faithfulness factor (5.2.0), with both recorded.

    `composite_before_penalty` is `_composite`'s number; `faithfulness_factor` is the soft floor
    ``(1 - F/(F + min(T, 3E))) ** penalty_k`` from g3 (1.0 with no claims or no false claims; on the
    smoke path F counts only the deterministic reasons and the factor is still applied); `composite` is their product. A factor g3 never set is an organizer
    fault: a scored unit passed g3, so its absence means the contract changed.
    """
    factor = ctx.get("_faithfulness_factor")
    if not isinstance(factor, float) or not 0.0 <= factor <= 1.0:
        raise T4OrganizerFault(
            "a unit reached scoring without a faithfulness factor in [0, 1] from g3"
        )
    before = _composite_value(parts)
    out = dict(parts)
    out["composite_before_penalty"] = before
    out["faithfulness_factor"] = factor
    out["composite"] = float(before * factor)
    return out


def _composite_value(parts: Mapping[str, float | None]) -> float:
    """The composite, narrowed to a float.

    Only the interval keys (`interval_coverage`, `interval_score`, `naive_interval_score`,
    `raw_interval_quality`, `interval_quality`) are nullable in a `_composite` result -- they are
    None exactly for a pure-label unit, where the interval leg does not apply. The composite itself
    is always a number, and a None here would mean `_composite` returned something it has no branch
    for, so this raises rather than letting `clip_to_domain` receive a None it would crash on later.
    """
    value = parts["composite"]
    if value is None:
        raise T4OrganizerFault(
            "the scorer produced no composite for a unit it reported as scored"
        )
    return float(value)


def mean_interval_score(
    lo: Sequence[float], hi: Sequence[float], y: Sequence[float], interval_level: float
) -> float:
    """Mean interval score over every roster row: the toolkit's
    ``qfbench2_common.scoring.faithfulness.mean_interval_score``, re-exported here.

    The formula lives in the toolkit (moved there 2026-09-28); this wrapper only keeps the track's
    fault class. The toolkit raises its base ``OrganizerFault`` on an empty roster or misaligned
    lo/hi/y; the track re-raises that as ``T4OrganizerFault`` so the run aborts as every other
    track organizer fault does.
    """
    try:
        return F.mean_interval_score(lo, hi, y, interval_level)
    except T4OrganizerFault:
        raise
    except OrganizerFault as exc:
        raise T4OrganizerFault(str(exc)) from exc


def anchored_quality(quality: float, anchor: float) -> float:
    """5.2.0: a classification / ranking quality in [0, 1] mapped piecewise linearly so
    that 0 -> 0, the anchor -> 0.5 and a perfect answer -> 1. An anchor of 1 (the naive rule is already
    perfect) gives 1.0 only to a perfect answer and 0.5 * q otherwise; an anchor of 0 gives 0.5 below it."""
    if quality >= anchor:
        if anchor >= 1.0 - 1e-12:
            return 1.0
        return 0.5 + 0.5 * (quality - anchor) / (1.0 - anchor)
    return 0.5 * quality / anchor if anchor > 0.0 else 0.5


#: 5.2.0 safeguard (adopted 2026-09-29): the anchor is the stronger of the declared naive rule and a
#: trivial constant rule known ex ante. For ranking that is a constant forecast, whose Spearman quality is
#: 0.5 by `predictive_quality`'s own definition; classification has no ex-ante constant rule (no unit
#: publishes a pre-cutoff label history), so its anchor is the declared naive.
CONSTANT_RULE_QUALITY = {"ranking": 0.5}


def _anchored_to_naive(
    target_type: str,
    quality: float,
    naive_aligned: AlignedPredictions | None,
    true_labels: list[str],
    true_values: list[float] | None,
) -> tuple[float, dict[str, float | None]]:
    """The anchored prediction leg and its record: raw quality, the declared naive's quality, the anchor."""
    if naive_aligned is None:
        raise T4OrganizerFault(
            f"a {target_type} unit reached scoring without its declared naive rule "
            f"(reference/{NAIVE_ANSWER_FILE}); from 5.2.0 its prediction leg is anchored to that "
            "rule, so it is a missing scoring parameter"
        )
    naive_quality = float(
        F.predictive_quality(
            target_type,
            list(naive_aligned.pred_labels),
            true_labels,
            list(naive_aligned.pred_values),
            true_values if true_values is not None else [],
        )
    )
    constant = CONSTANT_RULE_QUALITY.get(target_type)
    anchor = naive_quality if constant is None else max(naive_quality, constant)
    return anchored_quality(quality, anchor), {
        "raw_predictive_quality": quality,
        "naive_predictive_quality": naive_quality,
        "predictive_anchor": anchor,
    }


def _composite(
    aligned: AlignedPredictions,
    realized: Mapping[str, Any],
    params: ScoringParams,
    roster: EntityRoster,
    *,
    naive_aligned: AlignedPredictions | None = None,
) -> dict[str, float | None]:
    true_labels, true_values = _true_vectors(
        realized, roster, target_type=params.target_type
    )
    naive_values: list[float] | None = None
    if params.target_type == "regression":
        if naive_aligned is None:
            raise T4OrganizerFault(
                f"a regression unit reached scoring without its declared naive rule "
                f"(reference/{NAIVE_ANSWER_FILE}); that is a missing scoring parameter"
            )
        naive_values = list(naive_aligned.pred_values)
    quality = F.predictive_quality(
        params.target_type,
        list(aligned.pred_labels),
        true_labels,
        list(aligned.pred_values),
        true_values if true_values is not None else [],
        naive_values=naive_values,
    )
    anchor_parts: dict[str, float | None] = {}
    if params.target_type in ("classification", "ranking"):
        quality, anchor_parts = _anchored_to_naive(
            params.target_type, float(quality), naive_aligned, true_labels, true_values
        )
    w_a, w_c = params.composite_weights

    if true_values is None or not _interval_leg_scored(params, numeric_truth=True):
        # Pure-label unit, or a classification unit that DECLARED its interval leg off
        # (`interval_leg = false`): no interval leg. Reported as None rather than 0.0, so a
        # reader can tell "not applicable" from "measured, and it was zero" -- the same distinction
        # `scored` draws in the shared aggregate. 5.2.0: the composite is the prediction
        # leg alone, not w_a times it, so a unit without an interval leg is not capped at w_a.
        return {
            "predictive_quality": float(quality),
            **anchor_parts,
            "interval_coverage": None,
            "interval_score": None,
            "naive_interval_score": None,
            "raw_interval_quality": None,
            "interval_quality": None,
            "composite": float(quality),
        }

    # Every row below is a roster row. Alignment already guaranteed every lo/hi/level is finite and
    # ordered, so there is no row to drop and no denominator to shrink.
    if naive_aligned is None:
        raise T4OrganizerFault(
            f"a unit with numeric truth reached scoring without its declared naive rule "
            f"(reference/{NAIVE_ANSWER_FILE}); the interval leg is measured against the naive "
            "interval, so that is a missing scoring parameter"
        )
    zero_width = sum(
        1 for lo, hi in zip(naive_aligned.lo, naive_aligned.hi) if not hi > lo
    )
    naive_is = mean_interval_score(
        naive_aligned.lo, naive_aligned.hi, true_values, params.interval_level
    )
    if zero_width or not naive_is > 0.0:
        raise T4OrganizerFault(
            f"the declared naive rule (reference/{NAIVE_ANSWER_FILE}) has a zero-width interval "
            f"on {zero_width} of {roster.count} roster rows (naive interval score "
            f"{'zero' if not naive_is > 0.0 else 'positive'}); a naive rule without an interval "
            "on every row cannot anchor the interval leg, and that is an organizer fault"
        )
    own_is = mean_interval_score(
        aligned.lo, aligned.hi, true_values, params.interval_level
    )
    raw_interval_quality = naive_is / (naive_is + own_is)
    # 5.2.1: interval credit above the naive rule's 0.5 never exceeds the prediction leg's own
    # credit above 0.5. The declared naive bands can be wide, so narrowing the band around the
    # naive point used to score above the naive rule with no information. With this cap an answer
    # whose points are the naive rule's (quality exactly 0.5) scores at most 0.5 whatever its band.
    # Below 0.5 the interval leg is unchanged, so a bad band still costs.
    interval_quality = min(raw_interval_quality, max(0.5, float(quality)))
    # Coverage is no longer scored; it stays as an operator diagnostic.
    covered = sum(
        1 for lo, hi, y in zip(aligned.lo, aligned.hi, true_values) if lo <= y <= hi
    )
    coverage = covered / roster.count
    composite = w_a * quality + w_c * interval_quality
    return {
        "predictive_quality": float(quality),
        **anchor_parts,
        "interval_coverage": float(coverage),
        "interval_score": float(own_is),
        "naive_interval_score": float(naive_is),
        "raw_interval_quality": float(raw_interval_quality),
        "interval_quality": float(interval_quality),
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
    # A rankable judge must expose its window (the judged-window guard); see `judged_passage`.
    ctx["_require_judge_window"] = judge_provenance.rankable
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
            # `_faithfulness` is carried out of the failure path when the gate that refused had
            # already computed it -- an EVIDENCE_UNSUPPORTED refusal populates it BEFORE raising.
            # Discarding it threw away the one number that says HOW FAR a submission was from
            # grounding its predictions, exactly where that matters most: for an operator asking
            # why a unit failed.
            # It is `None`, never 0.0, when the unit failed before g3 ran -- "not computed" and
            # "measured zero" are different facts and a consumer must be able to tell them apart.
            #
            # This is `diagnostics`, which is OPERATOR-ONLY and never serialized into a
            # participant-visible artifact; the participant projection is `detail`, which admits
            # an enum code and non-negative integers only. So this discloses nothing to a
            # participant, and in particular does not tell one how close it came.
            diagnostics={
                "reason": failure.reason.value,
                "faithfulness": ctx.get("_faithfulness"),
                **_claim_diagnostics(ctx),
            },
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
                **_claim_diagnostics(ctx),
            },
        )

    parts = _apply_faithfulness_penalty(
        _composite(
            ctx["_aligned"],
            realized,
            params,
            ctx["_roster"],
            naive_aligned=ctx.get("_naive_aligned"),
        ),
        ctx,
    )
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
            **_claim_diagnostics(ctx),
            "expected_entity_count": ctx["_roster"].count,
            "graded_entity_count": ctx["_aligned"].count,
            **(ctx.get("_naive_provenance") or {}),
            **parts,
        },
    )


def _claim_diagnostics(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """Operator-only claim counts and the recorded prediction-relevance diagnostic.

    `prediction_relevance` is `None`, never 0.0, when g3 did not reach the judge: "not computed"
    and "measured zero" are different facts (see `test_faithfulness_survives_refusal`).
    """
    out: dict[str, Any] = {"prediction_relevance": ctx.get("_prediction_relevance")}
    report = ctx.get("_claim_report")
    if isinstance(report, ClaimReport):
        out.update(report.diagnostics())
    return out


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
                "prediction_relevance": scoring_ctx.get("_prediction_relevance"),
                "faithfulness_gate_applied": scoring_ctx.get(
                    "_faithfulness_gate_applied", True
                ),
                "note": "no resolved outcome (public practice unit)",
            }
        parts = _apply_faithfulness_penalty(
            _composite(
                scoring_ctx["_aligned"],
                realized,
                params,
                scoring_ctx["_roster"],
                naive_aligned=scoring_ctx.get("_naive_aligned"),
            ),
            scoring_ctx,
        )
        return {
            "score": clip_to_domain(_composite_value(parts)),
            "rankable": provenance.rankable,
            "judge_mode": provenance.judge_mode,
            "faithfulness": scoring_ctx.get("_faithfulness"),
            "prediction_relevance": scoring_ctx.get("_prediction_relevance"),
            "faithfulness_gate_applied": scoring_ctx.get(
                "_faithfulness_gate_applied", True
            ),
            "expected_entity_count": scoring_ctx["_roster"].count,
            "graded_entity_count": scoring_ctx["_aligned"].count,
            **(scoring_ctx.get("_naive_provenance") or {}),
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
    ctx["_require_judge_window"] = provenance.rankable
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
