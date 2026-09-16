"""Exact entity-set equality and the numeric contract, both enforced before any metric runs.

The measured defect this module exists to remove: alignment iterated the **participant's**
``entity_predictions`` and did ``if outcome is None: continue``. Everything followed from that one
line. Answering one of three required entities scored ``0.67`` where the honest full-roster answer
scored ``0.203`` — a 3.3x gain for answering less — because the truth vector was rebuilt from
whatever the participant chose to answer. Duplicating an entity re-consumed the same truth row and
reweighted accuracy and coverage; an unknown entity was silently dropped.

The shared ``predictive_quality`` already enforces coverage against the truth vector it is handed.
That protection was a no-op because Track 4 handed it a truth vector derived from the submission.
This module fixes the input, not the shared math:

* alignment is driven by the **trusted roster**, in the trusted canonical order;
* the participant's entity set must be **exactly** the roster set — unique, complete, nothing extra;
* a missing, duplicated or unknown id fails before a metric is computed, so no aggregate is ever
  produced from a re-weighted denominator;
* NaN/Inf anywhere in a submitted number is a participant failure. Rows are **never dropped** to
  make a denominator smaller.

`target_type`, `interval_level` and the label vocabulary come from the trusted task/plan side. A
value read out of the mounted unit directory is only ever used to *cross-check* and never to
decide: the mounted card sits in the same tree the participant's container had.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .codes import T4OrganizerFault, T4ParticipantFailure, T4Reason

__all__ = [
    "TARGET_TYPES",
    "AlignedPredictions",
    "EntityRoster",
    "align_predictions",
    "validated_label_vocabulary",
]

#: Closed. An unknown target type is refused rather than defaulting to classification — the old
#: fallback meant a card typo silently changed which metric the leaderboard reported.
TARGET_TYPES: tuple[str, ...] = ("classification", "regression", "ranking")


def validated_label_vocabulary(
    raw: object, target_type: object, *, source: str
) -> tuple[str, ...]:
    """The one rule for what a label vocabulary is, for both roster sources.

    `EntityRoster.labels` is built from `task.json` on the local path and from the signed C1 entry
    on the platform path. Those two used to disagree: the plan side refused a one-label, duplicate,
    padded or wrongly-typed vocabulary, and the task side accepted all four and fell through to
    ``labels = None`` — which SKIPS the label check — whenever it could not read one at all. A
    submission could therefore be refused locally and scored on the platform, or the reverse.

    Both sources are organizer material, so a malformed vocabulary is an organizer fault on either.
    Stating the rule once is what stops the two drifting apart again; `scoring/tests/
    test_label_vocabulary_symmetry.py` hands the same vocabulary to both and asserts one verdict.

    The rule: a vocabulary belongs only to a ``classification`` unit, and it is two or more unique,
    non-empty, unpadded strings. Absence is not malformation — a regression or ranking unit carries
    no vocabulary by design, and neither caller reaches this function for one.
    """
    if target_type != "classification":
        raise T4OrganizerFault(
            f"{source} carries a label vocabulary on a {target_type!r} unit; "
            "only classification units have one"
        )
    if (
        not isinstance(raw, list)
        or len(raw) < 2
        or not all(isinstance(x, str) and x and x == x.strip() for x in raw)
        or len(set(raw)) != len(raw)
    ):
        raise T4OrganizerFault(
            f"{source} must be two or more unique non-empty strings; this is organizer "
            "material, so a malformed vocabulary is ours, not the participant's"
        )
    return tuple(raw)


@dataclass(frozen=True, slots=True)
class EntityRoster:
    """The trusted, ordered entity ids for one unit, plus the closed label vocabulary."""

    entity_ids: tuple[str, ...]
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.entity_ids:
            raise T4OrganizerFault("an entity roster may not be empty")
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise T4OrganizerFault("the trusted entity roster repeats an entity_id")
        for entity_id in self.entity_ids:
            if not isinstance(entity_id, str) or not entity_id:
                raise T4OrganizerFault("a trusted entity_id must be a non-empty string")
            if unicodedata.normalize("NFC", entity_id) != entity_id:
                raise T4OrganizerFault(
                    f"trusted entity_id {entity_id!r} is not NFC-normalized"
                )

    @property
    def count(self) -> int:
        return len(self.entity_ids)

    @classmethod
    def from_task(cls, task: Mapping[str, Any]) -> EntityRoster:
        """Build the roster from the organizer-authored ``task.json``.

        `task.json` is organizer material staged into the unit; it is the roster source for the
        public practice path. In a sealed run the roster comes from C1 (`expected_units[]
        .entity_roster`) and is passed in directly — see `build_verifier`'s ``entity_roster``
        context key, which takes precedence over anything read from the tree.
        """
        entities = task.get("entities")
        if not isinstance(entities, list) or not entities:
            raise T4OrganizerFault(
                "task.json declares no entities[]; there is no roster to score"
            )
        ids: list[str] = []
        for index, entity in enumerate(entities):
            if not isinstance(entity, Mapping) or not isinstance(
                entity.get("entity_id"), str
            ):
                raise T4OrganizerFault(
                    f"task.json entities[{index}] has no string entity_id"
                )
            ids.append(entity["entity_id"])
        target = task.get("target")
        labels: tuple[str, ...] | None = None
        if isinstance(target, Mapping) and "labels" in target:
            labels = validated_label_vocabulary(
                target["labels"], target.get("type"), source="task.json target.labels"
            )
        return cls(entity_ids=tuple(ids), labels=labels)


@dataclass(frozen=True, slots=True)
class AlignedPredictions:
    """Per-entity arrays in **trusted roster order**, one element per roster entry, no gaps."""

    entity_ids: tuple[str, ...]
    pred_labels: tuple[str, ...]
    pred_values: tuple[float, ...]
    lo: tuple[float, ...]
    hi: tuple[float, ...]
    ranks: tuple[int | None, ...]
    citations_by_entity: tuple[tuple[Mapping[str, Any], ...], ...]
    claim_texts_by_entity: tuple[tuple[str, ...], ...]

    @property
    def count(self) -> int:
        return len(self.entity_ids)

    def all_citations(self) -> list[Mapping[str, Any]]:
        return [cite for group in self.citations_by_entity for cite in group]


def _finite(value: Any, *, what: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise T4ParticipantFailure(
            T4Reason.NONFINITE_VALUE, f"{what} is not a number", nonfinite_count=1
        ) from exc
    if not math.isfinite(out):
        raise T4ParticipantFailure(
            T4Reason.NONFINITE_VALUE,
            f"{what} is not finite; NaN and Inf are participant data faults and the row is never "
            "dropped to hide them",
            nonfinite_count=1,
        )
    return out


def _entity_index(
    entity_predictions: Sequence[Any], roster: EntityRoster
) -> dict[str, Mapping[str, Any]]:
    """Exact unique set equality with the trusted roster, or a participant failure with counts."""
    seen: dict[str, Mapping[str, Any]] = {}
    duplicates = 0
    unknown = 0
    malformed = 0
    roster_set = set(roster.entity_ids)
    for row in entity_predictions:
        if not isinstance(row, Mapping):
            malformed += 1
            continue
        entity_id = row.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            malformed += 1
            continue
        if unicodedata.normalize("NFC", entity_id) != entity_id:
            unknown += 1
            continue
        if entity_id not in roster_set:
            unknown += 1
            continue
        if entity_id in seen:
            duplicates += 1
            continue
        seen[entity_id] = row
    if malformed:
        raise T4ParticipantFailure(
            T4Reason.SCHEMA_INVALID,
            "entity_predictions contains rows with no usable entity_id",
            invalid_row_count=malformed,
            expected_count=roster.count,
        )
    if duplicates:
        raise T4ParticipantFailure(
            T4Reason.ENTITY_DUPLICATE,
            "entity_predictions repeats an entity_id; a duplicate re-consumes the same truth row "
            "and reweights the metric",
            invalid_row_count=duplicates,
            expected_count=roster.count,
            observed_count=len(entity_predictions),
        )
    if unknown:
        raise T4ParticipantFailure(
            T4Reason.ENTITY_UNKNOWN,
            "entity_predictions names entities that are not on the trusted roster",
            extra_count=unknown,
            expected_count=roster.count,
            observed_count=len(entity_predictions),
        )
    missing = [eid for eid in roster.entity_ids if eid not in seen]
    if missing:
        raise T4ParticipantFailure(
            T4Reason.ENTITY_MISSING,
            "entity_predictions omits entities on the trusted roster; the roster fixes the graded "
            "set and answering a favourable subset can never shrink the denominator",
            missing_count=len(missing),
            expected_count=roster.count,
            observed_count=len(seen),
        )
    return seen


def _interval(
    row: Mapping[str, Any], entity_id: str, interval_level: float
) -> tuple[float, float]:
    interval = row.get("interval")
    if not isinstance(interval, Mapping):
        raise T4ParticipantFailure(
            T4Reason.INTERVAL_INVALID,
            f"entity {entity_id} carries no interval object",
            invalid_row_count=1,
        )
    if "level" not in interval:
        raise T4ParticipantFailure(
            T4Reason.INTERVAL_LEVEL_MISMATCH,
            f"entity {entity_id} carries no interval level; an absent level is an error, never the "
            "declared one",
            invalid_row_count=1,
        )
    level = _finite(interval.get("level"), what=f"entity {entity_id} interval.level")
    if level != float(interval_level):
        raise T4ParticipantFailure(
            T4Reason.INTERVAL_LEVEL_MISMATCH,
            f"entity {entity_id} reports interval level {level} but the unit requires "
            f"{interval_level}",
            invalid_row_count=1,
        )
    lo = _finite(interval.get("lo"), what=f"entity {entity_id} interval.lo")
    hi = _finite(interval.get("hi"), what=f"entity {entity_id} interval.hi")
    if lo > hi:
        raise T4ParticipantFailure(
            T4Reason.INTERVAL_INVALID,
            f"entity {entity_id} reports an inverted interval (lo > hi)",
            invalid_row_count=1,
        )
    return lo, hi


def _citations(
    row: Mapping[str, Any], entity_id: str
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    claims = row.get("claims")
    if not isinstance(claims, list) or not claims:
        raise T4ParticipantFailure(
            T4Reason.CITATION_MALFORMED,
            f"entity {entity_id} carries no claims[]; every prediction must be evidence-grounded",
            invalid_row_count=1,
        )
    cites: list[Mapping[str, Any]] = []
    texts: list[str] = []
    for claim in claims:
        if not isinstance(claim, Mapping):
            raise T4ParticipantFailure(
                T4Reason.CITATION_MALFORMED,
                f"entity {entity_id} has a non-object claim",
                invalid_row_count=1,
            )
        if isinstance(claim.get("citations"), list):
            # Single-entity ("camp A") shape: the claim carries an explicit citations[].
            nested = [c for c in claim["citations"] if isinstance(c, Mapping)]
            if len(nested) != len(claim["citations"]) or not nested:
                raise T4ParticipantFailure(
                    T4Reason.CITATION_MALFORMED,
                    f"entity {entity_id} has a claim whose citations[] is empty or malformed",
                    invalid_row_count=1,
                )
            cites.extend(nested)
        else:
            # Multi-entity ("camp B") shape: the claim object IS the citation.
            cites.append(claim)
        text = (
            claim.get("claim")
            if isinstance(claim.get("claim"), str)
            else claim.get("text")
        )
        texts.append(text if isinstance(text, str) else "")
    if not cites:
        raise T4ParticipantFailure(
            T4Reason.CITATION_MALFORMED,
            f"entity {entity_id} has claims but no citations",
            invalid_row_count=1,
        )
    return tuple(cites), tuple(texts)


def align_predictions(
    answer: Mapping[str, Any],
    roster: EntityRoster,
    *,
    target_type: str,
    interval_level: float,
) -> AlignedPredictions:
    """Validate and align a submission against the trusted roster. Raises before any metric runs."""
    if target_type not in TARGET_TYPES:
        raise T4OrganizerFault(
            f"trusted target_type {target_type!r} is not one of {list(TARGET_TYPES)}"
        )
    declared = answer.get("target_type")
    if declared is not None and declared != target_type:
        raise T4ParticipantFailure(
            T4Reason.TARGET_TYPE_MISMATCH,
            "the answer declares a target_type that disagrees with the trusted task",
            invalid_row_count=1,
        )
    rows = answer.get("entity_predictions")
    if not isinstance(rows, list) or not rows:
        raise T4ParticipantFailure(
            T4Reason.SCHEMA_INVALID,
            "answer.json carries no entity_predictions[]",
            expected_count=roster.count,
            observed_count=0,
        )
    by_id = _entity_index(rows, roster)

    pred_labels: list[str] = []
    pred_values: list[float] = []
    los: list[float] = []
    his: list[float] = []
    ranks: list[int | None] = []
    cites: list[tuple[Mapping[str, Any], ...]] = []
    texts: list[tuple[str, ...]] = []

    for entity_id in roster.entity_ids:
        row = by_id[entity_id]
        lo, hi = _interval(row, entity_id, interval_level)
        los.append(lo)
        his.append(hi)

        label = row.get("label")
        if target_type == "classification":
            if not isinstance(label, str) or not label:
                raise T4ParticipantFailure(
                    T4Reason.LABEL_INVALID,
                    f"entity {entity_id} carries no label on a classification unit",
                    invalid_row_count=1,
                )
            if roster.labels is not None and label not in roster.labels:
                raise T4ParticipantFailure(
                    T4Reason.LABEL_INVALID,
                    f"entity {entity_id} reports a label outside the task's declared vocabulary",
                    invalid_row_count=1,
                )
        pred_labels.append(label if isinstance(label, str) else "")

        raw_value = row.get("point_forecast")
        if target_type in ("regression", "ranking"):
            if raw_value is None:
                raise T4ParticipantFailure(
                    T4Reason.SCHEMA_INVALID,
                    f"entity {entity_id} carries no point_forecast on a {target_type} unit",
                    invalid_row_count=1,
                )
            pred_values.append(
                _finite(raw_value, what=f"entity {entity_id} point_forecast")
            )
        elif raw_value is None:
            pred_values.append(0.0)
        else:
            pred_values.append(
                _finite(raw_value, what=f"entity {entity_id} point_forecast")
            )

        raw_rank = row.get("rank")
        if raw_rank is None:
            ranks.append(None)
        elif isinstance(raw_rank, bool) or not isinstance(raw_rank, int):
            raise T4ParticipantFailure(
                T4Reason.SCHEMA_INVALID,
                f"entity {entity_id} reports a non-integer rank",
                invalid_row_count=1,
            )
        else:
            ranks.append(raw_rank)

        entity_cites, entity_texts = _citations(row, entity_id)
        cites.append(entity_cites)
        texts.append(entity_texts)

    supplied_ranks = [r for r in ranks if r is not None]
    if supplied_ranks:
        if len(supplied_ranks) != roster.count:
            raise T4ParticipantFailure(
                T4Reason.SCHEMA_INVALID,
                "rank is supplied for some entities but not all; a partial ranking is not a ranking",
                missing_count=roster.count - len(supplied_ranks),
                expected_count=roster.count,
            )
        if sorted(supplied_ranks) != list(range(1, roster.count + 1)):
            raise T4ParticipantFailure(
                T4Reason.SCHEMA_INVALID,
                "rank must be a permutation of 1..n over the trusted roster",
                invalid_row_count=len(supplied_ranks),
                expected_count=roster.count,
            )

    return AlignedPredictions(
        entity_ids=roster.entity_ids,
        pred_labels=tuple(pred_labels),
        pred_values=tuple(pred_values),
        lo=tuple(los),
        hi=tuple(his),
        ranks=tuple(ranks),
        citations_by_entity=tuple(cites),
        claim_texts_by_entity=tuple(texts),
    )
