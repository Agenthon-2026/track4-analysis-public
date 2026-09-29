"""Track-4 reason codes, the two fault-domain exceptions, and the public projection.

Track 4 needs richer internal reasons than the twelve frozen public failure codes, and it needs
them **without** any of that richness reaching a participant. This module is the seam. A gate
raises :class:`T4ParticipantFailure` or :class:`T4OrganizerFault` carrying a :class:`T4Reason` and
a handful of integer counts; the reason drives the internal ``FailureLabel`` and the operator log,
and the only thing that crosses into participant-visible output is
``qfbench2_common.failure_labels.public_detail`` applied to ``{code, <counts>}`` — one of the twelve
frozen codes plus non-negative integers, never a string.

That is the fix for the measured defect where a single missing property produced a ~1.5 KB
``{"reason": str(e)}`` detail embedding the whole answer schema. Free text cannot be emitted from
here because no key in the public projection accepts text.

The two exception types are the hub's, subclassed rather than reinvented, so a caller that already
distinguishes ``ParticipantFailure`` from ``OrganizerFault`` keeps working:

* :class:`T4ParticipantFailure` — the submission is at fault. The unit stays in the C1 denominator
  and takes the plan's committed worst value ``W`` (``-0.27`` for analysis). It is never dropped.
* :class:`T4OrganizerFault` — organizer material or infrastructure is at fault. No participant
  score at all; at whole-evaluation scope the driver aborts rather than publishing a partial board.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from qfbench2_common.contracts import FailureCode, OrganizerFault, ParticipantFailure
from qfbench2_common.failure_labels import FailureLabel, public_detail

__all__ = [
    "COUNT_KEYS",
    "DEFAULT_CODE",
    "T4OrganizerFault",
    "T4ParticipantFailure",
    "T4Reason",
    "public_code_for",
    "public_label_for",
    "t4_public_detail",
]


class T4Reason(StrEnum):
    """Track-4 internal reasons. Operator-side only; never participant-visible."""

    # --- artifact / schema -------------------------------------------------
    NO_ANSWER = "t4.no_answer"
    MALFORMED_ANSWER = "t4.malformed_answer"
    SCHEMA_INVALID = "t4.schema_invalid"
    TASK_ID_MISMATCH = "t4.task_id_mismatch"
    TARGET_TYPE_MISMATCH = "t4.target_type_mismatch"
    # --- entity roster -----------------------------------------------------
    ENTITY_MISSING = "t4.entity_missing"
    ENTITY_UNKNOWN = "t4.entity_unknown"
    ENTITY_DUPLICATE = "t4.entity_duplicate"
    # --- numeric contract --------------------------------------------------
    NONFINITE_VALUE = "t4.nonfinite_value"
    INTERVAL_INVALID = "t4.interval_invalid"
    INTERVAL_LEVEL_MISMATCH = "t4.interval_level_mismatch"
    LABEL_INVALID = "t4.label_invalid"
    # --- evidence ----------------------------------------------------------
    CITATION_MALFORMED = "t4.citation_malformed"
    CITATION_UNRESOLVED = "t4.citation_unresolved"
    CITATION_UNDATED = "t4.citation_undated"
    CITATION_POST_CUTOFF = "t4.citation_post_cutoff"
    #: The cited document is not about the entity whose prediction cites it. An NLI model reads
    #: a claim about company A against company B's filing as entailed whenever the wording
    #: matches, so entity correspondence is checked deterministically at the DOCUMENT level,
    #: before the judge.
    CITATION_WRONG_ENTITY = "t4.citation_wrong_entity"
    #: Too many claims state figures that appear nowhere in the passage they cite. Exact code
    #: checks the numbers before the judge, because an NLI model reads "$5.9 billion" against
    #: "$5.2 billion" as consistent. Refused only when these claims alone put the supported
    #: fraction under the threshold; see `numeric.py`.
    CLAIM_NUMBER_NOT_IN_SPAN = "t4.claim_number_not_in_span"
    EVIDENCE_UNSUPPORTED = "t4.evidence_unsupported"


#: Internal reason -> the cross-track label written to the operator failure map.
_LABEL_FOR: dict[T4Reason, FailureLabel] = {
    T4Reason.NO_ANSWER: FailureLabel.SCHEMA_INVALID_OUTPUT,
    T4Reason.MALFORMED_ANSWER: FailureLabel.SCHEMA_INVALID_OUTPUT,
    T4Reason.SCHEMA_INVALID: FailureLabel.SCHEMA_INVALID_OUTPUT,
    T4Reason.TASK_ID_MISMATCH: FailureLabel.SCHEMA_INVALID_OUTPUT,
    T4Reason.TARGET_TYPE_MISMATCH: FailureLabel.SCHEMA_INVALID_OUTPUT,
    T4Reason.ENTITY_MISSING: FailureLabel.SCHEMA_INVALID_OUTPUT,
    T4Reason.ENTITY_UNKNOWN: FailureLabel.SCHEMA_INVALID_OUTPUT,
    T4Reason.ENTITY_DUPLICATE: FailureLabel.SCHEMA_INVALID_OUTPUT,
    T4Reason.NONFINITE_VALUE: FailureLabel.SCHEMA_INVALID_OUTPUT,
    T4Reason.INTERVAL_INVALID: FailureLabel.T4_MISCALIBRATED_INTERVAL,
    T4Reason.INTERVAL_LEVEL_MISMATCH: FailureLabel.SCHEMA_INVALID_OUTPUT,
    T4Reason.LABEL_INVALID: FailureLabel.SCHEMA_INVALID_OUTPUT,
    T4Reason.CITATION_MALFORMED: FailureLabel.T4_UNFAITHFUL_CITATION,
    T4Reason.CITATION_UNRESOLVED: FailureLabel.T4_UNFAITHFUL_CITATION,
    T4Reason.CITATION_UNDATED: FailureLabel.T4_STALE_EVIDENCE,
    T4Reason.CITATION_POST_CUTOFF: FailureLabel.T4_STALE_EVIDENCE,
    T4Reason.CITATION_WRONG_ENTITY: FailureLabel.T4_UNFAITHFUL_CITATION,
    T4Reason.CLAIM_NUMBER_NOT_IN_SPAN: FailureLabel.T4_UNFAITHFUL_CITATION,
    T4Reason.EVIDENCE_UNSUPPORTED: FailureLabel.T4_UNFAITHFUL_CITATION,
}

#: Internal reason -> one of the twelve frozen public codes.
#:
#: Reasons with an exact counterpart are listed; everything else falls to
#: :data:`DEFAULT_CODE`.
#:
#: **Was "the least-wrong of eleven" until failure-code registry 1.1.0.** Every unlisted reason
#: used to report ``schema_invalid``, whose published gloss says the output "did not match the
#: published output schema". That sentence was false for an entity-roster mismatch, an invalid
#: interval, a nonfinite value and unsupported evidence — four well-formed artifacts that failed a
#: gate the unit publishes separately from its schema — and it sent the participant hunting for a
#: schema defect that is not there. Track 4 filed the request; registry 1.1.0 granted it as
#: ``domain_gate_failed`` (recorded in ``qfbench2_common/contracts/MIGRATIONS.md``, which ships
#: in the installed toolkit), which is why this is a closed-enum member and not a free-form
#: string invented at a call site.
_CODE_FOR: dict[T4Reason, FailureCode] = {
    T4Reason.NO_ANSWER: FailureCode.NO_OUTPUT,
    T4Reason.MALFORMED_ANSWER: FailureCode.MALFORMED_OUTPUT,
    # A track that means *schema* still says schema: this is the reason `_g1_schema` raises when
    # the answer fails the published JSON Schema, and nothing else.
    T4Reason.SCHEMA_INVALID: FailureCode.SCHEMA_INVALID,
    T4Reason.CITATION_UNDATED: FailureCode.CUTOFF_VIOLATION,
    T4Reason.CITATION_POST_CUTOFF: FailureCode.CUTOFF_VIOLATION,
    T4Reason.ENTITY_MISSING: FailureCode.INCOMPLETE_OUTPUT,
}

#: What an unlisted reason reports: the output was well-formed and then failed one of this unit's
#: published domain checks. Deliberately the same value as the hub's
#: ``failure_labels.DEFAULT_ADMISSIBILITY_CODE``, so the two projections of one verdict — the code
#: this module stamps and the code `public_failure_code(labels)` derives — cannot disagree.
DEFAULT_CODE: FailureCode = FailureCode.DOMAIN_GATE_FAILED

#: The integer-count keys a Track-4 gate may report. A strict subset of the frozen C4 public
#: projection, so `public_detail` cannot drop one silently and a typo shows up as a test failure
#: rather than as a missing count.
COUNT_KEYS: tuple[str, ...] = (
    "missing_count",
    "extra_count",
    "invalid_row_count",
    "nonfinite_count",
    "expected_count",
    "observed_count",
    "violation_count",
)


def public_label_for(reason: T4Reason) -> FailureLabel:
    return _LABEL_FOR[reason]


def public_code_for(reason: T4Reason) -> FailureCode:
    return _CODE_FOR.get(reason, DEFAULT_CODE)


def t4_public_detail(reason: T4Reason, **counts: int) -> dict[str, Any]:
    """Build the participant-visible detail for `reason`: the frozen code plus integer counts.

    Every value goes through the hub's shared redactor on the way out, so a caller that passes an
    unexpected key or a string gets it dropped rather than published. The reason itself is
    deliberately absent from the output: it is a Track-4 identifier, and the public projection is
    closed to the twelve frozen codes.
    """
    unknown = sorted(set(counts) - set(COUNT_KEYS))
    if unknown:
        raise ValueError(
            f"{unknown} are not Track-4 public count keys; the public projection is "
            f"{list(COUNT_KEYS)} plus 'code'"
        )
    payload: dict[str, Any] = {"code": public_code_for(reason).value}
    payload.update(counts)
    redacted: dict[str, Any] = dict(public_detail(payload))
    return redacted


class T4ParticipantFailure(ParticipantFailure):
    """The submission is at fault. Stays in the denominator, takes the committed worst value W."""

    def __init__(self, reason: T4Reason, message: str = "", **counts: int) -> None:
        self.reason = reason
        self.counts: Mapping[str, int] = dict(counts)
        super().__init__(message or reason.value)

    @property
    def label(self) -> FailureLabel:
        return public_label_for(self.reason)

    @property
    def code(self) -> FailureCode:
        return public_code_for(self.reason)

    def public_detail(self) -> dict[str, Any]:
        return t4_public_detail(self.reason, **self.counts)


class T4OrganizerFault(OrganizerFault):
    """Organizer material or infrastructure is at fault. No participant score; the run aborts.

    Deliberately carries no participant-visible projection. An organizer fault is not a unit
    outcome, so there is nothing to render on a leaderboard row — turning one into a participant
    zero is exactly what the global rules forbid.
    """
