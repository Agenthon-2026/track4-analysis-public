"""The prediction-relevance DIAGNOSTIC: a hypothesis derived from the submission's values.

What this module builds is **recorded, never enforced**. The gate asks the judge about the
participant's claim text (`scoring.evaluate_claims`); the canonical sentence built here -- the
submitted label / point forecast / rank / interval rendered in trusted words -- is put to the
same judge afterwards and lands in the review queue as ``prediction_relevance``. Nothing in
admission depends on it.

Why it is a diagnostic and not the gate. Asking whether a cited passage entails the *forecast*
is asking a question with no right answer: a forecast is a statement about an outcome after the
unit's cutoff, and every corpus document predates that cutoff, so no passage can entail one.
What such a score actually measures is topical consistency, and topical consistency cannot
separate the passage a submission honestly cited from an unrelated passage of the same document
-- both are about the same entity and the same subject matter, especially once the document has
already been bound to its entity by the document-level check. There is therefore no pair of
thresholds on it that both admits honest work and refuses an irrelevant passage. The gate keeps
the claim-by-claim question it was founded on, with what it lacked added in front: the entity
check and an exact-code numeric backstop. "Does the evidence support the forecast" is a
reasoning-quality question, and it is assigned to reasoning grading, which reads the claims as
an argument rather than as a bag of sentences.

What this module provides: the three phrasings and the median aggregation (see `PHRASINGS`), the
label-assertion construction for classification units, and the one-claim-per-roster-entity
denominator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .alignment import AlignedPredictions
from .codes import T4OrganizerFault

__all__ = [
    "HypothesisSpec",
    "canonical_hypothesis",
    "prediction_claims",
]


def _fmt(value: float) -> str:
    """Stable decimal rendering, so the same prediction always yields the same hypothesis string."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


@dataclass(frozen=True, slots=True)
class HypothesisSpec:
    """The trusted half of the hypothesis: everything that is NOT the participant's prediction."""

    target_type: str
    interval_level: float
    #: Human-readable name of the quantity being predicted, from the trusted task target.
    target_name: str = "outcome"
    #: Optional trusted display names, entity_id -> name. Absent ids fall back to the id.
    entity_names: Mapping[str, str] | None = None
    #: Trusted natural-language assertion per class label, ``label -> predicate``. The predicate
    #: completes the sentence "<entity> ___", e.g. ``no_event`` ->
    #: "will not suffer a default-class event within the year after 2031-03-31". REQUIRED for a
    #: classification unit; see `assertion_for` for why a missing one refuses rather than defaults.
    label_assertions: Mapping[str, str] | None = None

    def name_for(self, entity_id: str) -> str:
        if self.entity_names and entity_id in self.entity_names:
            return f"{self.entity_names[entity_id]} ({entity_id})"
        return entity_id

    def plain_name_for(self, entity_id: str) -> str:
        """The display name without the ticker parenthetical, for the phrasings that vary it."""
        if self.entity_names and entity_id in self.entity_names:
            return self.entity_names[entity_id]
        return entity_id

    def assertion_for(self, label: str) -> str:
        """The natural-language predicate for `label`, or an organizer fault.

        Fails closed on purpose. The alternative -- falling back to the old
        "The <target name> of <entity> is <label>." construction -- is the defect this replaces,
        and a silent fallback would reintroduce it on exactly the units nobody remembered to
        author. `align_predictions` has already refused any label outside the trusted vocabulary
        by the time this is reached, so a missing assertion is always the organizers' omission and
        never a participant's.
        """
        assertion = (self.label_assertions or {}).get(label)
        if not assertion:
            raise T4OrganizerFault(
                f"no natural-language assertion is configured for class label {label!r} "
                f"(target {self.target_name!r}). A classification unit must declare "
                "target.label_assertions for every label in target.labels; scoring refuses "
                "rather than falling back to naming the target, which does not survive the "
                "permutation control."
            )
        return assertion

    @classmethod
    def from_task(
        cls, task: Mapping[str, Any], *, target_type: str, interval_level: float
    ) -> HypothesisSpec:
        target = task.get("target")
        target_name = "outcome"
        if (
            isinstance(target, Mapping)
            and isinstance(target.get("name"), str)
            and target["name"]
        ):
            target_name = target["name"].replace("_", " ")
        names: dict[str, str] = {}
        entities = task.get("entities")
        if isinstance(entities, list):
            for entity in entities:
                if (
                    isinstance(entity, Mapping)
                    and isinstance(entity.get("entity_id"), str)
                    and isinstance(entity.get("name"), str)
                    and entity["name"]
                ):
                    names[entity["entity_id"]] = entity["name"]
        assertions: dict[str, str] = {}
        if isinstance(target, Mapping) and isinstance(
            target.get("label_assertions"), Mapping
        ):
            for label, assertion in target["label_assertions"].items():
                if not isinstance(label, str) or not isinstance(assertion, str):
                    raise T4OrganizerFault(
                        "target.label_assertions must map a label string to an assertion string"
                    )
                text = assertion.strip()
                if not text:
                    raise T4OrganizerFault(
                        f"target.label_assertions[{label!r}] is empty; an empty assertion would "
                        "silently reduce the hypothesis to the bare entity name"
                    )
                assertions[label] = text
        return cls(
            target_type=target_type,
            interval_level=float(interval_level),
            target_name=target_name,
            entity_names=names or None,
            label_assertions=assertions or None,
        )


#: The phrasings a prediction is asserted in. The judge is asked all of them and the per-claim
#: entailments are aggregated by MEDIAN, because a single phrasing is a noisy instrument: an NLI
#: cross-encoder's entailment probability moves with wording even when the content asserted is
#: identical, so a claim sitting near `tau_citation` can cross it on the rewording alone. The
#: judge already ensembles over two MODELS to damp model-specific noise; phrasing is a second,
#: independent source of the same kind of noise, and it was not damped at all.
#:
#: MEDIAN, not max. Max is mechanically >= any single phrasing, so it would raise admissions for
#: reasons unrelated to grounding. The goal is not to admit more, it is to stop admissibility
#: swinging on which wording the organizers happened to pick.
#:
#: Every template asserts EXACTLY the same content -- same entity, target, value, rank and
#: interval. They differ only in word order and in whether the interval clause repeats the
#: subject. Adding a template that asserts anything more would inflate entailment for a reason
#: that has nothing to do with the submission.
#:
#: THREE, not two: a median over two values is their mean, which does not reject an outlier.
PHRASINGS: tuple[str, ...] = ("canonical", "subject_first", "value_first")


def canonical_hypothesis(
    spec: HypothesisSpec,
    *,
    phrasing: str = "canonical",
    entity_id: str,
    label: str,
    point_forecast: float,
    rank: int | None,
    lo: float,
    hi: float,
) -> str:
    """The one sentence the judge is asked about, built only from trusted schema + submitted values.

    Deterministic by construction: the same submitted numbers always produce the same string, so a
    re-run of the same submission asks the judge the same question. Nothing the participant wrote
    in prose appears anywhere in it.

    CLASSIFICATION IS BUILT DIFFERENTLY, and the reason is structural. It used to read
    "The <target name> of <entity> is <label>." -- which asks the judge about a machine token
    inside a sentence dominated by the target's own name. A classification target routinely
    shares vocabulary with its own labels (a `credit_event_12m` target whose labels are
    `credit_event` and `no_event` is the ordinary shape, not a corner case), and then that
    sentence is a topic word wearing a prediction's clothes: a "credit event 12m ... is no_event"
    hypothesis put against a passage reporting a company's liquidity is not entailed by it, which
    is the right answer to the question asked and the wrong answer to the question meant. The
    entailment score then tracks whether the label token echoes the passage's vocabulary rather
    than whether the evidence says anything, so predictions separate by which label was chosen
    and not by what was cited -- a separation that disappears as soon as the same predictions are
    asserted in plain English.

    So a classification hypothesis now asserts the LABEL'S MEANING and never names the target:

        "Northwind Air Holdings (NWA) will not suffer a credit event
         within the year after 2031-03-31."

    The predicate is trusted organizer data (`target.label_assertions`), authored from the task
    definition and never from the corpus -- an assertion phrased to echo corpus vocabulary would
    lift entailment for exactly the wrong reason and rebuild this defect in a subtler form.

    NO INTERVAL CLAUSE ON CLASSIFICATION, which is a change of what is asserted and is deliberate.
    The clause read "The 90% prediction interval for the <target> of <entity> is 0.01 to 0.18",
    asserting an interval *on a category*. Its true referent is not even constant across these
    units -- for some it bounds a probability, for others a predicted EPS or an abnormal return --
    and no corpus passage states a forecaster's interval in any case, so the clause could only ever
    depress entailment while naming the target one more time. The interval is still scored: it is
    the `coverage` term of the composite, which is where an interval belongs. Regression and
    ranking keep their interval clause, where it bounds the same quantity the sentence asserts.
    """
    if phrasing not in PHRASINGS:
        raise T4OrganizerFault(
            f"unknown phrasing {phrasing!r}; known: {list(PHRASINGS)}"
        )
    if spec.target_type == "classification":
        return _classification_hypothesis(
            spec, phrasing=phrasing, entity_id=entity_id, label=label
        )
    subject = spec.name_for(entity_id)
    level_pct = _fmt(spec.interval_level * 100.0)
    target = spec.target_name
    # Regression and ranking only: classification returned above. Both assert a NUMBER, so naming
    # the target here is not the defect it was for a class label -- the sentence and its interval
    # are about the same quantity, and there is no token standing in for a meaning.
    value = _fmt(point_forecast)
    position = f"ranked {rank}" if rank is not None else "ranked"

    if phrasing == "canonical":
        interval_clause = (
            f" The {level_pct}% prediction interval for the {target} of {subject} "
            f"is {_fmt(lo)} to {_fmt(hi)}."
        )
        if spec.target_type == "ranking":
            head = (
                f"By {target}, {subject} is {position} among the entities in this task, "
                f"with a score of {_fmt(point_forecast)}."
            )
        else:
            head = f"The {target} of {subject} is {value}."
    elif phrasing == "subject_first":
        interval_clause = (
            f" Its {level_pct}% prediction interval is {_fmt(lo)} to {_fmt(hi)}."
        )
        if spec.target_type == "ranking":
            head = (
                f"Among the entities in this task, {subject} ranks {rank} by {target}, "
                f"with a score of {_fmt(point_forecast)}."
            )
        else:
            head = f"For {subject}, {target} is {value}."
    else:  # value_first
        interval_clause = (
            f" The {level_pct}% interval around it is {_fmt(lo)} to {_fmt(hi)}."
        )
        if spec.target_type == "ranking":
            head = (
                f"A score of {_fmt(point_forecast)} places {subject} {position} "
                f"by {target} among the entities in this task."
            )
        else:
            head = f"{value} is the {target} of {subject}."
    return head + interval_clause


def _classification_hypothesis(
    spec: HypothesisSpec, *, phrasing: str, entity_id: str, label: str
) -> str:
    """ "<entity> <predicate>." -- the label's MEANING, asserted of the entity. No target name.

    The three phrasings vary only how the entity is named and how the assertion is framed. They
    cannot vary word order the way the numeric templates do, because the predicate is one authored
    clause rather than slots this module assembles; what they can do, and all that is wanted, is
    stop one rendering of the subject from deciding admissibility. "It is the case that" is a bare
    assertoric frame: it adds no modality, no hedge and no content, so the three sentences remain
    exactly as true or false as one another.
    """
    predicate = spec.assertion_for(label)
    if phrasing == "canonical":
        return f"{spec.name_for(entity_id)} {predicate}."
    if phrasing == "subject_first":
        return f"{spec.plain_name_for(entity_id)} {predicate}."
    return f"It is the case that {spec.name_for(entity_id)} {predicate}."


def prediction_claims(
    aligned: AlignedPredictions, spec: HypothesisSpec
) -> list[dict[str, Any]]:
    """One normalized claim per roster entity: hypothesis = the prediction, citations = its own.

    The returned list is in trusted roster order and has exactly one element per roster entity, so
    the faithfulness denominator is the roster and not the number of sentences the participant
    chose to write. Padding a submission with extra prose can no longer move the score.
    """
    claims: list[dict[str, Any]] = []
    for index, entity_id in enumerate(aligned.entity_ids):
        rank = aligned.ranks[index]
        claims.append(
            {
                "text": canonical_hypothesis(
                    spec,
                    entity_id=entity_id,
                    label=aligned.pred_labels[index],
                    point_forecast=aligned.pred_values[index],
                    rank=rank,
                    lo=aligned.lo[index],
                    hi=aligned.hi[index],
                ),
                #: The same assertion in every phrasing, for the median aggregation in `g3`.
                #: `text` stays the canonical one so any consumer reading a single string is
                #: unchanged.
                "texts": [
                    canonical_hypothesis(
                        spec,
                        phrasing=phrasing,
                        entity_id=entity_id,
                        label=aligned.pred_labels[index],
                        point_forecast=aligned.pred_values[index],
                        rank=rank,
                        lo=aligned.lo[index],
                        hi=aligned.hi[index],
                    )
                    for phrasing in PHRASINGS
                ],
                "citations": list(aligned.citations_by_entity[index]),
            }
        )
    return claims
