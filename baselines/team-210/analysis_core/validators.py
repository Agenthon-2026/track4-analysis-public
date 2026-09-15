"""Refuse to write an answer the evidence does not support.

Track 4 states the rule outright: if the citations do not hold up under the NLI
check, the submission is ineligible *regardless of how accurate its predictions
were*. So accuracy is not the thing to protect here - admissibility is, and the
cheapest way to protect it is to check the answer against the same questions the
gate will ask, before writing it.

Two classes of check, and they fail differently:

  citations   a span that does not resolve, or a document past the embargo.
              Fixable: drop the claim and re-cite.
  schema      a missing interval bound, an interval level that disagrees with
              the card, a target_type that contradicts the card. Not fixable
              after the fact - g1 rejects the WHOLE submission at W = -0.27, so
              one malformed entity costs every other entity too.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .corpus import Document

#: How much of the claim text must actually appear in the cited span. Below
#: this the citation is decorative, and an NLI judge will say so.
MIN_SPAN_OVERLAP = 0.6


@dataclass(frozen=True)
class Problem:
    where: str
    what: str

    def __str__(self) -> str:
        return f"{self.where}: {self.what}"


def _overlap(claim: str, span_text: str) -> float:
    claim_words = {w for w in claim.lower().split() if len(w) > 3}
    if not claim_words:
        return 0.0
    span_words = {w for w in span_text.lower().split() if len(w) > 3}
    return len(claim_words & span_words) / len(claim_words)


def check_citations(prediction: dict, by_id: dict[str, Document],
                    cutoff: dt.date | None) -> list[Problem]:
    entity = prediction.get("entity_id", "?")
    problems: list[Problem] = []

    for index, claim in enumerate(prediction.get("claims", [])):
        where = f"{entity}.claims[{index}]"
        doc_id = claim.get("doc_id")
        document = by_id.get(doc_id)
        if document is None:
            problems.append(Problem(where, f"cites unknown document {doc_id!r}"))
            continue
        if cutoff and document.doc_date and document.doc_date > cutoff:
            problems.append(Problem(
                where, f"{doc_id} is dated {document.doc_date}, after the "
                       f"{cutoff} embargo"))
            continue

        start, end = claim.get("span_start"), claim.get("span_end")
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            problems.append(Problem(where, f"span [{start}, {end}) is not a range"))
            continue
        if end > len(document.text):
            problems.append(Problem(
                where, f"span ends at {end} but {doc_id} has "
                       f"{len(document.text)} characters"))
            continue

        text = document.span(start, end)
        ratio = _overlap(str(claim.get("claim", "")), text)
        if ratio < MIN_SPAN_OVERLAP:
            problems.append(Problem(
                where, f"claim shares only {ratio:.0%} of its terms with the "
                       "cited span; the citation does not carry the claim"))
    return problems


def check_schema(answer: dict, interval_level: float,
                 roster: list[str]) -> list[Problem]:
    """The g1 questions. One failure here voids the whole submission."""
    problems: list[Problem] = []

    for field in ("task_id", "entity_predictions"):
        if field not in answer:
            problems.append(Problem("answer", f"missing required field {field!r}"))

    predictions = answer.get("entity_predictions", [])
    seen = []
    for prediction in predictions:
        entity = prediction.get("entity_id")
        where = f"entity {entity!r}"
        seen.append(entity)

        interval = prediction.get("interval")
        if not isinstance(interval, dict):
            problems.append(Problem(where, "interval is required for every task type"))
            continue
        for bound in ("lo", "hi"):
            if not isinstance(interval.get(bound), (int, float)):
                problems.append(Problem(
                    where, f"interval.{bound} missing - this is not a coverage "
                           "penalty, it fails g1 and voids the submission"))
        level = interval.get("level")
        if level is None or abs(float(level) - interval_level) > 1e-9:
            problems.append(Problem(
                where, f"interval.level is {level}, must equal the card's "
                       f"{interval_level}"))
        if (isinstance(interval.get("lo"), (int, float))
                and isinstance(interval.get("hi"), (int, float))
                and interval["lo"] > interval["hi"]):
            problems.append(Problem(where, "interval.lo exceeds interval.hi"))

    missing = [e for e in roster if e not in seen]
    if missing:
        problems.append(Problem("answer", f"roster entities absent: {missing}"))
    return problems
