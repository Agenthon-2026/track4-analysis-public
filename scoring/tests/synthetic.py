"""Synthetic Track-4 units for the public test suite. No real corpus, no real outcome, ever.

Every fixture here is generated from constants in this file: made-up tickers (``SYN-A``…),
made-up filings, made-up numbers. Nothing is copied from a private unit, and the public/private
firewall is the reason — a public test that needs a real corpus document is a public test that
publishes one.

The builder writes a complete unit tree (``card.toml``, ``task.json``, ``manifest.json`` with real
sha256 digests, ``corpus/*.json``, optionally ``reference/outcome.json``) so the tests exercise the
real trusted-manifest path rather than a hand-built dict that skips it.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

__all__ = [
    "BEAT_MARKER",
    "CUTOFF",
    "MISS_MARKER",
    "POST_CUTOFF_DOC",
    "PRE_CUTOFF_DOC",
    "SUPPORTING_TEXT",
    "SYNTH_LABEL_ASSERTIONS",
    "StubJudge",
    "answer_for",
    "build_unit",
    "outcome_for",
    "write_naive_answer",
]

CUTOFF = "2026-02-15"

PRE_CUTOFF_DOC = "SYNTHDOC_PRE_20260201"
POST_CUTOFF_DOC = "SYNTHDOC_POST_20260301"
TRIVIA_DOC = "SYNTHDOC_TRIVIA_20260110"

#: The distinguishing phrase of each label's ASSERTION, for the stub judges that key on the
#: hypothesis. These track `target.label_assertions` above: a classification hypothesis asserts
#: what the label MEANS and no longer contains the label token, so a stub keying on "is beat"
#: would silently stop matching and the positive controls would pass for the wrong reason.
BEAT_MARKER = "above the consensus estimate"
MISS_MARKER = "below the consensus estimate"

#: The synthetic unit's per-label assertions. One definition, used both by `build_unit` (which
#: writes them into task.json) and by tests that construct a `HypothesisSpec` directly, so the
#: two cannot drift into asking the judge different questions about the same fixture.
SYNTH_LABEL_ASSERTIONS = {
    "beat": f"will post quarterly EPS {BEAT_MARKER}",
    "miss": f"will post quarterly EPS {MISS_MARKER}",
    "inline": "will post quarterly EPS in line with the consensus estimate",
}

#: The premise a judge is told to entail in the "supported prediction" tests.
SUPPORTING_TEXT = (
    "Synthetic Issuer A reported quarterly earnings above the published consensus."
)
TRIVIA_TEXT = (
    "The synthetic exchange observes a public holiday on the first Monday of February."
)
POST_TEXT = "Synthetic Issuer A published full-year results after the question cutoff."

_ENTITIES = ("SYN-A", "SYN-B", "SYN-C")


def _doc(doc_id: str, doc_date: str, text: str) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "doc_date": doc_date,
        "source": "synthetic",
        "spans": [{"span_id": 0, "start": 0, "end": len(text), "text": text}],
        "text": text,
    }


_DOCS: dict[str, dict[str, Any]] = {
    PRE_CUTOFF_DOC: _doc(PRE_CUTOFF_DOC, "2026-02-01", SUPPORTING_TEXT),
    TRIVIA_DOC: _doc(TRIVIA_DOC, "2026-01-10", TRIVIA_TEXT),
    POST_CUTOFF_DOC: _doc(POST_CUTOFF_DOC, "2026-03-01", POST_TEXT),
}

_CARD = """\
schema_version = "2.0"

[task]
id              = "t4-SYNTH"
track           = "analysis"
title           = "Synthetic Track-4 unit"
split           = "public-dev"
family          = "synthetic"
target_type     = "{target_type}"
prompt          = "synthetic"
cutoff_date     = "{cutoff}"
resolution_date = "2026-05-01"
adversarial     = false

[metadata]
author_name              = "synthetic"
author_email             = "synthetic@example.invalid"
difficulty               = "medium"
category                 = "synthetic"
tags                     = ["analysis", "synthetic"]
expert_time_estimate_min = 1.0
junior_time_estimate_min = 1.0

[provenance]
license             = "CC-BY-4.0"
data_source         = "synthetic"
data_cutoff         = "{cutoff}"
public_release_date = "2026-01-01"
redistributable     = true
manifest            = "manifest.json"

[contamination]
canary_guid = "00000000-0000-4000-8000-000000000000"

[scoring]
verifier            = "t4.faithful_analysis"
metric              = "analysis_composite"
admissibility_gates = ["g0_integrity", "g1_schema", "g2_cutoff_resource", "g3_domain_semantics"]

[scoring.params]
faithfulness_threshold = {faithfulness_threshold}
interval_level         = {interval_level}
target_type            = "{target_type}"
composite_weights      = [0.7, 0.3]
tau_citation           = 0.5

[environment]
cpus    = 4
memory  = "16G"
gpu     = false
network = "restricted"

[corpus]
pii_stripped      = true
manifest_required = true
manifest_path     = "manifest.json"

[embargo]
cutoff_field   = "cutoff_date"
doc_date_field = "doc_date"
strict         = true
"""


def build_unit(
    root: pathlib.Path,
    *,
    target_type: str = "classification",
    cutoff: str = CUTOFF,
    entities: tuple[str, ...] = _ENTITIES,
    with_outcome: bool = False,
    interval_level: float = 0.90,
    faithfulness_threshold: float = 0.80,
    docs: dict[str, dict[str, Any]] | None = None,
    labels: dict[str, dict[str, Any]] | None = None,
    unlabelled: bool = False,
    with_naive: bool | None = None,
) -> pathlib.Path:
    """Write a complete synthetic unit under `root` and return its directory.

    `with_naive` (default: same as `with_outcome`) also writes the declared naive rule
    `reference/naive_answer.json` via `write_naive_answer`. Every scored unit with numeric truth
    needs one from 5.1.0 on (the interval leg is measured against the naive interval), and its
    band equals `answer_for`'s default, so a default answer scores interval quality 0.5.

    Entity labels: by default every document is `shared` (any entity may cite it), which is the
    shared-corpus anatomy the document-level entity check admits unconditionally. Pass
    `labels={doc_id: {"entity_ids": [...]}}` to bind a document to entities (an absent doc_id
    keeps `shared`), or `unlabelled=True` to write no label at all -- the organizer-fault case.
    """
    unit = root / "t4-SYNTH"
    (unit / "corpus").mkdir(parents=True, exist_ok=True)
    payloads = dict(docs or _DOCS)
    labels = dict(labels or {})

    files: list[dict[str, Any]] = []
    for doc_id, document in payloads.items():
        blob = (json.dumps(document, indent=1) + "\n").encode("utf-8")
        (unit / "corpus" / f"{doc_id}.json").write_bytes(blob)
        entry: dict[str, Any] = {
            "path": f"corpus/{doc_id}.json",
            "role": "corpus",
            "source": "synthetic",
            "license": "CC-BY-4.0",
            "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
            "split": "public-dev",
            "cutoff": cutoff,
            "redistributable": True,
            "pii_stripped": True,
        }
        if not unlabelled:
            entry.update(labels.get(doc_id, {"shared": True}))
        files.append(entry)
    (unit / "manifest.json").write_text(
        json.dumps(
            {"manifest_version": "2.0", "unit_id": "t4-SYNTH", "files": files}, indent=1
        )
        + "\n",
        encoding="utf-8",
    )

    (unit / "card.toml").write_text(
        _CARD.format(
            target_type=target_type,
            cutoff=cutoff,
            interval_level=interval_level,
            faithfulness_threshold=faithfulness_threshold,
        ),
        encoding="utf-8",
    )

    task: dict[str, Any] = {
        "task_id": "t4-SYNTH",
        "schema_version": "3",
        "family": "synthetic",
        # A vocabulary belongs to a classification unit and to no other kind. This builder used to
        # emit one on all three types, which is a shape no real unit has: every classification unit
        # declares `target.labels` and no regression or ranking unit does.
        "target": {
            "name": "eps_outcome",
            "type": target_type,
            **(
                {
                    "labels": ["beat", "miss", "inline"],
                    # Required for classification: the hypothesis asserts the label's MEANING and
                    # never names the target. A unit without these refuses to score rather than
                    # falling back.
                    "label_assertions": dict(SYNTH_LABEL_ASSERTIONS),
                }
                if target_type == "classification"
                else {}
            ),
        },
        "prompt": "synthetic",
        "cutoff_date": cutoff,
        "resolution_date": "2026-05-01",
        "interval_level": interval_level,
        "entities": [
            {"entity_id": eid, "name": f"Synthetic Issuer {eid[-1]}"}
            for eid in entities
        ],
    }
    if cutoff is None:
        task.pop("cutoff_date")
    (unit / "task.json").write_text(json.dumps(task, indent=1) + "\n", encoding="utf-8")

    if with_outcome:
        (unit / "reference").mkdir(exist_ok=True)
        (unit / "reference" / "outcome.json").write_text(
            json.dumps(outcome_for(entities), indent=1) + "\n", encoding="utf-8"
        )
    if with_outcome if with_naive is None else with_naive:
        write_naive_answer(unit, entities, target_type=target_type)
    return unit


def outcome_for(entities: tuple[str, ...] = _ENTITIES) -> dict[str, Any]:
    """A synthetic resolved outcome: first entity beats, the rest miss."""
    return {
        "unit_id": "t4-SYNTH",
        "cutoff_date": CUTOFF,
        "target_type": "classification",
        "outcomes": [
            {
                "entity_id": eid,
                "true_label": "beat" if index == 0 else "miss",
                "y": 1.0 + index,
            }
            for index, eid in enumerate(entities)
        ],
    }


def answer_for(
    entities: tuple[str, ...] = _ENTITIES,
    *,
    doc_id: str = PRE_CUTOFF_DOC,
    label: str = "beat",
    interval_level: float = 0.90,
    lo: float = 0.5,
    hi: float = 3.5,
    point_forecast: float = 1.0,
    claim_text: str = "Synthetic Issuer A beat consensus.",
) -> dict[str, Any]:
    text_len = len(_DOCS[doc_id]["text"]) if doc_id in _DOCS else 64
    return {
        "task_id": "t4-SYNTH",
        "schema_version": "3",
        "target_type": "classification",
        "entity_predictions": [
            {
                "entity_id": eid,
                "label": label,
                "point_forecast": point_forecast,
                "interval": {"level": interval_level, "lo": lo, "hi": hi},
                "claims": [
                    {
                        "doc_id": doc_id,
                        "span_start": 0,
                        "span_end": text_len,
                        "claim": claim_text,
                    }
                ],
            }
            for eid in entities
        ],
    }


def write_naive_answer(
    unit: pathlib.Path,
    entities: tuple[str, ...] = _ENTITIES,
    *,
    point_forecast: float = 2.0,
    target_type: str = "regression",
) -> pathlib.Path:
    """Write a synthetic declared naive rule to ``reference/naive_answer.json``.

    Every scored unit with numeric truth carries one: its point forecasts anchor the regression
    soft ratio and its intervals anchor the interval leg. A flat forecast and `answer_for`'s
    default interval for every entity, in the same analysis-schema shape as a participant answer.
    """
    naive = answer_for(entities, point_forecast=point_forecast)
    naive["target_type"] = target_type
    naive["notes"] = {"baseline_id": "synthetic-flat"}
    (unit / "reference").mkdir(exist_ok=True)
    path = unit / "reference" / "naive_answer.json"
    path.write_text(json.dumps(naive, indent=1) + "\n", encoding="utf-8")
    return path


class StubJudge:
    """Entails only the premises it is told to. Records every (premise, hypothesis) pair.

    Used to prove *what question the judge was asked*, which is the whole of the
    prediction-bound-evidence fix — a judge that is consulted with the participant's own prose is
    a judge answering the wrong question, and only the recorded call reveals that.
    """

    def __init__(
        self,
        entailed_premises: tuple[str, ...] = (),
        score: float = 0.99,
        *,
        contradicted_premises: tuple[str, ...] = (),
        contradiction_score: float = 0.99,
    ) -> None:
        self.entailed_premises = set(entailed_premises)
        self.score = score
        self.contradicted_premises = set(contradicted_premises)
        self.contradiction_score = contradiction_score
        #: Every question asked, entail and contradiction alike, in order.
        self.calls: list[tuple[str, str]] = []
        #: Only the 5.2.0 penalty's questions (three-way P(contradiction)).
        self.contradiction_calls: list[tuple[str, str]] = []

    def entail(self, premise: str, hypothesis: str) -> float:
        self.calls.append((premise, hypothesis))
        return self.score if premise in self.entailed_premises else 0.01

    def contradiction(self, premise: str, hypothesis: str) -> float:
        self.calls.append((premise, hypothesis))
        self.contradiction_calls.append((premise, hypothesis))
        return (
            self.contradiction_score if premise in self.contradicted_premises else 0.01
        )


class HypothesisAwareJudge:
    """Entails a premise only when the hypothesis mentions the substring it was told to expect.

    This is how "supported trivia paired with an unsupported prediction" is expressed without a
    real model: the trivia premise is genuinely in the corpus and genuinely described accurately,
    and the judge still refuses because the *prediction* is not what the premise supports.
    """

    def __init__(
        self,
        premise: str,
        required_in_hypothesis: str,
        score: float = 0.99,
        *,
        contradicting_in_hypothesis: str | None = None,
    ) -> None:
        self.premise = premise
        self.required = required_in_hypothesis
        self.score = score
        #: 5.2.0: the hypothesis substring for which this premise CONTRADICTS the claim.
        self.contradicting = contradicting_in_hypothesis
        self.calls: list[tuple[str, str]] = []
        self.contradiction_calls: list[tuple[str, str]] = []

    def entail(self, premise: str, hypothesis: str) -> float:
        self.calls.append((premise, hypothesis))
        if premise == self.premise and self.required in hypothesis:
            return self.score
        return 0.01

    def contradiction(self, premise: str, hypothesis: str) -> float:
        self.calls.append((premise, hypothesis))
        self.contradiction_calls.append((premise, hypothesis))
        if (
            self.contradicting is not None
            and premise == self.premise
            and self.contradicting in hypothesis
        ):
            return self.score
        return 0.01
