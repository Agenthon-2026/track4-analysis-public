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
    "CUTOFF",
    "POST_CUTOFF_DOC",
    "PRE_CUTOFF_DOC",
    "SUPPORTING_TEXT",
    "StubJudge",
    "answer_for",
    "build_unit",
    "outcome_for",
]

CUTOFF = "2026-02-15"

PRE_CUTOFF_DOC = "SYNTHDOC_PRE_20260201"
POST_CUTOFF_DOC = "SYNTHDOC_POST_20260301"
TRIVIA_DOC = "SYNTHDOC_TRIVIA_20260110"

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
) -> pathlib.Path:
    """Write a complete synthetic unit under `root` and return its directory."""
    unit = root / "t4-SYNTH"
    (unit / "corpus").mkdir(parents=True, exist_ok=True)
    payloads = dict(docs or _DOCS)

    files: list[dict[str, Any]] = []
    for doc_id, document in payloads.items():
        blob = (json.dumps(document, indent=1) + "\n").encode("utf-8")
        (unit / "corpus" / f"{doc_id}.json").write_bytes(blob)
        files.append(
            {
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
        )
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
        # emit one on all three types, which is a shape no unit in either tree has: measured across
        # `units/` and `units_archived/` in this repository and the private one, 34 classification
        # units all declare `target.labels` and 0 of the 56 regression and ranking units do.
        "target": {
            "name": "eps_outcome",
            "type": target_type,
            **(
                {"labels": ["beat", "miss", "inline"]}
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


class StubJudge:
    """Entails only the premises it is told to. Records every (premise, hypothesis) pair.

    Used to prove *what question the judge was asked*, which is the whole of the
    prediction-bound-evidence fix — a judge that is consulted with the participant's own prose is
    a judge answering the wrong question, and only the recorded call reveals that.
    """

    def __init__(
        self, entailed_premises: tuple[str, ...] = (), score: float = 0.99
    ) -> None:
        self.entailed_premises = set(entailed_premises)
        self.score = score
        self.calls: list[tuple[str, str]] = []

    def entail(self, premise: str, hypothesis: str) -> float:
        self.calls.append((premise, hypothesis))
        return self.score if premise in self.entailed_premises else 0.01


class HypothesisAwareJudge:
    """Entails a premise only when the hypothesis mentions the substring it was told to expect.

    This is how "supported trivia paired with an unsupported prediction" is expressed without a
    real model: the trivia premise is genuinely in the corpus and genuinely described accurately,
    and the judge still refuses because the *prediction* is not what the premise supports.
    """

    def __init__(
        self, premise: str, required_in_hypothesis: str, score: float = 0.99
    ) -> None:
        self.premise = premise
        self.required = required_in_hypothesis
        self.score = score
        self.calls: list[tuple[str, str]] = []

    def entail(self, premise: str, hypothesis: str) -> float:
        self.calls.append((premise, hypothesis))
        if premise == self.premise and self.required in hypothesis:
            return self.score
        return 0.01
