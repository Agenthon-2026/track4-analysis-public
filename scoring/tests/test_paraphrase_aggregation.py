"""Admissibility must not swing on which wording the organizers happened to pick.

An NLI cross-encoder's entailment probability moves with wording even when the content asserted
is IDENTICAL, so a claim sitting near `tau_citation` can cross it on the rewording alone. The
judge already ensembles over two MODELS to damp model-specific noise; phrasing is a second,
independent source of the same kind of noise, and it was damped not at all.

These tests pin the three properties that make the aggregation trustworthy rather than merely
more permissive: every phrasing asserts the same content, the aggregate is a MEDIAN (so it is
not mechanically >= a single phrasing), and one outlying phrasing cannot decide admission. The
last two pin the schema invariant that keeps this module's narrow citation parser in agreement
with the shared primitive it no longer calls.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

import pytest

from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.hypothesis import (
    PHRASINGS,
    HypothesisSpec,
    canonical_hypothesis,
)
from qfbench2_track_analysis.scoring import _paraphrase_faithfulness

SPEC = HypothesisSpec(
    target_type="regression",
    interval_level=0.90,
    target_name="yield change bps",
    entity_names={"E1": "UST2Y"},
)


def _texts() -> list[str]:
    return [
        canonical_hypothesis(
            SPEC,
            phrasing=p,
            entity_id="E1",
            label="",
            point_forecast=1.5,
            rank=None,
            lo=-1.0,
            hi=4.0,
        )
        for p in PHRASINGS
    ]


def test_there_are_at_least_three_phrasings() -> None:
    """A median over two values is their mean: it rejects no outlier."""
    assert len(PHRASINGS) >= 3


def test_every_phrasing_asserts_the_same_numbers() -> None:
    """Same content, different word order. A template that added information would inflate
    entailment for a reason unrelated to the submission."""
    for text in _texts():
        assert "1.5" in text and "-1" in text and "4" in text
        assert "UST2Y" in text and "yield change bps" in text
        assert "90%" in text


def test_the_phrasings_are_actually_different() -> None:
    assert len(set(_texts())) == len(PHRASINGS)


def test_an_unknown_phrasing_is_an_organizer_fault() -> None:
    with pytest.raises(T4OrganizerFault):
        canonical_hypothesis(
            SPEC,
            phrasing="not-a-template",
            entity_id="E1",
            label="",
            point_forecast=1.0,
            rank=None,
            lo=0.0,
            hi=2.0,
        )


class _ScriptedJudge:
    """Returns a per-(premise, hypothesis) score from a table; anything unlisted scores 0."""

    def __init__(self, table: dict[str, float]) -> None:
        self.table = table

    def entail(self, premise: str, hypothesis: str) -> float:
        return self.table.get(hypothesis, 0.0)


def _claim(texts: list[str]) -> dict[str, Any]:
    return {
        "text": texts[0],
        "texts": texts,
        "citations": [{"doc_id": "DOC", "span_start": 0, "span_end": 4}],
    }


def _lookup(doc_id: str) -> Mapping[str, Any]:
    """The `doc_id -> document` callable `CorpusIndex.lookup()` returns."""
    return {"text": "abcd"}


def test_one_outlying_phrasing_cannot_admit_a_claim() -> None:
    """THE GUARD. Two phrasings below tau, one far above: the median refuses.

    Under `max` this claim would be admitted on the strength of a single wording, which is the
    inflation this aggregation exists to avoid.
    """
    texts = _texts()
    judge = _ScriptedJudge({texts[0]: 0.20, texts[1]: 0.30, texts[2]: 0.99})
    assert _paraphrase_faithfulness([_claim(texts)], _lookup, judge, tau=0.5) == 0.0


def test_one_outlying_phrasing_cannot_refuse_a_claim() -> None:
    """The other direction, which is the actual complaint: two above, one far below -> admitted."""
    texts = _texts()
    judge = _ScriptedJudge({texts[0]: 0.85, texts[1]: 0.80, texts[2]: 0.05})
    assert _paraphrase_faithfulness([_claim(texts)], _lookup, judge, tau=0.5) == 1.0


def test_a_claim_supported_in_every_phrasing_is_supported() -> None:
    """POSITIVE CONTROL."""
    texts = _texts()
    judge = _ScriptedJudge(dict.fromkeys(texts, 0.9))
    assert _paraphrase_faithfulness([_claim(texts)], _lookup, judge, tau=0.5) == 1.0


def test_a_claim_unsupported_in_every_phrasing_is_not() -> None:
    """NEGATIVE CONTROL. Without this, an aggregator that always returned 1.0 would pass."""
    texts = _texts()
    judge = _ScriptedJudge(dict.fromkeys(texts, 0.1))
    assert _paraphrase_faithfulness([_claim(texts)], _lookup, judge, tau=0.5) == 0.0


def test_a_median_exactly_at_tau_is_not_supported() -> None:
    """Support means the median EXCEEDS tau: a median equal to tau does not count."""
    texts = _texts()
    at_tau = _ScriptedJudge(dict.fromkeys(texts, 0.5))
    assert _paraphrase_faithfulness([_claim(texts)], _lookup, at_tau, tau=0.5) == 0.0
    just_above = _ScriptedJudge(dict.fromkeys(texts, 0.5000001))
    assert (
        _paraphrase_faithfulness([_claim(texts)], _lookup, just_above, tau=0.5) == 1.0
    )


def test_no_claims_is_zero_not_one() -> None:
    """An empty claim list must not be vacuously perfect."""
    assert _paraphrase_faithfulness([], _lookup, _ScriptedJudge({}), tau=0.5) == 0.0


# --------------------------------------------------------------------------- #
# The invariants the narrow citation parser depends on                          #
# --------------------------------------------------------------------------- #
# `_paraphrase_faithfulness` reads `span_start`/`span_end` and `document["text"]` directly,
# because the median has to be taken per claim and the shared primitive only aggregates to a unit
# fraction. That parser is narrower than `qfbench2_common.scoring.faithfulness`, which also accepts
# a `span: [s, e]` citation. The two never disagree only because `g1_schema` refuses the other
# shape BEFORE this gate runs. Nothing else pins that, so if the schema's `required` list were
# relaxed, such a citation would reach the aggregation, support nothing, and change scores with no
# test firing. This is that test.


def test_the_published_schema_still_requires_span_start_and_span_end() -> None:
    import json

    import jsonschema
    from qfbench2_common.taskcard import schema_path

    schema = json.loads(pathlib.Path(schema_path("analysis.schema.json")).read_text())
    validator = jsonschema.Draft202012Validator(schema)

    def answer(citation: dict[str, Any]) -> dict[str, Any]:
        return {
            "entity_predictions": [
                {
                    "entity_id": "E1",
                    "point_forecast": 1.0,
                    "interval": {"level": 0.90, "lo": 0.0, "hi": 2.0},
                    "claims": [citation],
                }
            ]
        }

    offsets = {"doc_id": "D", "span_start": 0, "span_end": 4, "claim": "c"}
    camp_a = {"doc_id": "D", "span": [0, 4], "claim": "c"}

    def span_errors(citation: dict[str, Any]) -> list[str]:
        return [
            e.message
            for e in validator.iter_errors(answer(citation))
            if "span" in e.message
        ]

    # The documented Track 4 shape draws no span complaint...
    assert span_errors(offsets) == []
    # ...and the shape the shared primitive would also have accepted is refused here, by name.
    # A set: `iter_errors` makes no ordering guarantee across jsonschema versions.
    assert set(span_errors(camp_a)) == {
        "'span_start' is a required property",
        "'span_end' is a required property",
    }


def test_a_citation_the_schema_refuses_would_indeed_support_nothing_here() -> None:
    """Why the test above matters: the narrow parser silently scores such a claim 0."""
    texts = _texts()
    judge = _ScriptedJudge(dict.fromkeys(texts, 0.99))
    claim = {
        "text": texts[0],
        "texts": texts,
        "citations": [{"doc_id": "DOC", "span": [0, 4]}],
    }
    assert _paraphrase_faithfulness([claim], _lookup, judge, tau=0.5) == 0.0
