"""A classification hypothesis asserts what the label MEANS, and never names the target.

The defect these pin. The hypothesis used to read "The <target name> of <entity> is <label>.",
so wherever the target name shares vocabulary with its own labels -- the ordinary shape of a
classification target, not a corner case -- the sentence put to the judge was dominated by a
topic word:

    "The credit event 12m of Northwind Air Holdings (NWA) is no_event."

put against a passage reporting the issuer's liquidity. The judge declines to call one an
entailment of the other, which is the right answer to the question it was asked and the wrong
answer to the question we meant. Worse, the score then tracks whether the label token echoes the
passage's vocabulary rather than whether the evidence says anything: predictions separate by
which label was chosen rather than by what was cited, and that separation disappears as soon as
the same predictions are asserted in plain English. Hence `target.label_assertions`, and hence
these tests.

What these tests do NOT pin, deliberately: entity-nonspecificity. An NLI model still reads a
claim about one company against another company's filing as entailed when the wording matches,
and nothing here touches that -- it is at full strength on regression and ranking, which have no
label vocabulary. That is the document-level entity check's job
(`test_entity_bound_citations.py`).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.hypothesis import (
    PHRASINGS,
    HypothesisSpec,
    canonical_hypothesis,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

SPEC = HypothesisSpec(
    target_type="classification",
    interval_level=0.90,
    target_name="credit event 12m",
    entity_names={"NWA": "Northwind Air Holdings"},
    label_assertions={
        "no_event": "will not suffer a default-class event within the year after 2031-03-31",
        "credit_event": "will suffer a default-class event within the year after 2031-03-31",
    },
)


def _hypothesis(phrasing: str = "canonical", label: str = "no_event") -> str:
    return canonical_hypothesis(
        SPEC,
        phrasing=phrasing,
        entity_id="NWA",
        label=label,
        point_forecast=0.06,
        rank=None,
        lo=0.01,
        hi=0.18,
    )


def test_the_hypothesis_is_the_label_assertion() -> None:
    assert _hypothesis() == (
        "Northwind Air Holdings (NWA) will not suffer a default-class event "
        "within the year after 2031-03-31."
    )


def test_the_target_name_is_gone() -> None:
    """THE DEFECT. The target name is what the judge was matching on topically."""
    for phrasing in PHRASINGS:
        assert "credit event 12m" not in _hypothesis(phrasing)


def test_the_label_token_is_gone() -> None:
    """`no_event` is a machine token no filing sentence ever resembles."""
    for phrasing in PHRASINGS:
        assert "no_event" not in _hypothesis(phrasing)


def test_no_interval_clause_on_a_category() -> None:
    """An interval on a class label is not something a corpus passage can support.

    It is still scored -- it is the coverage term of the composite -- but it is not asked about.
    """
    for phrasing in PHRASINGS:
        text = _hypothesis(phrasing)
        assert "0.01" not in text and "0.18" not in text
        assert "interval" not in text.lower()


def test_regression_keeps_its_interval_clause() -> None:
    """The contrast: there the sentence and its interval are about the same quantity."""
    spec = HypothesisSpec(
        target_type="regression", interval_level=0.90, target_name="eps"
    )
    text = canonical_hypothesis(
        spec, entity_id="X", label="", point_forecast=1.25, rank=None, lo=1.0, hi=1.5
    )
    assert "1.25" in text and "interval" in text.lower()


def test_the_opposite_prediction_asks_the_opposite_question() -> None:
    assert "will not suffer" in _hypothesis(label="no_event")
    assert "will not suffer" not in _hypothesis(label="credit_event")


def test_every_phrasing_is_distinct_and_asserts_the_same_thing() -> None:
    texts = [_hypothesis(p) for p in PHRASINGS]
    assert len(set(texts)) == len(PHRASINGS)
    for text in texts:
        assert (
            "will not suffer a default-class event within the year after 2031-03-31"
            in text
        )
        assert "Northwind Air Holdings" in text


def test_a_missing_assertion_refuses_rather_than_falling_back() -> None:
    """Fail closed. A silent fallback would reinstate the defect on any unit nobody authored."""
    spec = HypothesisSpec(
        target_type="classification",
        interval_level=0.90,
        target_name="credit event 12m",
    )
    with pytest.raises(T4OrganizerFault, match="label_assertions"):
        canonical_hypothesis(
            spec,
            entity_id="NWA",
            label="no_event",
            point_forecast=0.06,
            rank=None,
            lo=0.01,
            hi=0.18,
        )


def test_an_empty_assertion_is_refused_at_parse_time() -> None:
    """An empty predicate would reduce the hypothesis to the bare entity name."""
    task = {
        "target": {
            "name": "t",
            "type": "classification",
            "labels": ["a"],
            "label_assertions": {"a": "   "},
        },
        "entities": [],
    }
    with pytest.raises(T4OrganizerFault, match="empty"):
        HypothesisSpec.from_task(
            task, target_type="classification", interval_level=0.90
        )


# --------------------------------------------------------------------------------------------
# The shipped units
# --------------------------------------------------------------------------------------------
def _classification_units() -> list[pathlib.Path]:
    return sorted(
        p
        for p in (REPO_ROOT / "units").glob("*/task.json")
        if json.loads(p.read_text()).get("target", {}).get("type") == "classification"
    )


def test_there_are_classification_units_to_check() -> None:
    """Without this, the parametrised test below would pass vacuously if the glob broke."""
    assert _classification_units()


@pytest.mark.parametrize(
    "task_path", _classification_units(), ids=lambda p: p.parent.name
)
def test_every_shipped_classification_unit_declares_every_assertion(
    task_path: pathlib.Path,
) -> None:
    """Caught here at authoring time rather than as an organizer fault mid-run."""
    target = json.loads(task_path.read_text())["target"]
    declared = target.get("label_assertions") or {}
    missing = [label for label in target["labels"] if not declared.get(label)]
    assert not missing, f"{task_path.parent.name} has no assertion for {missing}"
    assert not [
        label for label in declared if label not in target["labels"]
    ], "an assertion for a label outside the closed vocabulary"


@pytest.mark.parametrize(
    "task_path", _classification_units(), ids=lambda p: p.parent.name
)
def test_no_assertion_merely_restates_the_target_name(task_path: pathlib.Path) -> None:
    """The whole point: the assertion must say what the label MEANS, not name the target again.

    Only the MACHINE spelling of a label is banned -- an underscored token like `no_event` or
    `going_concern_added`, which is the thing no filing sentence ever resembles. A single-word
    label such as `up` or `beat` is ordinary English, and "will be revised up" is the assertion
    doing its job, not the token leaking through.
    """
    target = json.loads(task_path.read_text())["target"]
    name = target["name"].replace("_", " ").lower()
    for label, assertion in (target.get("label_assertions") or {}).items():
        assert name not in assertion.lower(), f"{task_path.parent.name}/{label}"
        if "_" in label:
            assert (
                label.lower() not in assertion.lower()
            ), f"{task_path.parent.name}/{label}"
