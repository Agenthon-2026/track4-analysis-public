"""Scorer 5.2.0: the every-figure rule, the task-table citation, and the date-shape grammar fixes.

Synthetic inputs only. Each case names what a regression would look like."""

from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal

import pytest

from qfbench2_track_analysis import numeric as NUM
from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.corpus import (
    TASK_DOC_ID,
    CorpusIndex,
    TrustedDoc,
    task_table_text,
)

SPAN = "Net revenue was $9,246 million, against $6,819 million a year earlier; diluted EPS $0.75."


@pytest.mark.parametrize(
    "claim, any_status, every_status",
    [
        ("Net revenue was $9,246 million.", "anchored", "anchored"),
        (
            "Net revenue was $9,246 million, up 36% from $6,819 million.",
            "anchored",
            "unanchored",
        ),
        ("Net revenue was $9,250 million.", "unanchored", "unanchored"),  # altered
        ("Net revenue rose over the period.", "no_figures", "no_figures"),
        ("Diluted EPS was $0.75 against a forecast of 0.81.", "anchored", "unanchored"),
    ],
)
def test_every_rule_versus_any_rule(
    claim: str, any_status: str, every_status: str
) -> None:
    assert NUM.claim_number_status(claim, [SPAN]) == any_status
    assert NUM.claim_number_status(claim, [SPAN], rule="every") == every_status


def test_own_scored_value_stays_exempt_under_every() -> None:
    claim = "Diluted EPS was $0.75 and we forecast 0.81."
    assert (
        NUM.claim_number_status(claim, [SPAN], submitted=[0.81], rule="every")
        == "anchored"
    )
    # exact equality only: 0.810 == 0.81 numerically, 0.8 is not
    assert (
        NUM.claim_number_status(claim, [SPAN], submitted=[0.8], rule="every")
        == "unanchored"
    )


def test_every_rule_takes_the_union_of_cited_spans() -> None:
    claim = "Net revenue was $9,246 million and EPS was $0.75."
    assert (
        NUM.claim_number_status(claim, ["$9,246 million", "EPS $0.75"], rule="every")
        == "anchored"
    )
    missing, stated = NUM.missing_figures(claim, ["$9,246 million"])
    assert stated == 2 and missing == (Decimal("0.75"),)


def test_unknown_rule_is_refused() -> None:
    with pytest.raises(ValueError):
        NUM.claim_number_status("x 1", ["1"], rule="most")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "text, expected",
    [
        # a day-month date may not eat the decimals of an amount on the previous line
        ("close | 48.30\n  August 2026 month-to-date", ("48.30",)),
        # the year slot of "Month D, YYYY" takes a year, never the integer part of an amount
        ("at 4182.64 on February 27, 4390.12 on April 30", ("4182.64", "4390.12")),
        ("March 31, 2025 and 31 March 2025 and May 6-7, 2025", ()),
        # counts of periods are not amounts, the same as their hyphenated form
        ("rose 3.87 percent over the 26 weeks ending 2026-08-14", ("3.87",)),
        ("over the 13-week period, within 12 months, the last 10 sessions", ()),
        ("revenue rose 12 percent", ("12",)),
    ],
)
def test_grammar_fixes(text: str, expected: tuple[str, ...]) -> None:
    assert NUM.figures(text) == tuple(Decimal(x) for x in expected)


TASK = {
    "cutoff_date": "2026-03-31",
    "entities": [
        {"entity_id": "AAA", "name": "Alpha", "prior_value": 12.5},
        {"entity_id": "BBB", "name": "Beta", "prior_value": 7.25},
    ],
}


def test_task_table_text_is_the_published_rendering() -> None:
    text, rows = task_table_text(TASK)
    lines = [
        json.dumps(r, ensure_ascii=False, separators=(", ", ": "))
        for r in TASK["entities"]
    ]
    assert text == "\n".join(lines)
    assert text[slice(*rows["AAA"])] == lines[0]
    assert text[slice(*rows["BBB"])] == lines[1]


def _index() -> CorpusIndex:
    doc = TrustedDoc(
        "D1", "corpus/D1.json", "sha256:0", dt.date(2026, 1, 1), {"text": "x"}, ("AAA",)
    )
    return CorpusIndex({"D1": doc}).with_task_table(TASK, dt.date(2026, 3, 31))


def test_task_citation_is_bound_to_the_entitys_own_row() -> None:
    idx = _index()
    task = idx.resolve(TASK_DOC_ID)
    text, rows = task_table_text(TASK)
    a0, a1 = rows["AAA"]
    b0, b1 = rows["BBB"]
    assert task.admits_citation(
        "AAA", {"doc_id": "task", "span_start": a0, "span_end": a1}
    )
    assert not task.admits_citation(
        "AAA", {"doc_id": "task", "span_start": b0, "span_end": b1}
    )
    assert not task.admits_citation(
        "AAA", {"doc_id": "task", "span_start": a0, "span_end": b1}
    )
    assert not task.admits_citation(
        "ZZZ", {"doc_id": "task", "span_start": a0, "span_end": a1}
    )
    # dated at the cutoff: never post-cutoff
    rep = idx.embargo_report(
        [{"doc_id": "task", "span_start": a0, "span_end": a1}], dt.date(2026, 3, 31)
    )
    assert rep.clean
    # the task table's own labels do not make an unlabelled corpus look labelled
    bare = CorpusIndex({}).with_task_table(TASK, dt.date(2026, 3, 31))
    assert bare.labelled is False


def test_task_figure_needs_the_task_citation() -> None:
    text, rows = task_table_text(TASK)
    row = text[slice(*rows["AAA"])]
    claim = "Alpha's prior value was 12.5."
    assert NUM.claim_number_status(claim, [row], rule="every") == "anchored"
    assert (
        NUM.claim_number_status(claim, ["Alpha is a company."], rule="every")
        == "unanchored"
    )


def test_a_corpus_document_named_task_is_an_organizer_fault() -> None:
    doc = TrustedDoc(
        "task",
        "corpus/task.json",
        "sha256:0",
        dt.date(2026, 1, 1),
        {"text": "x"},
        ("AAA",),
    )
    with pytest.raises(T4OrganizerFault):
        CorpusIndex({"task": doc}).with_task_table(TASK, dt.date(2026, 3, 31))


def test_a_verbatim_quote_cut_mid_token_is_anchored() -> None:
    span = "2023-11-08 | 34.61\n2023-11-09 | 35.04\n2023-11-10 | 34.88"
    quote = "2023-11-08 | 34.61\n2023-11-09 | 35.04\n20"  # cut inside the next date
    assert NUM.figures(quote)[-1] == Decimal("20")
    assert NUM.claim_number_status(quote, [span], rule="every") == "anchored"
    # whitespace runs compare as one space; an altered figure is no longer verbatim
    assert (
        NUM.claim_number_status("2023-11-08 |   34.61", [span], rule="every")
        == "anchored"
    )
    assert (
        NUM.claim_number_status("2023-11-08 | 34.71", [span], rule="every")
        == "unanchored"
    )


def test_a_dotted_evidence_label_is_not_a_figure() -> None:
    assert NUM.figures("see (E3.3) and (E1.7); CIK0001033225") == ()
