"""Scorer 5.2.0: refinements of the every-figure rule.

1. Reading fixes: a decimal whose digits start
   like a year ("1.2002" read as 1), a decimal before a hyphenated word ("2.6-percent" read as 2),
   "<label>: <verbatim quote>" claims (the quote's cut edges are not figures; the label's figures
   are still checked), glued basis points / "MM" ("25bp", "$1,537MM" were never read), and "9M 2025"
   (nine months, not 9 million).
2. The unit's own entity names and tickers are not figures ("Phillips 66", "S&P 500", "3M"); only
   those, and only as written in the name.
3. A cited span longer than `FIGURE_SPAN_CAP` characters anchors no figure (verbatim quotes aside).
4. A claim longer than `CLAIM_MAX_JUDGE_TOKENS` judge tokens, or than `CLAIM_TEXT_MAX_CHARS`
   characters, is malformed (false).
Each positive case fails on the reader before these refinements; each negative case shows the change does not exempt
more than it says. Synthetic inputs only.
"""

from __future__ import annotations

import json
import pathlib
from decimal import Decimal

import pytest

from qfbench2_track_analysis import numeric as NUM
from qfbench2_track_analysis.alignment import CLAIM_TEXT_MAX_CHARS
from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.scoring import (
    CLAIM_MAX_JUDGE_TOKENS,
    FIGURE_SPAN_CAP,
    _claim_too_long,
    unit_entity_names,
)

from .test_claim_penalty import SPAN, TextJudge, _answer, _score, _unit


def every(claim: str, spans: list[str], **kw: object) -> str:
    return NUM.claim_number_status(claim, spans, rule="every", **kw)  # type: ignore[arg-type]


# --- 1. reading fixes ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("| 1.2002 | 0.2027 |", ("1.2002", "0.2027")),  # not years
        ("FY2024, Q1 2026 and 2025", ()),  # still years
        ("It cost $2030 million", ("2030",)),
        ("after a 2.6-percent increase", ("2.6",)),
        ("the 10-K, the 13-week bill, the 2-Year note", ()),
        ("cut by 25bp, then -5bps", ("25", "5")),
        ("NII of $1,537MM on 300mm wafers", ("1.537E+9",)),
        ("EPS fell 17% in 9M 2025", ("17",)),
        ("it paid $9M in 2025; it bought 9M shares", ("9E+6", "9E+6")),
    ],
)
def test_reading_fixes(text: str, expected: tuple[str, ...]) -> None:
    assert NUM.figures(text) == tuple(Decimal(x) for x in expected)


def test_an_invented_decimal_is_no_longer_hidden_by_the_year_reading() -> None:
    span = "| 1.2002 | 0.8827 |"
    assert every("The ratio was 1.2089.", [span]) == "unanchored"
    assert every("The ratio was 1.2002.", [span]) == "anchored"


def test_label_and_verbatim_quote() -> None:
    span = "Net income per share, basic and diluted $ 3,472 $ 6,918 $ 1,845 $ 2,733"
    # the quote is cut inside "1,845": the cut is not a figure
    assert (
        every(
            "Evidence for Synthetic Issuer A: Net income per share, basic and diluted $ 3,472 $ 6,918 $ 1,8",
            [span],
        )
        == "anchored"
    )
    # the label's own figures are checked like any claim figure
    assert (
        every(
            "Revenue was $4.1 billion: Net income per share, basic and diluted $ 3,472 $ 6,918",
            [span],
        )
        == "unanchored"
    )
    # a quote that is not verbatim is read as an ordinary claim
    assert (
        every(
            "Evidence for Synthetic Issuer A: Net income per share, basic and diluted $ 3,472 $ 6,928",
            [span],
        )
        == "unanchored"
    )
    # a tail shorter than the minimum is not a quote
    assert (
        every(
            "Evidence: $ 6,9",
            ["Net income per share, basic and diluted $ 3,472 $ 6,918"],
        )
        == "unanchored"
    )


# --- 2. the unit's own names ---------------------------------------------------------------------
NAMES = ("E-mini S&P 500 futures", "Phillips 66", "3M Company", "PSX")


@pytest.mark.parametrize(
    "claim, span, expected",
    [
        ("Phillips 66 reported lower refining margins.", "Margins fell.", "no_figures"),
        (
            "S&P 500 net exposure dropped to -4.62 percent.",
            "net exposure of -4.62 percent",
            "anchored",
        ),
        ("3M's tax rate was 23.7%.", "a rate of 23.7%", "anchored"),
        # not the name: still figures
        ("Phillips 67 reported lower margins.", "Margins fell.", "unanchored"),
        # the spans carry what a name blanked without its boundary would leave ("0", ".5")
        ("S&P 5000 positioning fell.", "positioning fell 0.1", "unanchored"),
        (
            "Phillips 66.5 reported lower margins.",
            "Margins fell 0.5 points.",
            "unanchored",
        ),
        ("Phillips 66 earned 77 cents.", "Margins fell.", "unanchored"),
        ("Fund 42 Holdings bought the notes.", "bought the notes", "unanchored"),
        ("phillips 66 reported lower margins.", "Margins fell.", "unanchored"),
    ],
)
def test_only_the_units_own_names_are_exempt(
    claim: str, span: str, expected: str
) -> None:
    assert every(claim, [span], names=NAMES) == expected
    # without the names every one of these numbers is a figure again
    assert every(claim, [span]) in ("unanchored", "anchored")
    if expected == "no_figures":
        assert every(claim, [span]) == "unanchored"


def test_another_units_name_is_not_exempt() -> None:
    assert (
        every(
            "Phillips 66 reported lower margins.", ["Margins fell."], names=("Valero",)
        )
        == "unanchored"
    )


def test_unit_entity_names_reads_name_id_and_tickers() -> None:
    task = {
        "entities": [
            {
                "entity_id": "PSX",
                "name": "Phillips 66",
                "holding_company_ticker": "XYZ",
                "sector": "Energy 1",
            },
            {"entity_id": "MMM", "name": "3M Company"},
        ]
    }
    assert unit_entity_names(task) == ("PSX", "Phillips 66", "XYZ", "MMM", "3M Company")


def test_names_reach_the_scorer(tmp_path: pathlib.Path) -> None:
    unit = _unit(tmp_path)
    task_path = unit / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["entities"][0]["name"] = "Synthetic 66 Holdings"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    claim = "Synthetic 66 Holdings reported revenue of $5.2 billion."
    outcome = _score(
        tmp_path,
        unit,
        _answer({"SYN-A": [claim, "Synthetic 67 Holdings grew."]}),
        TextJudge(),
    )
    got = {c["claim"]: tuple(c["reasons"]) for c in outcome.diagnostics["false_claims"]}
    assert claim not in got
    assert got["Synthetic 67 Holdings grew."] == ("unanchored",)


# --- 3. span cap ---------------------------------------------------------------------------------
FILLER = "Filler text without amounts. " * 400


def test_a_span_over_the_cap_anchors_no_figure() -> None:
    assert FIGURE_SPAN_CAP == 8000
    deep = FILLER + " value 4,321.77 here"
    early = "value 4,321.77 " + FILLER
    assert len(deep) > FIGURE_SPAN_CAP and len(early) > FIGURE_SPAN_CAP
    assert (
        every("The figure was 4,321.77.", [deep], span_cap=FIGURE_SPAN_CAP)
        == "unanchored"
    )
    assert (
        every("The figure was 4,321.77.", [early], span_cap=FIGURE_SPAN_CAP)
        == "unanchored"
    )
    assert every("The figure was 4,321.77.", [deep]) == "anchored"  # no cap: whole span
    # a verbatim quote is recognised in a span of any length
    assert every("value 4,321.77 here", [deep], span_cap=FIGURE_SPAN_CAP) == "anchored"
    # another cited span within the cap still anchors
    assert (
        every(
            "The figure was 4,321.77.",
            [deep, "value 4,321.77"],
            span_cap=FIGURE_SPAN_CAP,
        )
        == "anchored"
    )


def test_the_cap_boundary() -> None:
    at_cap = "v 4,321.77 " + "x" * (FIGURE_SPAN_CAP - 11)
    assert len(at_cap) == FIGURE_SPAN_CAP
    assert every("It was 4,321.77.", [at_cap], span_cap=FIGURE_SPAN_CAP) == "anchored"
    assert (
        every("It was 4,321.77.", [at_cap + "x"], span_cap=FIGURE_SPAN_CAP)
        == "unanchored"
    )


# --- 4. claim length guard -----------------------------------------------------------------------
class CountingJudge(TextJudge):
    def claim_tokens(self, hypothesis: str) -> int:
        return len(hypothesis.split())


def test_a_claim_over_the_token_cap_is_malformed(tmp_path: pathlib.Path) -> None:
    long_claim = " ".join(["word"] * (CLAIM_MAX_JUDGE_TOKENS + 1))
    at_cap = " ".join(["word"] * CLAIM_MAX_JUDGE_TOKENS)
    judge = CountingJudge()
    outcome = _score(
        tmp_path, _unit(tmp_path), _answer({"SYN-A": [long_claim, at_cap]}), judge
    )
    got = {c["claim"]: tuple(c["reasons"]) for c in outcome.diagnostics["false_claims"]}
    assert got[long_claim] == ("malformed",)
    assert at_cap not in got
    assert long_claim not in {h for _, h in judge.calls}  # the judge never read it


def test_a_claim_over_the_character_cap_is_malformed(tmp_path: pathlib.Path) -> None:
    """A claim longer than 4,000 characters is malformed (false) and never reaches the judge."""
    assert CLAIM_TEXT_MAX_CHARS == 4000
    over = "w" * (CLAIM_TEXT_MAX_CHARS + 1)
    at_cap = "v" * CLAIM_TEXT_MAX_CHARS
    judge = TextJudge()
    outcome = _score(
        tmp_path, _unit(tmp_path), _answer({"SYN-A": [over, at_cap]}), judge
    )
    got = {c["claim"]: tuple(c["reasons"]) for c in outcome.diagnostics["false_claims"]}
    assert got[over] == ("malformed",)
    assert at_cap not in got
    judged = {h for _, h in judge.calls}
    assert over not in judged and at_cap in judged


def test_the_guard_needs_a_counting_judge_where_it_matters() -> None:
    assert _claim_too_long(TextJudge(), "x", require_window=False) is False
    with pytest.raises(T4OrganizerFault):
        _claim_too_long(TextJudge(), "x", require_window=True)

    class Ensemble:
        _judges = (CountingJudge(),)

    with pytest.raises(T4OrganizerFault):
        _claim_too_long(Ensemble(), "x")


def test_span_used_by_the_shared_fixture_is_short() -> None:
    assert len(SPAN) < FIGURE_SPAN_CAP


def test_the_span_cap_reaches_the_scorer(tmp_path: pathlib.Path) -> None:
    from .synthetic import PRE_CUTOFF_DOC, _doc, build_unit

    text = FILLER + " revenue of $7.4 billion here"
    unit = build_unit(
        tmp_path,
        entities=("SYN-A",),
        with_outcome=True,
        docs={PRE_CUTOFF_DOC: _doc(PRE_CUTOFF_DOC, "2026-02-01", text)},
    )
    answer = _answer(
        {"SYN-A": ["Revenue was $7.4 billion.", "revenue of $7.4 billion here"]}
    )
    for row in answer["entity_predictions"]:
        for claim in row["claims"]:
            claim["span_end"] = len(text)
    outcome = _score(tmp_path, unit, answer, TextJudge())
    got = {c["claim"]: tuple(c["reasons"]) for c in outcome.diagnostics["false_claims"]}
    assert got == {"Revenue was $7.4 billion.": ("unanchored",)}


# --- optional: a verbatim quote is never put to the contradiction check --------------------------
def test_a_verbatim_quote_is_not_put_to_the_judge(tmp_path: pathlib.Path) -> None:
    quote = SPAN[:60]
    judge = TextJudge(
        contradicted=(quote, "Synthetic Issuer A reported lower revenue.")
    )
    outcome = _score(
        tmp_path,
        _unit(tmp_path),
        _answer({"SYN-A": [quote, "Synthetic Issuer A reported lower revenue."]}),
        judge,
    )
    got = {c["claim"]: tuple(c["reasons"]) for c in outcome.diagnostics["false_claims"]}
    assert quote not in got
    assert got["Synthetic Issuer A reported lower revenue."] == ("contradicted",)
    assert quote not in {h for _, h in judge.calls}
