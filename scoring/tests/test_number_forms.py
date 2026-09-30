"""Scorer 5.2.1: equivalent number forms, and the sign-aware own-value exemption.

Equivalent forms. The claim check compares a claim's figures with the figures of the passage it
cites, value to value. Before 5.2.1 several ordinary ways of writing an amount were invisible to
it, so a correct claim was marked unanchored: "cut by 25 bps" against a statement that reads "by
1/4 percentage point", "decreased 4 bps" against "Zent slid four basis points", "7.3x" and
"$212mm" (no figure read at all). Some dates and labels were read as amounts, so a claim that
copied one was penalised for a figure no passage states: "on 3/20" (3 and 20), "(1982-84=100)"
(100), "2025-03-18/19" (19), "Q4-25" (25), "Rule 12b-2" (12 billion). Each form is now read the
same way in claim and span. The value comparison itself is unchanged, so a wrong figure still
fails: 50 bps is not "1/4 percentage point", and 3.93 is not 3.95.

Sign-aware own values. A claim may state the participant's own scored values without the passage
carrying them. The exemption used to compare absolute values, so "yields declined about 20 bps"
was exempt against an upper bound of +20. It now compares the SIGNED written value. A claim may
also state its own interval as a half-width ("±10 bps"): that figure is exempt when it equals half
the width of the scored interval.

Every number here is synthetic.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

import pytest

from qfbench2_track_analysis.alignment import EntityRoster, align_predictions
from qfbench2_track_analysis.numeric import (
    _signed_figures,
    claim_number_status,
    figures,
    missing_figures,
    url_ranges,
)
from qfbench2_track_analysis.scoring import evaluate_claims

from .synthetic import StubJudge, answer_for


def _missing(
    claim: str,
    span: str,
    *,
    submitted: tuple[float, ...] = (),
    own_intervals: tuple[tuple[float, float], ...] = (),
) -> bool:
    miss, _ = missing_figures(
        claim, [span], submitted=submitted, own_intervals=own_intervals
    )
    return bool(miss)


# --- fractions and fraction words of a point ------------------------------------------------------
@pytest.mark.parametrize(
    ("claim", "span"),
    [
        (
            "Velmora trimmed 25 bps",
            "Velmora trimmed 1/4 percentage point, band 7 to 7-1/4",
        ),
        (
            "Velmora trimmed 25 basis points",
            "Velmora sees two more quarter-point trims",
        ),
        (
            "a Velmora trim of 0.25 percentage point",
            "Velmora moved 1/4 percentage point",
        ),
        ("Quillan wanted 50 bps", "Quillan wanted a 1/2 percentage-point trim"),
        ("a 25 bps move", "Quillan cuts one quarter percentage point"),
        ("a 50 bps move", "Quillan moved half a percentage point"),
        ("a 50 bps move", "Zent (half point) hike"),
        ("a 75 bps move", "Quillan moved three-quarters of a point"),
        ("a 50 bps move", "a one-half percent Quillan bump"),
        ("a 12.5 bps change", "every one-eighth point Quillan step"),
        ("a 25 bps cut", "a ¼ point cut"),
        ("Brimco needs 66.67%", "Brimco needs 66⅔% of its members"),
        ("Brimco needs 12.67%", "Brimco needs twelve and two-thirds percent"),
        ("Brimco band top 7.25%", "Brimco band 7 to 7-1/4 pct"),
        ("Brimco band 7.25-7.50", "Brimco band 7-1/4 to 7-1/2"),
    ],
)
def test_a_fraction_of_a_point_carries_its_decimal(claim: str, span: str) -> None:
    assert not _missing(claim, span)


@pytest.mark.parametrize(
    ("claim", "span"),
    [
        pytest.param(
            "Velmora trimmed 50 bps",
            "Velmora trimmed its band by 1/4 percentage point",
            id="50-vs-1/4",
        ),
        pytest.param(
            "Velmora trimmed 25 bps",
            "Quillan wanted a 1/2 percentage-point trim",
            id="25-vs-1/2",
        ),
        pytest.param(
            "Velmora trimmed 75 bps",
            "Velmora sees two more quarter-point trims",
            id="75-vs-quarter-point",
        ),
        pytest.param("Brimco band top 7.5%", "band 7-1/4", id="7.5-vs-7-1/4"),
        pytest.param(
            "a 25 bps change", "each one-eighth point change", id="25-vs-one-eighth"
        ),
    ],
)
def test_a_different_fraction_still_fails(claim: str, span: str) -> None:
    """The guards: the equivalence rewrites the form, never the value."""
    assert _missing(claim, span)


def test_a_mixed_fraction_stays_the_mixed_fraction() -> None:
    """Never after a digit and a hyphen: "4-1/4" is 4.25, not 4 and 0.25."""
    assert figures("to 4-1/4 percent") == (figures("4.25")[0],)


def test_a_fraction_without_a_percent_or_point_word_is_not_a_decimal() -> None:
    """ "1/4 of Brimco's book" is not 0.25 of a point (it stays the numbers 1 and 4), and
    "1/4/2025" is a date."""
    assert figures("1/4 of Brimco's book") == (Decimal(1), Decimal(4))
    assert figures("Brimco on 1/4/2025") == ()


# --- numbers in words, and glued forms -------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Zent slid four basis points to 3.60%", ("4", "3.60")),
        ("Zent moved twenty-five basis points", ("25",)),
        ("Zent aims at two percent", ("2",)),
        ("Zent holds one hundred fifty billion", ("150",)),
        ("Zent sees four and a half percent", ("4.5",)),
        ("a Zent 25-basis-point trim", ("25",)),
        ("a 5-percent stake", ("5",)),
        ("Zent at 7.3x and 2.6X", ("7.3", "2.6")),
        ("Zent paid $212mm and $1mil", ("212000000", "1000000")),
        ("Zent $4,321.5mmTotal Zent", ("4321500000",)),
        ("Zent holds $61.2B5 now", ("61200000000", "5")),
        ("12,345.6Total Loans", ("12345.6",)),
        ("Zent moved 1.5pp and -1ppt", ("1.5", "1")),
        ("duration of 6.3yrs", ("6.3",)),
        ("a CNY2.40 fee", ("2.40",)),
        ("Zent $318.4² at 3.2%¹", ("318.4", "3.2")),
        ("Zent ê5% and Zent é4 bps", ("5", "4")),
    ],
)
def test_the_form_reads_as_its_amount(text: str, expected: tuple[str, ...]) -> None:
    assert tuple(str(v.normalize()) for v in figures(text)) == tuple(
        str(figures(e)[0].normalize()) for e in expected
    )


@pytest.mark.parametrize(
    "text",
    [
        "three times a year",
        "Five Point Energy",
        "one of the banks",
        "a 10-year note",
        "the 10yr tenor",
        "300mm wafers",
        "under Rule 12b-2 and Rule 13a-15(e)",
    ],
)
def test_words_tenors_lengths_and_rule_numbers_stay_unread(text: str) -> None:
    assert figures(text) == ()


def test_a_spelled_amount_matches_its_digits_both_ways() -> None:
    assert not _missing("Zent slid 4 bps", "Zent slid four basis points")
    assert not _missing("Zent down four basis points", "down 4 bps")
    assert _missing("Zent slid 5 bps", "Zent slid four basis points")


def test_a_multiple_and_millions_match_their_spelled_out_forms() -> None:
    assert not _missing("Zent at 7.3 times", "Zent at 7.3x and 9.1x")
    assert not _missing("Zent paid $212 million", "Zent paid $212mm")
    assert _missing("Zent at 7.2 times", "Zent at 7.3x and 9.1x")


# --- dates and labels that are not amounts ---------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "on 3/20",
        "at 12/31 vs. 9/30",
        "Brimco met 2031-03-18/19",
        "Brimco met 2031-03-18/2031-03-19",
        "Brimco index (1982-84=100)",
        "Brimco output (index 2017=100)",
        "as of 03/2025",
        "Q4-25 and 4Q'24 and FY-24",
    ],
)
def test_the_date_or_label_is_not_an_amount(text: str) -> None:
    assert figures(text) == ()


def test_a_date_without_a_year_keeps_the_yield_check() -> None:
    assert not _missing("Zent 3Y 3.93% on 3/20", "Zent 3Y 3.93 percent")
    assert _missing("Zent 3Y 3.93% on 3/20", "Zent 3Y 3.95 percent")
    assert _missing("0.15 percent", "on 3/20 the yield")


def test_an_index_level_with_its_base_keeps_the_level_check() -> None:
    assert not _missing("Brimco index 250.123 (1982-84=100)", "Brimco 250.123")
    assert _missing("Brimco index 250.124 (1982-84=100)", "Brimco 250.123")


def test_a_quarter_label_and_a_value_in_a_table_are_left_alone() -> None:
    """ "Q4 25" with a space is not blanked: in a table the two digits may be the next value."""
    assert figures("Q4 25") != ()


# --- the sign-aware own-value exemption ------------------------------------------------------------
OWN = (-10.0, -25.0, 0.0)


@pytest.mark.parametrize(
    "claim",
    [
        "we forecast -25 bps",
        "we forecast (25) bps",
        "we forecast −25 bps",
        "Zent declines of roughly 10 bps",
        "a 10 bps decline",
        "a 25 bps cut",
        "10 bps lower",
    ],
)
def test_a_negative_written_value_is_exempt_against_a_negative_own_value(
    claim: str,
) -> None:
    assert not _missing(claim, "Zent note", submitted=OWN)


@pytest.mark.parametrize(
    "claim",
    [
        pytest.param("Zent rose 20 bps", id="rise-before"),
        pytest.param("Zent at +25 bps", id="plus-sign"),
        pytest.param("Zent saw an increase of 10 bps", id="rise-noun"),
    ],
)
def test_a_written_rise_is_not_exempt_against_a_negative_own_value(claim: str) -> None:
    assert _missing(claim, "Zent note", submitted=(-10.0, -20.0, -25.0))


@pytest.mark.parametrize(
    ("claim", "own"),
    [
        pytest.param(
            "Velmora noted the 25 bps trim", (-25.0,), id="direction-unstated"
        ),
        pytest.param("the Fed's 20 bps projection", (-20.0,), id="unsigned"),
        pytest.param("a 10 bps rise", (-10.0,), id="rise-word-after"),
        pytest.param("We expect a 62.5% loss ratio.", (62.5,), id="loss-ratio-after"),
        pytest.param(
            "EPS was $1.20 down from $1.35 a year ago", (1.20, 1.35), id="down-from"
        ),
        pytest.param("We see 4.25% cuts ahead", (4.25,), id="cuts-ahead"),
        pytest.param("a 4.2% decline rate", (4.2,), id="decline-rate"),
        pytest.param("a 2.1% easing bias", (2.1,), id="easing-bias"),
        pytest.param("a 3.1% drop-off", (3.1,), id="drop-off"),
        pytest.param("Velmora moved (25 bps)", (-25.0,), id="aside-parentheses"),
        pytest.param("EPS of -$0.12", (-0.12,), id="minus-before-currency"),
        pytest.param("EPS of \u2013$0.12", (-0.12,), id="en-dash-before-currency"),
        pytest.param("EPS of \u2011$0.12", (-0.12,), id="nb-hyphen-before-currency"),
        pytest.param("After the cut we forecast 4.25.", (4.25,), id="fall-word-nearby"),
        pytest.param("The loss ratio will be 62.5.", (62.5,), id="loss-ratio"),
        pytest.param("Zent sees revenue of $5.9bn", (5.9,), id="scale-billions"),
        pytest.param("Zent sees revenue of $5.9bn", (5.9e9,), id="scale-units"),
        pytest.param("Zent sees 12%", (0.12,), id="percent-of-ratio"),
        pytest.param("Zent guidance \u2013 $1.20", (1.20,), id="spaced-dash-currency"),
        pytest.param("Zent guidance \u2013 1.20", (1.20,), id="spaced-dash"),
    ],
)
def test_a_restatement_of_an_own_value_stays_exempt(
    claim: str, own: tuple[float, ...]
) -> None:
    """The own value, whatever direction the text leaves unsaid, and at any scale step the span
    matcher allows ("$5.9bn" for 5.9 in billions)."""
    assert not _missing(claim, "Zent note", submitted=own)


@pytest.mark.parametrize(
    ("claim", "own"),
    [
        pytest.param("EPS of -$0.12", (0.12,), id="minus-vs-positive"),
        pytest.param("Zent sees revenue of $5.8bn", (5.9,), id="other-value"),
        pytest.param("Zent sees revenue of $59bn", (5.9,), id="not-a-scale-step"),
    ],
)
def test_a_contradicting_or_different_value_is_not_exempt(
    claim: str, own: tuple[float, ...]
) -> None:
    assert _missing(claim, "Zent note", submitted=own)


def test_a_fall_is_not_the_upper_bound() -> None:
    """The traced case: "declined about 20 bps" is -20, not the upper bound +20."""
    claim = "Zent 2Y declined about 20 bps"
    assert _missing(claim, "Zent note", submitted=(10.0, 0.0, 20.0))
    assert not _missing(claim, "Zent note", submitted=(10.0, -20.0, 0.0))
    assert _missing("Velmora cut rates by 25 bps", "Zent note", submitted=(25.0,))


def test_a_negative_bound_is_not_a_positive_own_value() -> None:
    assert not _missing("upper bound 5 bps", "text", submitted=(-10.0, -25.0, 5.0))
    assert _missing("upper bound -5 bps", "text", submitted=(-10.0, -25.0, 5.0))


@pytest.mark.parametrize(
    "claim",
    [
        pytest.param("Zent 10Y to fall to 4.10%", id="fall-to-a-level"),
        pytest.param("Zent declined from 4.10%", id="fall-from-a-level"),
        pytest.param("Zent lower bound 4.10%", id="lower-bound"),
        pytest.param("Zent narrowed sharply, Zent sees 4.10", id="new-clause"),
    ],
)
def test_a_level_after_a_fall_word_keeps_its_sign(claim: str) -> None:
    """A figure after "to", "from" or "at", a "lower bound", and a figure the fall word does not
    govern are levels, not the fall: the participant's own positive forecast stays exempt."""
    assert not _missing(claim, "text", submitted=(4.10,))


def test_signed_figures_reads_the_same_figures_as_figures() -> None:
    text = "fell 20 bps to 4.10%, a (2)% Zent loss, ±10 bps, on 3/20, Zent 1/4 percentage point cut"
    assert tuple(v for v, _, _ in _signed_figures(text)) == figures(text)
    assert [(s, pm) for _, s, pm in _signed_figures(text)] == [
        (-1, False),
        (0, False),
        (-1, False),
        (0, True),
        (0, False),
    ]


# --- the "±" half-width of the own interval --------------------------------------------------------
def test_a_half_width_of_the_own_interval_is_exempt() -> None:
    assert not _missing(
        "we forecast 10 bps ±10bps",
        "text",
        submitted=(10.0, 0.0, 20.0),
        own_intervals=((0.0, 20.0),),
    )
    assert not _missing("a band of +/- 0.15", "text", own_intervals=((4.1, 4.4),))
    assert not _missing("plus or minus 0.15", "text", own_intervals=((4.1, 4.4),))


def test_a_half_width_is_exempt_only_as_a_half_width() -> None:
    """A "±" figure is not an own value, and a plain figure equal to the half-width is not
    exempt."""
    assert _missing(
        "we forecast ±20 bps",
        "text",
        submitted=(10.0, 0.0, 20.0),
        own_intervals=((0.0, 20.0),),
    )
    assert _missing("a move of 10 bps", "text", own_intervals=((0.0, 20.0),))
    assert _missing("a band of ±12 bps", "text", own_intervals=((0.0, 20.0),))


ROSTER = ("SYN-A",)
SPAN = "Synthetic Issuer A reported revenue of $5.2 billion, up 8% year over year."


def _status(claim: str, *, interval_scored: bool, lo: float, hi: float) -> str:
    answer = answer_for(
        entities=ROSTER, claim_text=claim, point_forecast=1.0, lo=lo, hi=hi
    )
    answer["target_type"] = "regression"
    answer["entity_predictions"][0]["claims"][0]["span_end"] = len(SPAN)
    aligned: Any = align_predictions(
        answer,
        EntityRoster(entity_ids=ROSTER),
        target_type="regression",
        interval_level=0.90,
    )
    report = evaluate_claims(
        aligned,
        lambda _doc_id: {"text": SPAN},
        StubJudge((SPAN,)),
        target_type="regression",
        interval_scored=interval_scored,
        contradiction_bar=0.9,
    )
    (verdict,) = report.verdicts
    return verdict.status


def test_the_gate_exempts_the_half_width_only_when_the_interval_is_scored() -> None:
    claim = "Zent $5.2 billion, band ±0.15"
    assert _status(claim, interval_scored=True, lo=0.85, hi=1.15) == "neutral"
    assert _status(claim, interval_scored=False, lo=0.85, hi=1.15) == "unanchored"


def test_the_any_rule_is_unchanged() -> None:
    """Rule "any" (before 5.2.x) keeps its sign-blind exemption; only rule "every" reads signs."""
    assert (
        claim_number_status("yields declined 20 bps", ["text"], submitted=(20.0,))
        == "no_figures"
    )
    assert (
        claim_number_status(
            "yields declined 20 bps", ["text"], submitted=(20.0,), rule="every"
        )
        == "unanchored"
    )


# --- a range, a period word, a month/day that is not a date (5.2.1 review follow-ups) ---------------
@pytest.mark.parametrize(
    "claim",
    [
        "our band is 3.9%-4.2%",
        "our band is 3.9-4.2%",
        "our band is 3.9–4.2%",
        "our band is 3.9 to 4.2",
        "our band is 3.9%\u22124.2%",
    ],
)
def test_an_own_interval_written_as_a_range_is_exempt(claim: str) -> None:
    """A dash straight after a figure is a range, not a minus: both bounds are the own values."""
    assert not _missing(claim, "text", submitted=(4.0, 3.9, 4.2))


@pytest.mark.parametrize(
    "claim", ["the yield moved -0.3%", "the yield fell 0.3", "rates of 2.1%, -0.3%"]
)
def test_a_genuine_negative_keeps_its_sign(claim: str) -> None:
    assert _missing(claim, "text", submitted=(0.3, 2.1))
    assert not _missing(claim, "text", submitted=(-0.3, 2.1))


@pytest.mark.parametrize(
    "text",
    [
        "Zent quarter-over-quarter percent",
        "quarter-on-quarter percent",
        "Zent fourth quarter point",
        "Zent linked quarter percent",
        "second half percent",
        "q/q percent",
        "QoQ percent",
    ],
)
def test_a_period_word_is_not_a_fraction_of_a_point(text: str) -> None:
    assert Decimal("0.25") not in figures(text) and Decimal("0.5") not in figures(text)


@pytest.mark.parametrize(
    "text",
    [
        "a quarter percent cut",
        "the quarter-point cut",
        "each one-eighth point change",
        "on a quarter-point cut",
        "to a half-point move",
        "at one-half percent",
    ],
)
def test_the_fraction_word_still_reads_without_a_period_word(text: str) -> None:
    assert figures(text) and figures(text)[0] in (
        Decimal("0.25"),
        Decimal("0.5"),
        Decimal("0.125"),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a 3/2 split", (3, 2)),
        ("Zent unit is a 1/64 th interest", (1, 64)),
        ("Zent unit is 1/4,096th interest", (1, 4)),
        ("Zent weights 70/30", (70, 30)),
        ("the Phase 1/2 trial", (1, 2)),
        ("24/7 access", (24, 7)),
        ("3/4 of the book", (3, 4)),
        ("$0.16 2/3 par value", (Decimal("0.16"), 2, 3)),
        ("Zent on 14/35", (14, 35)),
        ("one share and 1/3 warrant", (1, 3)),
        ("Zent at 3/8\u201d thickness", (3, 8)),
        ("Zent from 1/2 mile out", (1, 2)),
        ("Zent at 1/4,096th", (1, 4)),
        ("Zent on 1/25 th", (1, 25)),
    ],
)
def test_a_slash_that_is_not_a_date_stays_numbers(
    text: str, expected: tuple[Any, ...]
) -> None:
    assert figures(text) == tuple(Decimal(v) for v in expected)


@pytest.mark.parametrize(
    "text",
    [
        "on 3/20",
        "at 12/31 vs. 9/30",
        "Zent ended 8/14, 8/13, and 8/12",
        "as of 12/31)",
        "from 3/3 to 3/7",
        "Zent 3/3 - 3/7 window",
    ],
)
def test_a_month_day_date_with_a_date_word_is_not_an_amount(text: str) -> None:
    assert figures(text) == ()


# --- own values without float noise, and the interval level (5.2.1 review follow-up) -----------------
def test_an_own_value_is_read_without_float_noise() -> None:
    """A submitted 1.2999999999999998 is the 1.3 the claim writes; a real 1.35 is not."""
    assert not _missing(
        "Zent band 1.3 to 3.8", "Zent note", submitted=(2.5, 1.2999999999999998, 3.8)
    )
    assert _missing("Zent band 1.3 to 3.8", "Zent note", submitted=(2.5, 1.35, 3.8))
    assert not _missing(
        "Zent band ±0.15", "Zent note", own_intervals=((0.1, 0.39999999999999997),)
    )


@pytest.mark.parametrize(
    "claim",
    ["the 90% band", "a 0.9 band", "the 90 percent band", "the 90-percent band"],
)
def test_the_interval_level_is_an_own_value(claim: str) -> None:
    miss, _ = missing_figures(claim, ["Zent note"], own_levels=(0.9,))
    assert miss == ()
    miss, _ = missing_figures(claim, ["Zent note"])
    assert miss != ()


def _level_status(claim: str, *, interval_level: float | None) -> str:
    answer = answer_for(
        entities=ROSTER, claim_text=claim, point_forecast=1.0, lo=0.85, hi=1.15
    )
    answer["target_type"] = "regression"
    answer["entity_predictions"][0]["claims"][0]["span_end"] = len(SPAN)
    aligned: Any = align_predictions(
        answer,
        EntityRoster(entity_ids=ROSTER),
        target_type="regression",
        interval_level=0.90,
    )
    report = evaluate_claims(
        aligned,
        lambda _doc_id: {"text": SPAN},
        StubJudge((SPAN,)),
        target_type="regression",
        interval_scored=True,
        contradiction_bar=0.9,
        interval_level=interval_level,
    )
    (verdict,) = report.verdicts
    return verdict.status


def test_the_gate_exempts_the_units_interval_level() -> None:
    claim = "Revenue of $5.2 billion; our 90% band"
    assert _level_status(claim, interval_level=0.9) == "neutral"
    assert _level_status(claim, interval_level=None) == "unanchored"


def test_score_unit_passes_the_interval_level_to_the_claim_check(tmp_path: Any) -> None:
    import json

    from qfbench2_track_analysis.judge_factory import build_smoke_judge
    from qfbench2_track_analysis.scoring import score_unit

    from .synthetic import SUPPORTING_TEXT, build_unit

    unit = build_unit(tmp_path, entities=ROSTER, with_outcome=True)
    answer = answer_for(entities=ROSTER, claim_text=SUPPORTING_TEXT + " Our 90% band.")
    answer["entity_predictions"][0]["claims"][0]["span_end"] = len(SUPPORTING_TEXT)
    out = tmp_path / "res"
    out.mkdir()
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    outcome = score_unit(
        {"unit_dir": unit, "output_dir": out},
        judge=StubJudge((SUPPORTING_TEXT,)),
        judge_provenance=provenance,
    )
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["unanchored_claim_count"] == 0


@pytest.mark.parametrize(
    "claim",
    [
        "Zent revenue grew 90%",
        "Zent spread of 900 bps",
        "Zent paid $900M",
        "Zent at 0.9 times",
        "Zent band of 900 bps",
        "Zent range of $0.09",
    ],
)
def test_the_level_is_exempt_only_as_the_level(claim: str) -> None:
    """Only 0.9 or 90, and only beside an interval word: a fabricated 90% growth is not hidden."""
    miss, _ = missing_figures(claim, ["Zent note"], own_levels=(0.9,))
    assert miss != ()


@pytest.mark.parametrize(
    "span",
    [
        "Zent slowed at the half point of the year",
        "Zent saw the quarter percent change",
        "Zent kept the quarter point estimate",
    ],
)
def test_a_fraction_word_without_a_move_does_not_anchor_bps(span: str) -> None:
    """A fraction word is a move of a point only beside a count, a change verb or a change noun."""
    assert _missing("Velmora will move 50 bps", span)
    assert _missing("Velmora will move 25 bps", span)


def test_a_glued_points_suffix_reads_as_points() -> None:
    assert tuple(str(v) for v in figures("Zent moved 2.5pts and 1pt")) == ("2.5", "1")
    assert not _missing("Zent moved 2.5 points", "Zent up 2.5pts")


@pytest.mark.parametrize(
    "claim",
    ["Zent +/-10 bps", "Zent +/- 10 bps", "Zent ±10 bps", "Zent plus or minus 10 bps"],
)
def test_every_written_half_width_form_is_exempt(claim: str) -> None:
    assert not _missing(claim, "Zent note", own_intervals=((90.0, 110.0),))
    assert _missing(claim, "Zent note", own_intervals=((90.0, 114.0),))


@pytest.mark.parametrize(
    "claim", ["Zent ranges of ±0.9", "Zent band +/-0.9", "Zent band plus or minus 90"]
)
def test_a_half_width_is_never_the_interval_level(claim: str) -> None:
    miss, _ = missing_figures(claim, ["Zent note"], own_levels=(0.9,))
    assert miss != ()


# --- numbers inside a web address are not figures (5.2.2) -------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "Zent filing (https://www.example.invalid/news/2031/05/release-17.html).",
        "See www.example.invalid/data/series/42 today.",
        "Source: https://example.invalid/a?id=77&y=2031 and more.",
        "[link](https://ex.invalid/p/88) here",
        "https://example.invalid/cgi-bin/browse?action=getcompany&CIK=0000999991",
        "https://example.invalid/Archives/edgar/data/999991/000099999131000012/x.htm",
        "http://example.invalid:8080/q",
    ],
)
def test_numbers_inside_a_url_are_not_figures(text: str) -> None:
    assert figures(text) == ()


def test_figures_outside_a_url_are_still_read() -> None:
    assert [
        str(v)
        for v in figures(
            "Source: https://example.invalid/a?id=77, net $4.2B and 12 bps"
        )
    ] == [
        "4.2E+9",
        "12",
    ]


def test_a_url_in_a_claim_no_longer_makes_it_unanchored() -> None:
    claim = "Zent revenue grew 12% per https://example.invalid/a?CIK=0000999991"
    assert not _missing(claim, "Zent revenue grew 12%")


def test_a_url_in_a_span_no_longer_anchors_a_figure() -> None:
    span = "Zent filing at https://example.invalid/Archives/edgar/data/999991/0000999991-31-000012.txt"
    assert _missing("Zent holds 999991 units", span)


def test_a_wrong_figure_outside_the_url_is_still_checked() -> None:
    assert _missing(
        "Zent revenue grew 13% per https://example.invalid/12", "Zent revenue grew 12%"
    )


@pytest.mark.parametrize(
    "text",
    [
        "see https\uff1a\uff0f\uff0fexample.invalid/series/42",
        "see ht\u200btps://example.invalid/a?id=77",
        "see https\u2236\u2044\u2044example.invalid/p/88",
        "see //example.invalid/data/31",
    ],
)
def test_numbers_inside_a_disguised_url_are_not_figures(text: str) -> None:
    assert figures(text) == ()


def test_a_long_letter_run_is_scanned_for_urls_in_linear_time() -> None:
    """The URL scheme is bounded to 64 characters, so a 1 MB run of letters is scanned once, not
    once per starting letter (unbounded, it took minutes). The bound is generous: about 0.1 s here."""
    text = "a" * 1_000_000
    start = time.perf_counter()
    assert url_ranges(text) == []
    assert time.perf_counter() - start < 1.0


def test_a_scheme_up_to_64_characters_is_masked_whole() -> None:
    scheme = "a" * 64
    assert url_ranges(f"{scheme}://x.test/1 end") == [(0, 64 + len("://x.test/1"))]


def test_a_longer_scheme_is_masked_from_its_last_64_characters() -> None:
    """The fast check on the bound (the 1 MB timing test above takes minutes to fail): with an
    unbounded scheme the whole 70-letter run would match from index 0."""
    scheme = "a" * 70
    assert url_ranges(f"{scheme}://x.test/1 end") == [(6, 70 + len("://x.test/1"))]
