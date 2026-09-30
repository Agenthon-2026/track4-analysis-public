"""The numeric backstop: a claim's figures must be anchored in the passage it cites.

Why it exists. The faithfulness gate asks the NLI judge whether the cited passage entails the
participant's *claim*. NLI models are weak exactly where a financial claim is most easily
fabricated: a number. "Revenue was $5.9 billion" against a passage that says "$5.2 billion"
reads as topically consistent to a cross-encoder trained on MNLI. So the numbers are checked by
exact code, before any model runs, and the model is asked only about the claims whose figures
the passage actually carries. `scoring/tests/test_numeric_backstop.py` enumerates the cases.

What the check is, precisely. A **figure** is a number that states *how much*: ``12%``, ``1,234``,
``$5.9 billion``, ``0.28``, ``.28%``, ``-2.55``, ``4-1/4`` (a fraction, 4.25). It is NOT a number
that states *when* or *which*: a date in any common rendering (``2025-03-31``, ``March 31, 2025``,
``31 March 2025``, ``3/31/2025``), a year or fiscal period (``2024``, ``FY2024``, ``Q1 2026``,
``2026Q1``), an ordinal (``3rd``), an identifier (``CIK 1033225``, ``CERT 58210``,
``S000002564``), a form or item number (``10-K``, ``8-K``, ``Item 2.02``), or a hyphenated count
word (``10-year``, ``2-Year``). None of those is evidence of an amount, and they are the numbers
an honest claim most often carries without the cited span repeating them -- a claim naming the
filing it read, the period it covers or the form it came from is not thereby fabricating a
figure.

A claim's figures are compared with the figures of every span the claim cites, **value to
value** after normalisation: thousands separators dropped (``1,234`` = ``1234``); a trailing
scale suffix applied (``$8.5M`` = ``8,500,000``); percent and ratio forms matched both ways
(``12%`` = ``0.12``); table scale tolerated in steps of a thousand (``$1.5 billion`` matches
``1,498,614`` in a table headed "in thousands"); sign ignored (``(2)%`` = ``-2%`` = ``2%``); and
the span's figure **rounded to the precision the claim wrote it at** (``$4.08 million`` matches
``$4,075,000``, ``0.28%`` matches ``0.2812``). Every one of these only ever ADMITS more: a
coincidental match is a claim the judge is then asked about, and the composite and the reasoning
grader still score the work. A false refusal costs W = 0.0, as much as the worst possible
prediction, so the tolerance is deliberately on the admitting side.

The participant's own SCORED values for the entity are excluded from the claim's figures: a claim
that says what the participant predicts is stating the output, not citing evidence, and the
forecast is by construction not in the corpus. From 5.1.3 the exclusion is narrow. Only a value the
unit scores is exempt (the point forecast on a regression or ranking unit; the interval bounds only
when the unit's interval leg is scored; never the rank), and only a claim figure EXACTLY equal to
it: no scale step, no rounding. Before 5.1.3 any submitted value, rank and unscored bounds
included, exempted every figure it matched under the span tolerances, so a participant could set
an unscored bound to a fabricated amount and have the claim skip this check. The caller
(`scoring.evaluate_claims`) chooses which values are scored; this module only compares.

**The verdict is per claim and it is "anchored" or "unanchored", not "every figure present".**
A claim is *unanchored* when it carries at least one figure and NONE of them appears in any
passage it cites. That is the case the check is sure about: a claim whose every amount is absent
from the evidence it points at is misattributed evidence, whatever the prose says. A claim in
which at least one figure is present in the passage may legitimately carry others the passage
does not state -- a change computed from two rows, a value copied from the task table, a
rounded total -- and exact code cannot tell a derived figure from an invented one. "Every figure
must be present" would therefore refuse ordinary honest work: a claim that quotes a row and then
states the change it implies, or that repeats the target value the task table already gave, has
fabricated nothing. The narrower rule is the one this gate's own principle -- refuse only what it
is sure about -- selects. Its limit is stated, not hidden: a single fabricated figure beside a
genuine one passes this check and is left to the judge and to reasoning grading.

A claim with no figures at all passes this check and goes straight to the judge.

**From 5.2.x the scorer uses rule "every"** (`claim_number_status(..., rule="every")`,
selected by `scoring.evaluate_claims(figure_rule=...)`): a claim is unanchored when ANY of its
figures appears in none of its cited spans, read WHOLE (not the judge's window). The paragraph above
is the reasoning for "any" under an all-or-nothing admission gate, where one false refusal cost the
unit; under the per-claim penalty a false claim costs only its share, so the check can require what
an extractive claim always satisfies. The contract says so: claims state what their passage states,
and computed figures belong in `submitted_reasons`. A figure from the task table is cited like any
other, with ``doc_id: "task"`` (`corpus.TASK_DOC_ID`). The same version fixes three date-shape
readings that blanked part of an amount ("56.50\\n July 2026" read as 56; "January 30, 5611.85" read as
0.85) and reads a count of periods ("13 weeks", "10 sessions") as not a figure, like its hyphenated
form ("13-week").

**5.2.1** reads equivalent forms of a number as the same figure in claim and span: a fraction of a
point ("1/4 percentage point", "quarter-point", "¼ point" are 0.25, so the x100 scale step relates
them to 25 bps), a number in words before a unit ("four basis points"), and glued forms ("7.3x",
"$212mm", "1.5pp", "$318.4²"). Dates without a year ("3/20"), two-day meeting dates, index bases
("1982-84=100") and rule numbers ("Rule 12b-2") are not figures. The own-value exemption of rule
"every" allows the span matcher's scale steps (exact, no rounding) and refuses a figure whose
written direction contradicts the own value's sign (`_signed_figures`, `_is_own`); it also exempts
a "±" half-width of the scored interval. `scripts/scan_number_forms.py` checks every corpus for forms this module misreads.

**5.2.2** does not read numbers inside a web address as figures, disguised ones included (look-alike
colons and slashes, invisible characters, a scheme-less "//host", a "www." host: `url_ranges`).
"""

from __future__ import annotations

import bisect
import functools
import re
import unicodedata
from collections.abc import Iterable, Sequence
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Literal

__all__ = [
    "FIGURE_RULES",
    "ClaimNumberStatus",
    "FigureRule",
    "claim_number_status",
    "figures",
    "missing_figures",
    "name_fragments",
    "verbatim_quote",
]

ClaimNumberStatus = Literal["no_figures", "anchored", "unanchored"]

#: Which figures of a claim must a cited passage carry (5.2.x).
#: ``"any"``   before 5.2.x: at least one (the claim is "unanchored" only when NONE is carried).
#: ``"every"`` from 5.2.x: all of them (the claim is "unanchored" when ANY figure is missing).
#: A claim is an extractive fact: every amount it states is one the passage it cites states. A
#: figure computed from the evidence (a change, an average, a share) is argument, not evidence,
#: and belongs in `submitted_reasons`, where reasoning grading judges derivations. The participant's
#: own scored values stay exempt under both rules.
FigureRule = Literal["any", "every"]
FIGURE_RULES: tuple[str, ...] = ("any", "every")

#: A month token: a full month name or its standard abbreviation, ending at a word boundary, with
#: an optional abbreviation dot. The boundary matters: without it "Decreased 12.5%" read "Dec"
#: as a month and blanked "Decreased 12", leaving a bogus figure 0.5 (5.1.2).
_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\b\.?"
)

#: Text that carries a number which is not a figure. Each pattern is blanked before figures are
#: read, so "March 31, 2025" contributes neither 31 nor 2025, and "10-K" contributes nothing.
_NOT_A_FIGURE: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # ISO date
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),  # 3/31/2025
    re.compile(  # March 31, 2025 / May 6-7, 2025 / Mar. 31
        # 5.2.x: the year slot takes a year only, never the integer part of a following amount
        # ("on January 30, 5611.85" blanked "5611" and left a bogus figure 0.85).
        rf"\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*[-–]\s*\d{{1,2}})?"
        rf"(?:,?\s+(?:19|20)\d{{2}}(?![.,]\d))?\b"
    ),
    re.compile(  # 31 March 2025 / 6-7 May
        # 5.2.x: the day may not be the decimals of an amount ("56.50\n July 2026" blanked "50"
        # and left 56), and the year slot takes a year only.
        rf"(?<![\d.,])\b\d{{1,2}}(?:\s*[-–]\s*\d{{1,2}})?\s+{_MONTH}"
        rf"(?:\s+(?:19|20)\d{{2}}(?![.,]\d))?\b"
    ),
    # 5.2.x: a count of periods is not an amount, the same as its hyphenated form ("13-week",
    # below): "over the 13 weeks ending", "the last 10 sessions", "within 2 years".
    re.compile(
        r"(?<![\d.,])\b\d{1,3}\s+(?:days?|weeks?|months?|quarters?|years?|sessions?)\b",
        re.IGNORECASE,
    ),
    # 5.2.0: "9M 2025" is nine months of 2025, not 9 million (a period count, like "13 weeks")
    re.compile(r"(?<![\d.,$€£¥])\b\d{1,2}M(?=\s+(?:FY\s?)?(?:19|20)\d{2}\b)"),
    re.compile(  # 2024, FY2024, Q1 2026, 1Q26 is not covered (two-digit years are figures)
        # Not a year when the context says amount (5.1.2): a currency sign in front ("$2030
        # million"), a decimal part ("1950.5") or a scale word after ("2030 million").
        # 5.2.0: never the decimals of an amount ("1.2002" read as 1, "0.2027" as 0).
        r"\b(?:FY|CY|Q[1-4]|[1-4]Q|H[12])?[-\s]?(?<![$€£¥])(?<![$€£¥]\s)(?<![\d.,])(?:19|20)\d{2}"
        r"(?:Q[1-4]|[- ]?[1-4]Q|[-/]\d{2})?\b"
        r"(?!\.\d)(?!\s*(?:thousand|million|billion|trillion|mn|bn|tn)\b)",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d+(?:st|nd|rd|th)\b"),  # ordinals
    re.compile(r"\bItem\s+\d+(?:\.\d+)?\b", re.IGNORECASE),  # Item 2.02
    # 10-K, 8-K, 10-year, 2-Year; 5.2.0: never the decimals of an amount ("2.6-percent" read as 2);
    # 5.2.1: a hyphenated unit is an amount ("a 25-basis-point cut", "a 5-percent stake")
    re.compile(
        r"(?<![\d.,])\b\d{1,3}-(?!(?:basis|bps?|percent|percentage|point|pp)\b)[A-Za-z]"
    ),
    # S000002564, CIK0001033225, T10Y2Y; 5.2.x: dotted labels too ("E3.3" read as 0.3 before)
    re.compile(r"\b[A-Z]{1,5}\d+(?:\.\d+)*[A-Z0-9]*\b"),
    re.compile(r"\b(?:CERT|CIK|No\.?|#|ticker|Form)\s*\d+\b", re.IGNORECASE),
)

#: 4-1/4 -> 4.25 before anything else is read; FOMC statements write the target range this way.
_FRACTION = re.compile(r"(?<!\d)(\d+)[-‑–](\d)/(\d)\b")

#: 5.2.1: a percent or point word follows (the lookahead the fraction forms below need).
_POINT_WORD = (
    r"(?=\s*(?:-\s*)?(?:percentage[- ]points?|percent(?:age)?\b|points?\b|pp\b|%))"
)

_FRACTION_VALUES = {
    "1/2": "0.5",
    "1/4": "0.25",
    "3/4": "0.75",
    "1/3": "0.333333",
    "2/3": "0.666667",
    "1/8": "0.125",
    "3/8": "0.375",
    "5/8": "0.625",
    "7/8": "0.875",
    "½": "0.5",
    "¼": "0.25",
    "¾": "0.75",
    "⅓": "0.333333",
    "⅔": "0.666667",
    "⅛": "0.125",
    "⅜": "0.375",
    "⅝": "0.625",
    "⅞": "0.875",
}
_FRACTION_FORM = r"(1/2|1/4|3/4|1/3|2/3|1/8|3/8|5/8|7/8|[¼½¾⅓⅔⅛⅜⅝⅞])"

#: 5.2.1: a fraction of a point written as a fraction ("by 1/4 percentage point", "a ½ point
#: cut", "66⅔%", "66 2/3%") is its decimal, so the ×100 scale step relates it to basis points
#: (25 bps = 0.25 percentage point). Only before a percent or point word; never after a digit
#: and a hyphen ("4-1/4" is the mixed fraction above) and never inside a date ("1/4/2025").
_SIMPLE_FRACTION = re.compile(
    r"(?<![\w/.,])(?<!\d[-‑–])(?:(\d{1,3})(?: (?=\d/)|(?=[¼½¾⅓⅔⅛⅜⅝⅞])))?"
    + _FRACTION_FORM
    + r"(?![\d/])"
    + _POINT_WORD
)


def _fraction_decimal(match: re.Match[str]) -> str:
    whole = Decimal(match.group(1) or 0)
    return str(whole + Decimal(_FRACTION_VALUES[match.group(2)]))


#: 5.2.1: the fraction words of a point or a percent: "quarter-point", "a quarter (of a)
#: percentage point", "half-point", "half a percentage point", "one-half percent",
#: "three-quarters of a point", "one-eighth point". The word must be attached to
#: "point", "percentage point" or "percent".
_AFTER_FRACTION_WORD = r"(?=percentage[- ]points?|percent\b|points?\b)"
_FRACTION_WORDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\bthree[- ]quarters?(?: of a)?[- ]" + _AFTER_FRACTION_WORD, re.IGNORECASE
        ),
        "0.75 ",
    ),
    (
        re.compile(
            r"\b(?:a |one[- ])?quarter(?:[- ]of a)?[- ]" + _AFTER_FRACTION_WORD,
            re.IGNORECASE,
        ),
        "0.25 ",
    ),
    (
        re.compile(
            r"\b(?:a |one[- ])?half(?:[- ]a)?[- ]" + _AFTER_FRACTION_WORD, re.IGNORECASE
        ),
        "0.5 ",
    ),
    (
        re.compile(
            r"\b(?:an |one[- ])eighth(?:[- ]of a)?[- ]" + _AFTER_FRACTION_WORD,
            re.IGNORECASE,
        ),
        "0.125 ",
    ),
)

#: 5.2.1: "quarter" and "half" after a hyphen or slash, or after a word that makes them a period
#: ("quarter-over-quarter percent", "fourth quarter point estimate", "second half percent"), are
#: periods, not fractions of a point.
_PERIOD_WORD_BEFORE = re.compile(
    r"(?:[-/‑–]|\b(?:over|on|to|first|second|third|fourth|last|linked|prior|same|this"
    r"|next|previous|latest|calendar|fiscal|trailing)\s+)$",
    re.IGNORECASE,
)


#: 5.2.1: a fraction word is a move of a point only in the context of a move: a count or article
#: before it ("a quarter-point", "two quarter-point", "by half a point", "each one-eighth point"),
#: a change verb before it ("moved half a percentage point"), or a change noun after the point word
#: ("the quarter-point cut"). "the half point of the year", "the quarter percent change" and "the
#: quarter point estimate" stay words.
_FRACTION_CONTEXT_BEFORE = re.compile(
    r"\b(?:an?|one|two|three|four|five|six|another|further|additional|each|every|single|by|of"
    r"|cut|cuts|raised|lowered|reduced|increased|trimmed|eased|hiked|moved|rose|fell|dropped"
    r"|widened|narrowed)\s+$",
    re.IGNORECASE,
)
_CHANGE_NOUN_AFTER = re.compile(
    r"^(?:percentage[- ]points?|percent(?:age)?|points?)[\s)\-]+(?:rate[\s-]+)?(?:cuts?|hikes?"
    r"|increases?|decreases?|rises?|reductions?|lowerings?|lowering|moves?|steps?|easing|tightening"
    r"|raises?|drops?|declines?|trims?|bumps?|adjustments?)\b",
    re.IGNORECASE,
)


def _fraction_word(decimal: str, match: re.Match[str]) -> str:
    before = match.string[max(0, match.start() - 16) : match.start()]
    # "a" / "one" before the word makes it a fraction ("on a quarter-point cut"); only a bare
    # "quarter" or "half" can be a period.
    if re.match(r"(?:an?|one)[- ]", match.group(0), re.IGNORECASE):
        return decimal
    if _PERIOD_WORD_BEFORE.search(before):
        return match.group(0)
    after = match.string[match.end() : match.end() + 40]
    if _FRACTION_CONTEXT_BEFORE.search(before) or _CHANGE_NOUN_AFTER.match(after):
        return decimal
    return match.group(0)


#: 5.2.1: a number written in words before a unit is its figure: "four basis points",
#: "twenty-five basis points", "two percent", "one hundred fifty billion", "four and a half
#: percent". Only before percent, percentage point(s), basis point(s), bps, thousand, million,
#: billion or trillion: "three times", "Five Point" and "one of" stay words.
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_WORD_NUMBER = r"(?:" + "|".join(_NUMBER_WORDS) + r")"
_SPELLED_AMOUNT = re.compile(
    rf"\b{_WORD_NUMBER}(?:[- ](?:and[- ])?(?:{_WORD_NUMBER}|hundred))*"
    r"(?:[- ]and[- ](?:a|one|two)[- ](?:half|quarter|third|thirds))?"
    r"(?=[- ](?:percent(?:age)?\b|basis[- ]points?\b|bps\b|thousand\b|million\b|billion\b"
    r"|trillion\b))",
    re.IGNORECASE,
)
_SPELLED_FRACTIONS = {
    "a half": Decimal("0.5"),
    "one half": Decimal("0.5"),
    "a quarter": Decimal("0.25"),
    "one quarter": Decimal("0.25"),
    "a third": Decimal("0.333333"),
    "one third": Decimal("0.333333"),
    "two thirds": Decimal("0.666667"),
}


def _spelled_decimal(match: re.Match[str]) -> str:
    words = match.group(0).lower().replace("-", " ").split()
    fraction = Decimal(0)
    if len(words) >= 3 and " ".join(words[-2:]) in _SPELLED_FRACTIONS:
        fraction = _SPELLED_FRACTIONS[" ".join(words[-2:])]
        words = words[:-3]
    whole = 0
    for word in words:
        if word == "hundred":
            whole = (whole or 1) * 100
        elif word != "and":
            whole += _NUMBER_WORDS[word]
    return str(whole + fraction)


#: 5.2.1: glued forms whose amount the figure reader could not see. Each rewrite only separates
#: or renames; the value comparison is unchanged.
_GLUED_FORMS: tuple[tuple[re.Pattern[str], str], ...] = (
    # "$212mm", "$1mil" after a currency sign are millions ("300mm" alone stays a length)
    (re.compile(r"([$€£¥]\s?\d[\d,]*(?:\.\d+)?)(?:mm|mil)(?:\b|(?=[A-Z]))"), r"\1MM "),
    # "$61.2B5": a footnote digit glued to the scale letter
    (re.compile(r"([$€£¥]\s?\d[\d,]*(?:\.\d+)?[KMBT])(?=\d\b)"), r"\1 "),
    # "12,345.6Total Loans": a table value glued to the next row label (PDF extraction)
    (re.compile(r"(\d[.,]\d+)(?=[A-Z][a-z]{2,})"), r"\1 "),
    # "6.3yrs": a duration with decimals ("10yr" is a tenor and stays unread)
    (re.compile(r"(\d\.\d+)(?=yrs?\b)"), r"\1 "),
    # "CNY2.40": a currency code glued to the amount
    (re.compile(r"\b(?:USD|EUR|GBP|JPY|CNY|CAD|CHF|AUD|HKD)(?=\d)"), " "),
    # "$318.4²", "3.2%¹": a superscript footnote marker glued to the amount (a superscript
    # digit is a word character, so it hid the amount)
    (re.compile(r"(?<=[\d%])(?=[¹²³⁴⁵⁶⁷⁸⁹⁰⁽])"), " "),
    # "ê5%", "é4 bps": a symbol-font arrow glued to the amount by text extraction
    (re.compile(r"(?<!\w)[éê](?=\d)"), " "),
)

#: 5.2.1: numbers that are not figures, blanked before the passes above can split them: a
#: two-day ISO meeting date ("2025-03-18/19" left 19), an index base ("1982-84=100",
#: "2017=100" left 100), a date without a year ("3/20" read as 3 and 20), a month and year
#: ("03/2025" left 3), a rule or section number ("Rule 12b-2" read as 12 billion) and a period
#: label with a two-digit year ("Q4-25" read as 25). A
#: month/day date is never followed by a percent or point word (that is a fraction) and never
#: follows a digit and a hyphen ("4-1/4").
_NOT_A_FIGURE_FIRST: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}/\d{1,2}\b"),
    re.compile(
        r"\b(?:19|20)\d{2}(?:\s*[-–]\s*(?:(?:19|20)\d{2}|\d{2}))?\s*=\s*100\b(?![.,]\d)"
    ),
    re.compile(r"(?<![\w./])(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}\b(?![/.,-]\d)"),
    re.compile(r"(?<![\w.])\d{1,3}[a-z]\d?-\d+\b"),
    # "Q4-25", "4Q'25", "FY-24": a period label with a two-digit year ("Q4 25" is left alone: in
    # a table it may be a label and then a value)
    re.compile(r"\b(?:Q[1-4]|[1-4]Q|FY|CY|H[12]|[12]H)[-'’]\d{2}\b(?![.,]\d)"),
)

#: 5.2.1: a month/day date without a year ("on 3/20", "at 12/31 vs. 9/30") is not a figure. Only a
#: valid month and day, and only with a date word right before it ("on", "at", "as of", "ended",
#: "vs." ...) or next to another month/day date in a list or range ("9/29, 9/28"): a share
#: fraction ("1/25th interest"), a split, a mix ("70/30"), "Phase 1/2" or "24/7" stays numbers.
#: Never before a percent or point word (a fraction) and never after a digit and a hyphen.
_MONTH_DAY = re.compile(
    r"(?<![\w./$€£¥])(?<!\d[-‑–])(?<!\d )(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])\b"
    r"(?![\d/]|,\d|\s?th\b|\s?[\"”]|\s+mile)"
    r"(?!\s*(?:-\s*)?(?:percent|percentage|points?\b|pp\b|%))"
)
_DATE_WORD_BEFORE = re.compile(
    r"(?:\b(?:on|at|as of|since|through|thru|from|until|till|by|ended|ending|before|after"
    r"|vs|versus|between|dated|week of)\.?|\d{1,2}/\d{1,2}(?:,?\s*and|[,;]|\s+to|\s*[-–]))\s*$",
    re.IGNORECASE,
)
_DATE_AFTER = re.compile(
    r"^\s*(?:[-–,;]|and|to|vs\.?)\s*\d{1,2}/\d{1,2}\b", re.IGNORECASE
)


def _month_day_blank(match: re.Match[str]) -> str:
    text, start, end = match.string, match.start(), match.end()
    if _DATE_WORD_BEFORE.search(text[max(0, start - 16) : start]) or _DATE_AFTER.match(
        text[end : end + 16]
    ):
        return " "
    return match.group(0)


#: One figure: optional sign, digits with optional thousands separators, optional decimals,
#: optional scale suffix. A leading-dot decimal (".28") is a figure. The trailing lookahead
#: refuses a figure glued to a letter it did not consume, so "1Q" and "CIK123" do not yield 1
#: and 123, while a trailing period ("... $58,428,612.00.") does not break the match.
_FIGURE = re.compile(
    r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?"
    r"(?:(?:[kKmMbBtT]n?|bn|mn|tn|MM|bps?|ppt|pp|pts?|x|X)\b)?(?![\w]|\.\d|,\d)"
)
#: 5.2.0: "MM" (millions, upper case only: "300mm" is a length) and basis points glued to the
#: number ("25bp", "-5bps") are figures too; before, the glued letters hid the amount entirely.
#: 5.2.1: so are a multiple ("7.3x", "2.6X") and percentage points ("1.5pp", "-1ppt").
_SUFFIX_EXPONENT = {
    "k": 3,
    "m": 6,
    "mn": 6,
    "mm": 6,
    "b": 9,
    "bn": 9,
    "t": 12,
    "tn": 12,
    "bp": 0,
    "bps": 0,
    "pp": 0,
    "ppt": 0,
    "pt": 0,
    "pts": 0,
    "x": 0,
}

#: Scale steps a span figure may be re-based by to match a claim figure: powers of a thousand
#: (table headed "in thousands" / "in millions", or a claim written in billions), plus the two
#: steps that relate a percent to its ratio.
_SCALE_STEPS: tuple[int, ...] = tuple(range(-12, 13, 3)) + (-2, 2)

#: The longest text the check will read. A cited span can be a whole filing; the claim is capped
#: upstream. Nothing beyond this many characters is examined, in either direction.
_MAX_CHARS = 200_000


#: Disguised URLs (scorer 5.2.2), the reasoning grader's own detection: a URL is found in a FOLDED view of
#: the text, each character NFKC-normalised, the slash and colon look-alikes NFKC keeps apart read as
#: "/" and ":", and format characters (Unicode category Cf) and Hangul fillers removed.
_SLASH_LOOKALIKES = (
    "⁄",
    "∕",
    "⧸",
    "╱",
    "⟋",
    "〳",
    "᜵",
    "⳺",
    "﹨",
    "⹊",
    "̸",
)
_COLON_LOOKALIKES = ("∶", "꞉", "ː")
_LOOKALIKES: dict[str, str] = {
    **{c: "/" for c in _SLASH_LOOKALIKES},
    **{c: ":" for c in _COLON_LOOKALIKES},
}
_INVISIBLE_FILLERS = frozenset("ᅟᅠㅤﾠ")
_URL_TAIL = r"[^\s,;()\[\]{}<>\"']*"
URL_DETECTION_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9+.\-]{0,63}://"
    + _URL_TAIL
    + r"|(?<![\w/:])//[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9-]+)+"
    + _URL_TAIL
    + r"|(?<![\w.@/])[Ww]{3}\.[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*"
    + _URL_TAIL
)


def fold_for_detection(text: str) -> tuple[str, list[int] | None]:
    """(folded text, the original index of each folded character; None when the text is ASCII)."""
    if text.isascii():
        return text, None
    out: list[str] = []
    index: list[int] = []
    for position, char in enumerate(text):
        if char in _INVISIBLE_FILLERS or unicodedata.category(char) == "Cf":
            continue
        folded = _LOOKALIKES.get(char) or unicodedata.normalize("NFKC", char)
        out.append(folded)
        index.extend([position] * len(folded))
    return "".join(out), index


def url_ranges(text: str) -> list[tuple[int, int]]:
    """The [start, end) ranges, in `text`'s own offsets, of every URL found in its folded view
    (`URL_DETECTION_PATTERN`); every original character from a match's first to its last, format
    characters included, is inside a range."""
    folded, index = fold_for_detection(text)
    ranges: list[tuple[int, int]] = []
    for match in URL_DETECTION_PATTERN.finditer(folded):
        start = match.start() if index is None else index[match.start()]
        end = match.end() if index is None else index[match.end() - 1] + 1
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
            continue
        ranges.append((start, end))
    return ranges


def _blank_urls(text: str) -> str:
    """5.2.2: a web address states no amount ("?id=77", "/series/42", ":8080", an EDGAR accession
    path), disguised or not. Every URL (`url_ranges`) is blanked, equal length (so offsets and the
    span cap are unchanged), before any other pass, in claim and span alike. The verbatim-quote
    check and the judge still read the text as written."""
    ranges = url_ranges(text)
    if not ranges:
        return text
    chars = list(text)
    for start, end in ranges:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def _normalise(text: str) -> str:
    text = _blank_urls(text[:_MAX_CHARS])
    # 5.2.1: the equivalent forms first, so claim and span are read the same way.
    text = _SPELLED_AMOUNT.sub(_spelled_decimal, text)
    for pattern, decimal in _FRACTION_WORDS:
        text = pattern.sub(functools.partial(_fraction_word, decimal), text)
    text = _SIMPLE_FRACTION.sub(_fraction_decimal, text)
    for pattern, replacement in _GLUED_FORMS:
        text = pattern.sub(replacement, text)
    for pattern in _NOT_A_FIGURE_FIRST:
        text = pattern.sub(" ", text)
    text = _MONTH_DAY.sub(_month_day_blank, text)
    text = _FRACTION.sub(
        lambda m: str(int(m.group(1)) + int(m.group(2)) / int(m.group(3))), text
    )
    for pattern in _NOT_A_FIGURE:
        text = pattern.sub(" ", text)
    return text


def figures(text: str) -> tuple[Decimal, ...]:
    """Every figure in `text`, as an absolute decimal value, in order of appearance."""
    out: list[Decimal] = []
    for match in _FIGURE.finditer(_normalise(text)):
        value = _figure_value(match.group(0))
        if value is not None:
            out.append(value)
    return tuple(out)


def _figure_value(raw: str) -> Decimal | None:
    """The absolute value of one `_FIGURE` match, or None when it is not a figure."""
    token = raw.replace(",", "").lstrip("+-")
    exponent = 0
    suffix = re.search(r"[A-Za-z]+$", token)
    if suffix:
        # An unknown suffix ("12kn") is no scale, never an exception (5.1.2): a KeyError
        # here aborted a whole evaluation on one claim's prose.
        exponent = _SUFFIX_EXPONENT.get(suffix.group(0).lower(), 0)
        token = token[: suffix.start()]
    if not token or not any(ch.isdigit() for ch in token):
        return None
    try:
        value = Decimal(token).scaleb(exponent)
    except InvalidOperation:  # pragma: no cover - the regex admits only valid decimals
        return None
    return abs(value)


#: 5.2.1: words that give the figure they govern a direction ("fell 20 bps", "a 25 bps cut",
#: "rose 10 bps", "a 10 bps increase").
_FALL_VERB = (
    r"(?:declined|decreased|fell|dropped|lowered|reduced|eased|narrowed|cut|slid|slipped|lost"
    r"|shed|down|declines?|decreases?|falls?|drops?|narrows?|eases?|reduces?|lowers?)"
)
_FALL_NOUN = (
    r"(?:declines?|decreases?|drops?|falls?|cuts?|reductions?|losses|loss|narrowing|easing"
    r"|contraction)"
)
_RISE_VERB = (
    r"(?:rose|risen|increased|gained|climbed|widened|advanced|jumped|added|up|rises?"
    r"|increases?|gains?|climbs?|widens?|adds?)"
)
_RISE_NOUN = r"(?:rises?|increases?|gains?|widening|hikes?)"
_APPROX = (
    r"(?:about|around|roughly|approximately|nearly|almost|some|just|over|another|a further"
    r"|more than|less than|a total of)"
)


def _governed(verb: str, noun: str) -> re.Pattern[str]:
    """The word governs the figure FROM BEFORE it: "<verb> [<object> by] [about]" or "<noun> of
    [about]" right before it. A word after the figure never gives it a direction ("a 62.5% loss
    ratio", "$1.20 down from $1.35", "4.25% cuts ahead", "a 4.2% decline rate"), and a word that is
    merely near it ("After the cut we forecast 4.25") does not either."""
    return re.compile(
        rf"(?:\b{verb}(?:\s+(?:\w+\s+){{0,3}}?by)?|\b{noun}\s+of)(?:\s+{_APPROX})?\s*$",
        re.IGNORECASE,
    )


_FALL_BEFORE = _governed(_FALL_VERB, _FALL_NOUN)
_RISE_BEFORE = _governed(_RISE_VERB, _RISE_NOUN)
#: "±10 bps", "+/- 0.15", "plus or minus 10": a half-width, not a signed amount.
_PLUS_MINUS_BEFORE = re.compile(r"(?:±|\+/[-−]|plus or minus)\s*$", re.IGNORECASE)
#: A minus sign: ASCII hyphen, U+2212, en dash or non-breaking hyphen, also before a currency
#: sign ("EPS of -$0.12", "–$0.12"), with nothing between: a spaced dash is punctuation
#: ("guidance – $1.20"). A dash straight after a digit or "%" is a range ("3.9%-4.2%").
_MINUS_BEFORE = re.compile(r"(?<![\d%])[-−–‑][$€£¥]?$")
_PLUS_BEFORE = re.compile(r"(?<![\d%])\+\s*[$€£¥]?\s*$")
_RANGE_BEFORE = re.compile(r"[\d%]$")


def _signed_figures(text: str) -> tuple[tuple[Decimal, int, bool], ...]:
    """`figures(text)` with the DIRECTION each one is written with (5.2.1): (absolute value,
    -1 / 0 / +1, written as a half-width "±"). -1 when the figure carries a minus sign (also
    before a currency sign), is in accounting parentheses ("(25)", "(2)%"), or a fall word governs
    it from before ("fell 20 bps", "declines of about 10 bps"); +1 for a plus sign or a rise word
    that governs it from before ("rose 10 bps", "an increase of 10 bps"); 0 when the text does not
    say (a level, "we forecast 4.25", "a 25 bps cut", "a 62.5% loss ratio")."""
    out: list[tuple[Decimal, int, bool]] = []
    norm = _normalise(text)
    for match in _FIGURE.finditer(norm):
        raw = match.group(0)
        value = _figure_value(raw)
        if value is None:
            continue
        before = norm[max(0, match.start() - 80) : match.start()]
        after = norm[match.end() : match.end() + 40]
        # "+/-10": the figure took the "-" as its sign, so the "±" is read with it
        if _PLUS_MINUS_BEFORE.search(before) or (
            raw.startswith("-") and _PLUS_MINUS_BEFORE.search(before + "-")
        ):
            out.append((value, 0, True))
            continue
        words = before.rstrip(" $€£¥")
        if raw.startswith("-"):
            # "3.9%-4.2%": a dash straight after a figure is a range, not a minus
            direction = 0 if _RANGE_BEFORE.search(before) else -1
        elif raw.startswith("+") or _PLUS_BEFORE.search(before):
            direction = 1
        elif _MINUS_BEFORE.search(before):
            direction = -1
        elif words.endswith("(") and re.match(r"\s*%?\s*\)", after):
            direction = -1
        elif _FALL_BEFORE.search(words):
            direction = -1
        elif _RISE_BEFORE.search(words):
            direction = 1
        else:
            direction = 0
        out.append((value, direction, False))
    return tuple(out)


def _is_own(value: Decimal, direction: int, own: frozenset[Decimal]) -> bool:
    """`value` equals an own value re-based by one of the span matcher's scale steps (5.2.1:
    "$5.9bn" is an own value of 5.9 in billions), and its written direction does not contradict
    that value's sign ("declined about 20 bps" is not an upper bound of +20)."""
    for own_value in own:
        if direction and own_value and (direction > 0) != (own_value > 0):
            continue
        magnitude = abs(own_value)
        if any(magnitude.scaleb(step) == value for step in _SCALE_STEPS):
            return True
    return False


#: Significant digits an own value keeps: a submitted float carries binary noise in its last
#: digits (1.2999999999999998 for 1.3), which no participant writes.
_OWN_VALUE_DIGITS = 10


def _denoised(value: Decimal) -> Decimal:
    """`value` rounded to `_OWN_VALUE_DIGITS` significant digits (half-even), trailing zeros kept
    harmless because Decimal equality is numeric."""
    if not value.is_finite() or value == 0:
        return value
    return value.quantize(Decimal(1).scaleb(value.adjusted() - _OWN_VALUE_DIGITS + 1))


#: A word that says a figure next to it is the interval's level ("the 90% interval", "a 0.9 band").
_INTERVAL_WORD = re.compile(
    r"\b(?:intervals?|bands?|ranges?|confidence|credible|prediction|coverage|levels?|quantiles?)\b",
    re.IGNORECASE,
)


def _level_free(
    text: str, claim_figures: list[Decimal], levels: tuple[Decimal, ...]
) -> list[Decimal]:
    """`claim_figures` without the mentions of the unit's interval level (5.2.1): the level as a
    fraction or a percent (0.9, 90) with an interval word within 40 characters. Only those two
    forms and only there, so "grew 90%" or "$900M" on a 0.9-level unit stay figures."""
    forms = {form for level in levels for form in (level, level.scaleb(2))}
    if not forms or not claim_figures:
        return claim_figures
    norm = _normalise(text)
    out = list(claim_figures)
    for match in _FIGURE.finditer(norm):
        value = _figure_value(match.group(0))
        if value is None or value not in forms or value not in out:
            continue
        before = norm[max(0, match.start() - 20) : match.start()]
        if _PLUS_MINUS_BEFORE.search(before) or (
            match.group(0).startswith("-") and _PLUS_MINUS_BEFORE.search(before + "-")
        ):
            continue  # "±0.9" is a half-width, never the level
        near = norm[max(0, match.start() - 40) : match.end() + 40]
        if _INTERVAL_WORD.search(near):
            out.remove(value)
    return out


def _own_value_free(
    text: str, own: frozenset[Decimal], half_widths: frozenset[Decimal]
) -> list[Decimal]:
    """The figures of `text` that are not the participant's own scored values (5.2.1, `_is_own`),
    and not a half-width ("±10 bps") equal to half the own scored interval's width."""
    return [
        value
        for value, direction, plus_minus in _signed_figures(text)
        if not (
            _is_own(value, 0, half_widths)
            if plus_minus
            else _is_own(value, direction, own)
        )
    ]


def _places(value: Decimal) -> int:
    exponent = value.as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


def _same_figure(claim_value: Decimal, candidate: Decimal) -> bool:
    """`candidate`, re-based by any tolerated scale step and rounded to `claim_value`'s own
    precision, equals `claim_value`."""
    quantum = Decimal(1).scaleb(-_places(claim_value))
    for step in _SCALE_STEPS:
        try:
            scaled = candidate.scaleb(step).quantize(quantum, rounding=ROUND_HALF_UP)
        except InvalidOperation:  # pragma: no cover - only for absurd magnitudes
            continue
        if scaled == claim_value:
            return True
    return False


def missing_figures(
    claim_text: str,
    span_texts: Iterable[str],
    *,
    submitted: Sequence[float | int | None] = (),
    names: Iterable[str] = (),
    span_cap: int | None = None,
    own_intervals: Sequence[tuple[float | int | None, float | int | None]] = (),
    own_levels: Sequence[float | None] = (),
) -> tuple[tuple[Decimal, ...], int]:
    """The claim's figures that no passage in `span_texts` carries, and how many figures the claim
    states in all (both after the exact own-value exemption `claim_number_status` applies).

    `names`: the unit's own entity names and tickers; a number written as part of one of
    them ("Phillips 66", "S&P 500", "3M") is not a figure (`name_fragments`). `span_cap`:
    a cited span longer than `span_cap` characters anchors no figure (`_readable`), so a
    whole-document citation cannot anchor a figure by coincidence somewhere in a long filing: a
    claim with figures cites the passage that states them. The verbatim-quote tests still read
    the whole span (a verbatim quote cannot misstate a figure). From 5.2.2 the claim check
    (`scoring.evaluate_claims`) makes a claim citing a span over the cap false before it gets
    here.

    5.2.1: a claim figure is exempt when it equals one of `submitted` re-based by a scale step
    of the span matcher ("$5.9bn" for 5.9), unless the direction the claim writes contradicts
    that value's sign (`_signed_figures`, `_is_own`): "two-year yields declined about 20 bps"
    is not the upper bound +20. `own_intervals`: the participant's own SCORED intervals as
    (lo, hi); a figure written as a half-width ("±10 bps") is exempt when it equals
    (hi - lo) / 2 of one of them. `own_levels`: the unit's interval level (0.9); a claim that
    names it next to an interval word ("the 90% interval", "a 0.9 band") states the task's
    parameter, not evidence (`_level_free`). Every own
    value is read without float noise (`_denoised`: 1.2999999999999998 is 1.3), but a written
    figure must still equal it exactly at a scale step (5.87 is not "5.9"). The comparison with the
    spans stays sign-blind, as above."""
    spans = list(span_texts)
    own = frozenset(_denoised(Decimal(str(v))) for v in submitted if v is not None)
    levels = tuple(Decimal(str(v)) for v in own_levels if v is not None)
    half_widths = frozenset(
        _denoised((Decimal(str(hi)) - Decimal(str(lo))) / 2)
        for lo, hi in own_intervals
        if lo is not None and hi is not None
    )
    fragments = name_fragments(tuple(names))
    read = _blank_names(claim_text, fragments) if fragments else claim_text
    claim_figures = _level_free(read, _own_value_free(read, own, half_widths), levels)
    if not claim_figures:
        return (), 0
    stated_all = len(claim_figures)
    if _verbatim(claim_text, spans):
        # A verbatim quote of a cited span states exactly what the span states: it cannot misstate
        # a figure, and a number cut at the quote's edges ("...\n20" of "2025-02-03") is not an
        # amount the claim asserts. Whitespace runs are compared as one space.
        return (), len(claim_figures)
    label = _quoted_label(claim_text, spans)
    if label is not None:
        # 5.2.0: "<label>: <verbatim quote>" ("Evidence for Synthetic Issuer A: ... $ 1,8") is a
        # label and a quote: the quote is read as a verbatim quote (its cut edges are not figures);
        # every figure of the label must still be carried like any other claim figure.
        read_label = _blank_names(label, fragments) if fragments else label
        claim_figures = _level_free(
            read_label, _own_value_free(read_label, own, half_widths), levels
        )
        if not claim_figures:
            return (), stated_all
    pool: set[Decimal] = set()
    for span in spans:
        pool.update(_figures_cached(_readable(span, span_cap)))
    ordered = sorted(pool)
    missing = tuple(v for v in claim_figures if not _carried(v, ordered))
    return missing, stated_all


_WS = re.compile(r"\s+")


#: The shortest quote the label form accepts: a shorter tail ("x: 12") is a claim, not a quote.
_MIN_QUOTE_CHARS = 20


def _quoted_label(claim_text: str, span_texts: Sequence[str]) -> str | None:
    """The label of a "<label>: <verbatim quote>" claim, or None. The quote is everything after the
    FIRST ": " and must be a verbatim piece (whitespace runs as one space) of a cited span, at
    least `_MIN_QUOTE_CHARS` long."""
    label, sep, quote = claim_text.partition(": ")
    if not sep or len(quote.strip()) < _MIN_QUOTE_CHARS:
        return None
    return label if _verbatim(quote, span_texts) else None


def _readable(span: str, cap: int | None) -> str:
    """The text of a cited span the figure check reads: the whole span when it is at most `cap`
    characters (or `cap` is None), nothing when it is longer. A span longer than the cap anchors
    no figure; the verbatim-quote tests read it whole."""
    return span if cap is None or len(span) <= cap else ""


@functools.lru_cache(maxsize=256)
def name_fragments(names: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    """The pieces of the unit's entity names/tickers whose number is part of the name.

    For each whitespace token of a name that carries a digit: the token itself when it also
    carries a letter ("3M", "T10Y2Y"), and the token joined to each neighbouring name word
    ("S&P 500", "500 futures", "Phillips 66"). A bare number is never a fragment on its own, so a
    claim's "500" or "66" standing alone is still a figure; "S&P 510" or "Phillips 67" are not the
    name and stay figures. Matching is case-sensitive, reads any whitespace run as one space, and needs
    a boundary on both sides: no word character before, and no word character or decimal part
    after ("S&P 5000", "Phillips 66.5" are figures)."""
    pieces: set[tuple[str, ...]] = set()
    for name in names:
        tokens = name.split()
        for k, token in enumerate(tokens):
            if not any(ch.isdigit() for ch in token):
                continue
            if any(ch.isalpha() for ch in token):
                pieces.add((token,))
            if k > 0:
                pieces.add((tokens[k - 1], token))
            if k + 1 < len(tokens):
                pieces.add((token, tokens[k + 1]))
    patterns = []
    for piece in sorted(pieces, key=lambda p: (-sum(map(len, p)), p)):
        body = r"\s+".join(re.escape(t) for t in piece)
        patterns.append(re.compile(rf"(?<!\w){body}(?!\w|[.,]\d)"))
    return tuple(patterns)


def _blank_names(text: str, fragments: Sequence[re.Pattern[str]]) -> str:
    for pattern in fragments:
        text = pattern.sub(" ", text)
    return text


def verbatim_quote(claim_text: str, span_texts: Iterable[str]) -> bool:
    """Is the claim a word-for-word piece of one of the spans (whitespace runs as one space)?"""
    return _verbatim(claim_text, span_texts)


def _verbatim(claim_text: str, span_texts: Iterable[str]) -> bool:
    claim = _WS.sub(" ", claim_text).strip()
    return bool(claim) and any(
        claim in _WS.sub(" ", span[:_MAX_CHARS]) for span in span_texts
    )


@functools.lru_cache(maxsize=512)
def _figures_cached(text: str) -> tuple[Decimal, ...]:
    """`figures`, memoised: a whole cited span (up to `_MAX_CHARS`) is often cited by many claims."""
    return figures(text)


def _carried(value: Decimal, ordered: list[Decimal]) -> bool:
    """`any(_same_figure(value, c) for c in ordered)`, without the linear scan.

    `_same_figure` holds exactly when some scale step s puts `c * 10**s` in the half-open interval
    that rounds (ROUND_HALF_UP, to `value`'s precision) to `value`: [value - q/2, value + q/2) for
    value > 0, [0, q/2) for 0. So each step bisects `ordered` for that interval, and every hit is
    confirmed with `_same_figure` itself, which keeps the verdict the reference function's."""
    quantum = Decimal(1).scaleb(-_places(value))
    half = quantum / 2
    lo_v, hi_v = (max(value - half, Decimal(0)), value + half)
    for step in _SCALE_STEPS:
        lo, hi = lo_v.scaleb(-step), hi_v.scaleb(-step)
        i = bisect.bisect_left(ordered, lo)
        while i < len(ordered) and ordered[i] < hi:
            if _same_figure(value, ordered[i]):
                return True
            i += 1
    return False


def claim_number_status(
    claim_text: str,
    span_texts: Iterable[str],
    *,
    submitted: Sequence[float | int | None] = (),
    rule: FigureRule = "any",
    names: Iterable[str] = (),
    span_cap: int | None = None,
    own_intervals: Sequence[tuple[float | int | None, float | int | None]] = (),
    own_levels: Sequence[float | None] = (),
) -> ClaimNumberStatus:
    """Classify one claim against the passages it cites.

    ``no_figures``  the claim states no amount (after the exclusions above and after
        removing any figure exactly equal to one of `submitted`, the participant's own scored
        values); nothing to anchor, the judge decides.
    ``anchored``    rule "any": at least one figure in the claim appears in at least one cited
        passage. Rule "every": every figure in the claim appears in at least one cited passage.
    ``unanchored``  rule "any": the claim carries figures and none of them appears in any cited
        passage. Rule "every": at least one of the claim's figures appears in no cited passage.

    Rule "every" reads the own values with their signs and exempts a "±" half-width of an
    interval in `own_intervals` (`missing_figures`); rule "any" is unchanged.
    """
    if rule not in FIGURE_RULES:
        raise ValueError(f"figure rule {rule!r} is not one of {list(FIGURE_RULES)}")
    if rule == "every":
        missing, stated = missing_figures(
            claim_text,
            span_texts,
            submitted=submitted,
            names=names,
            span_cap=span_cap,
            own_intervals=own_intervals,
            own_levels=own_levels,
        )
        if not stated:
            return "no_figures"
        return "unanchored" if missing else "anchored"
    # Exact equality only (5.1.3): the span tolerances (scale steps, rounding to the claim's
    # precision) exist to ADMIT a figure the passage carries; applied here they would widen what a
    # participant's own value can hide. Decimal equality is numeric, so 5.9 == 5.90.
    own = {abs(Decimal(str(v))) for v in submitted if v is not None}
    claim_figures = [f for f in figures(claim_text) if f not in own]
    if not claim_figures:
        return "no_figures"
    span_figures: list[Decimal] = []
    for span in span_texts:
        span_figures.extend(figures(span))
    for value in claim_figures:
        if any(_same_figure(value, candidate) for candidate in span_figures):
            return "anchored"
    return "unanchored"
