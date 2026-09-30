#!/usr/bin/env python3
"""Scan every corpus document of the given units for number forms the figure matcher misreads.

The claim check (`qfbench2_track_analysis.numeric`) compares the figures of a claim with the
figures of the passage it cites. A number the matcher cannot read, or reads as the wrong amount,
penalises an honest claim: the passage says "by 1/4 percentage point" and the claim "25 bps", or
the claim copies a date that the matcher reads as an amount. This script finds every numeric
expression of the known risky kinds in every corpus document and in the task table, reads each
one with the scorer's own `figures`, and compares the reading with what the expression means:

    fraction       "1/4 percentage point", "¼ point", "66⅔%", "4-1/4 percent"
    fraction_word  "quarter-point", "half a percentage point", "one-eighth point"
    spelled        "four basis points", "twenty-five basis points", "two percent"
    index_base     "(1982-84=100)", "2017=100"
    date           "3/20", "12/31/2025", "03/2025", "2025-03-18/19", "4Q24", "Mar-25", "8:30"
    range          "4.25-4.50", "3.5%–3.75%", "4 to 4-1/4"
    footnote       "12.5%1", "3.2%¹", "($417)2", "10.44*"
    glued          "7.3x", "$212mm", "25bp", "1.5pp", "$61.2B5", "12,345.6Total"

and, for completeness, every other token with a digit that the matcher reads nothing from.

Each finding is one of: ``covered`` (the matcher reads what the expression means), ``left_alone``
(a deliberate choice; the reason is printed with the count) or ``gap`` (the matcher misreads it and
no rule covers it). Any gap, and any unread token no class explains, makes the exit status 1, so a
new unit with an unreadable form fails here instead of silently penalising participants.

    python scripts/scan_number_forms.py units/                 # every unit under units/
    python scripts/scan_number_forms.py units/t4-EXAMPLE-eps-beat --json
    python scripts/scan_number_forms.py <root> --show gap      # print contexts (organiser only)

Corpus text of a held-out unit is private: the default output is counts only. ``--show`` prints
the text around each finding of the named verdict and is for organisers' own terminals.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from decimal import Decimal

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qfbench2_track_analysis.corpus import task_table_text  # noqa: E402
from qfbench2_track_analysis.numeric import figures  # noqa: E402

_POINT = r"(?:\s*-?\s*(?:percentage[- ]points?|percent(?:age)?\b|points?\b|pp\b|%))"
_UNICODE = {
    "½": "1/2",
    "¼": "1/4",
    "¾": "3/4",
    "⅓": "1/3",
    "⅔": "2/3",
    "⅛": "1/8",
    "⅜": "3/8",
    "⅝": "5/8",
    "⅞": "7/8",
}
_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_WORD_VALUE = {**{w: i for i, w in enumerate(_ONES)}, **_TENS}
_NUMBER_WORD = "(?:" + "|".join(sorted(_WORD_VALUE, key=len, reverse=True)) + ")"
_SPELLED_UNITS = (
    r"percent(?:age)?(?:[- ]points?)?|basis[- ]points?|bps|thousand|million"
    r"|billion|trillion"
)
_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November"
    r"|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\b\.?"
)
_SCALE = {"k": 3, "m": 6, "mn": 6, "mm": 6, "b": 9, "bn": 9, "t": 12, "tn": 12}


def _dec(text: str) -> Decimal:
    return Decimal(text.replace(",", ""))


def _fraction(text: str) -> Decimal:
    num, den = text.split("/")
    return Decimal(num) / Decimal(den)


def _reads(read: tuple[Decimal, ...], value: Decimal, places: int = 6) -> bool:
    """Is `value` among the readings, compared at `places` decimals (thirds are 0.333333)?"""
    q = Decimal(1).scaleb(-places)
    return any(abs(r - value) < q for r in read)


@dataclass
class Finding:
    family: str
    kind: str
    verdict: str  # covered | left_alone | gap
    reason: str = ""


@dataclass
class Report:
    documents: int = 0
    units: int = 0
    counts: collections.Counter[tuple[str, str, str, str]] = field(
        default_factory=collections.Counter
    )
    shown: list[str] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.counts[
            (finding.family, finding.kind, finding.verdict, finding.reason)
        ] += 1

    @property
    def failures(self) -> int:
        return sum(
            n for (_, _, verdict, _), n in self.counts.items() if verdict == "gap"
        )


# -- the families --------------------------------------------------------------------------


def _check_fraction(m: re.Match[str]) -> Finding:
    raw = m.group(0)
    whole, frac = m.group("whole"), m.group("frac")
    frac = _UNICODE.get(frac, frac)
    value = _fraction(frac) + (Decimal(whole) if whole else 0)
    num, den = (int(x) for x in frac.split("/"))
    kind = "mixed" if whole else "simple"
    if den not in (2, 3, 4, 8) or num >= den:
        return Finding("fraction", kind, "gap", f"denominator {den}")
    ok = _reads(figures(raw), value)
    return Finding("fraction", kind, "covered" if ok else "gap")


_PERIOD_BEFORE = re.compile(
    r"(?:[-/‑–]|\b(?:over|on|to|first|second|third|fourth|last|linked|prior|same|this|next"
    r"|previous|latest|calendar|fiscal|trailing)\s+)$",
    re.IGNORECASE,
)


#: This scan's own test of "a move of a point" beside a fraction word (independent wording of
#: the scorer's rule): a count or change verb before it, or a change noun after it.
_MOVE_CONTEXT_BEFORE = re.compile(
    r"\b(?:\w+ed|by|of|a|an|one|two|three|four|another|further|each|every|single|cuts?)\s+$",
    re.IGNORECASE,
)
_MOVE_NOUN_AFTER = re.compile(
    r"\b(?:cuts?|hikes?|increases?|decreases?|rises?|reductions?|lowering|moves?|steps?|easing"
    r"|tightening|raises?|drops?|declines?|trims?|bumps?|adjustments?)\b",
    re.IGNORECASE,
)


def _check_fraction_word(m: re.Match[str]) -> Finding:
    word = m.group("word").lower().replace("-", " ")
    value = {
        "quarter": Decimal("0.25"),
        "half": Decimal("0.5"),
        "eighth": Decimal("0.125"),
        "three quarters": Decimal("0.75"),
        "three quarter": Decimal("0.75"),
        "third": Decimal(1) / 3,
        "two thirds": Decimal(2) / 3,
    }[word]
    kind = word.replace(" ", "_")
    before = m.string[max(0, m.start() - 12) : m.start()]
    if _PERIOD_BEFORE.search(before):
        # "quarter-over-quarter percent", "fourth quarter point": a period, not a fraction
        bad = _reads(figures(m.string[max(0, m.start() - 12) : m.end()]), value)
        return Finding("fraction_word", "period_word", "gap" if bad else "covered")
    window = m.string[max(0, m.start() - 16) : m.end() + 40]
    if re.search(r"\band\s+$", before):
        # "four and a half percent": part of a number in words, read by the spelled family
        return Finding(
            "fraction_word", "part_of_spelled", "covered" if figures(window) else "gap"
        )
    ok = _reads(figures(window), value)
    moved = (
        _MOVE_CONTEXT_BEFORE.search(before)
        or _MOVE_NOUN_AFTER.search(m.string[m.end() : m.end() + 24])
        or re.match(r"(?:an?|one)[- ]", m.group(0), re.IGNORECASE)
    )
    if not moved:
        # "the half point of the year": not a move of a point, so it must read as nothing
        return Finding(
            "fraction_word",
            "no_move_context",
            "gap" if ok else "covered",
            "a fraction word with no count, change verb or change noun beside it is not a move",
        )
    if not ok and word in ("third", "two thirds"):
        return Finding(
            "fraction_word",
            kind,
            "left_alone",
            "a third of a point is not a quantity filings or FOMC texts state",
        )
    return Finding("fraction_word", kind, "covered" if ok else "gap")


def _spelled_value(words: str) -> Decimal:
    tokens = words.lower().replace("-", " ").split()
    frac = Decimal(0)
    if len(tokens) >= 3 and tokens[-3] == "and":
        frac = {
            "half": Decimal("0.5"),
            "quarter": Decimal("0.25"),
            "third": Decimal(1) / 3,
            "thirds": Decimal(2) / 3,
        }[tokens[-1]]
        tokens = tokens[:-3]
    whole = 0
    for tok in tokens:
        if tok == "hundred":
            whole = (whole or 1) * 100
        elif tok != "and":
            whole += _WORD_VALUE[tok]
    return whole + frac


def _check_spelled(m: re.Match[str]) -> Finding:
    unit = m.group("unit").lower()
    kind = (
        "percent"
        if unit.startswith("percent") and "point" not in unit
        else "point"
        if "point" in unit or unit == "bps"
        else "scale"
    )
    ok = _reads(figures(m.group(0)), _spelled_value(m.group("num")))
    return Finding("spelled", kind, "covered" if ok else "gap")


def _check_index_base(m: re.Match[str]) -> Finding:
    ok = not figures(m.group(0))
    return Finding("index_base", "year_base", "covered" if ok else "gap")


_DATE_KINDS: tuple[tuple[str, re.Pattern[str], str | None], ...] = (
    # (kind, pattern, reason it is left alone -- None when it must read as no figure)
    ("iso", re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}(?:/\d{1,2})?\b"), None),
    ("m_d_y", re.compile(r"(?<![\w./])\d{1,2}/\d{1,2}/\d{2,4}\b"), None),
    (
        "m_yyyy",
        re.compile(r"(?<![\w./])(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}\b(?![/.,]\d)"),
        None,
    ),
    (
        "month_day",
        re.compile(
            rf"\b{_MONTHS}\s+\d{{1,2}}(?:\s*[-–]\s*\d{{1,2}})?(?:,?\s+(?:19|20)\d{{2}})?\b"
        ),
        None,
    ),
    (
        "day_month",
        re.compile(rf"(?<![\d.,])\b\d{{1,2}}\s+{_MONTHS}(?:\s+(?:19|20)\d{{2}})?\b"),
        None,
    ),
    (
        "period_label",
        re.compile(
            r"\b(?:[1-4]Q|Q[1-4]|[12]H|H[12]|FY|CY)[-'’]?(?:(?:19|20)?\d{2})\b"
            r"(?![.,]\d)|\b(?:19|20)\d{2}\s?Q[1-4]\b"
        ),
        None,
    ),
    (
        "period_label_spaced",
        re.compile(r"\b(?:[1-4]Q|Q[1-4]|[12]H|H[12])\s\d{2}\b(?![.,]\d|\s*%)"),
        'a period label, a space and two digits ("Q4 25") is read as 25: in a table the two digits'
        " may be the value in the next column, so they are not blanked",
    ),
    (
        "month_yy",
        re.compile(rf"\b{_MONTHS}[-' ]'?\d{{2}}\b(?![\d,.:])(?!\s*(?:,|\d))"),
        'a two-digit year after a month ("Mar-25") is read as 25; "Mar 25" is a day, and the'
        " two cannot be told apart without the calendar; a claim that copies it finds the same"
        " reading in the span it copied from",
    ),
    (
        "apostrophe_yy",
        re.compile(r"(?<![\w])['’]\d{2}\b(?![\d.,%])"),
        'an apostrophe year ("\'25") reads as 25 in claim and span alike; claims write years in'
        " full",
    ),
    (
        "compact_ymd",
        re.compile(
            r"(?<![\w.,])(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\b(?![.,]\d)"
        ),
        'an eight-digit date ("20250917") is one token an amount could also be; it reads as the'
        " same number in claim and span",
    ),
    (
        "clock_time",
        re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\b(?![.,]\d)"),
        'a clock time ("8:30") reads as 8 and 30 in claim and span alike; it is left to the'
        " claim to cite the passage that gives the time",
    ),
)


def _check_date(kind: str, m: re.Match[str], reason: str | None) -> Finding:
    read = figures(m.group(0))
    if not read:
        return Finding("date", kind, "covered")
    if reason is not None:
        return Finding("date", kind, "left_alone", reason)
    return Finding("date", kind, "gap")


#: A month/day candidate: two small numbers and a slash, not before a percent or point word.
_MONTH_DAY_RE = re.compile(
    r"(?<![\w./-])(\d{1,2})/(\d{1,2})\b(?![\d/])(?!" + _POINT + ")"
)
#: This scan's own test of "not a date", independent of the scorer's rule: an impossible month or
#: day, or a share, split, mix, trial-phase, size or fraction context.
_NOT_DATE_AFTER = re.compile(
    r"^(?:\s?(?:th|st|nd|rd)\b|,\d{3}|\s*(?:interest|split|shares?\b|par\b|mile|of\b|basis|["
    r"”\"]|mix|\(|access|warrants?\b|monitoring|support|provision|retirement|plan|breakpoints|rules?\b|trials?\b))",
    re.IGNORECASE,
)
_NOT_DATE_BEFORE = re.compile(
    r"(?:\b(?:phase|block|mix|humulin|humalog|weighting of|approximately a|solar|u\.s\. white)"
    r"|\$\s?\d[\d.,]*|\d)\s*$",
    re.IGNORECASE,
)


def _figures_multiset(text: str) -> list[Decimal]:
    return sorted(figures(text))


def _check_month_day(m: re.Match[str]) -> Finding:
    text, start, end = m.string, m.start(), m.end()
    window = text[max(0, start - 40) : end + 40]
    blanked_window = (
        text[max(0, start - 40) : start] + " " * (end - start) + text[end : end + 40]
    )
    blanked = _figures_multiset(window) == _figures_multiset(blanked_window)
    month, day = int(m.group(1)), int(m.group(2))
    dated = bool(re.search(r"\b(?:on|at|as of)\s*$", text[max(0, start - 8) : start]))
    after = text[end : end + 16]
    not_date = (
        not (1 <= month <= 12 and 1 <= day <= 31)
        or (
            bool(_NOT_DATE_AFTER.match(after))
            and not (dated and re.match(r"\s*of\b", after))
        )
        or bool(_NOT_DATE_BEFORE.search(text[max(0, start - 20) : start]))
    )
    if not_date:
        if blanked:
            return Finding("date", "m_d_not_a_date", "gap", "a non-date was blanked")
        return Finding("date", "m_d_not_a_date", "covered")
    if blanked:
        return Finding("date", "m_d", "covered")
    return Finding(
        "date",
        "m_d_no_date_word",
        "left_alone",
        'a month/day with no date word or neighbouring date ("the 5/6 memo") is read as two'
        " numbers in claim and span alike; blanking every small fraction would hide shares,"
        " splits and mixes",
    )


def _year(text: str) -> bool:
    return bool(re.fullmatch(r"(?:19|20)\d{2}", text))


def _check_range(m: re.Match[str]) -> Finding:
    lo, hi = m.group("lo"), m.group("hi")
    read = figures(m.group(0))
    if _year(lo) and (_year(hi) or len(hi) == 2):
        return Finding("range", "years", "covered" if not read else "gap")
    ok = _reads(read, _dec(lo)) and _reads(read, _dec(hi))
    return Finding("range", "amounts", "covered" if ok else "gap")


def _check_footnote(m: re.Match[str]) -> Finding:
    amount = m.group("amount")
    read = figures(m.group(0))
    ok = _reads(read, abs(_dec(amount)))
    return Finding("footnote", m.lastgroup or "marker", "covered" if ok else "gap")


_GLUED_READ = {"bp": 0, "bps": 0, "pp": 0, "ppt": 0, "x": 0, **_SCALE}
_GLUED_LEFT = {
    "ordinal": ("st", "nd", "rd", "th"),
    "id": (),
}


def _check_glued(m: re.Match[str]) -> Finding:
    cur, number, suffix = m.group("cur"), m.group("number"), m.group("suffix")
    low = suffix.lower()
    read = figures(m.group(0))
    if low in ("mm", "mil") and not cur:
        return Finding(
            "glued",
            "mm_length",
            "left_alone",
            '"300mm" without a currency sign is a length',
        )
    if low in ("mm", "mil"):
        return Finding(
            "glued",
            "currency_mm",
            "covered" if _reads(read, _dec(number).scaleb(6), 0) else "gap",
        )
    if suffix in ("MM",) or (
        low in _GLUED_READ and suffix not in ("X",) or suffix == "X"
    ):
        exp = _GLUED_READ.get(low, 0)
        if (
            suffix == "m"
            and not cur
            and "." not in number
            and len(number.replace(",", "")) <= 2
        ):
            # "3m", "12m": a tenor or a month count as often as a million; either reading is
            # a number the span states, so an extra figure only admits more
            return Finding(
                "glued",
                "scale_m_short",
                "left_alone",
                '"3m" is a tenor as often as 3 million; both are read as 3 million'
                " in claim and span alike",
            )
        ok = _reads(read, _dec(number).scaleb(exp), 0)
        return Finding("glued", f"suffix_{low}", "covered" if ok else "gap")
    if low in ("st", "nd", "rd", "th"):
        if m.group(0).startswith(number) and ("," in number or "." in number):
            return Finding(
                "glued",
                "ordinal_of_amount",
                "left_alone",
                '"1/4,096th" and "12.75th percentile" state a share, not an ordinal;'
                " rare, and read as part of the amount in claim and span alike",
            )
        return Finding("glued", "ordinal", "covered" if not read else "gap")
    if re.fullmatch(r"yrs?", low):
        if "." in number:
            return Finding("glued", "duration", "covered" if read else "gap")
        return Finding(
            "glued",
            "tenor",
            "left_alone",
            '"10yr" names a tenor, like "10-year", and is not an amount',
        )
    if re.fullmatch(r"[A-Z][a-z]{2,}\w*", suffix) and ("." in number or "," in number):
        ok = _reads(read, _dec(number), 0)
        return Finding("glued", "value_then_label", "covered" if ok else "gap")
    return Finding(
        "glued",
        "label",
        "left_alone",
        "a number glued to letters is a label, an id or a technical unit"
        ' ("1A", "9Z", "12a", "3nm", "20mg", "4Q24", "7c")',
    )


# -- the finder ----------------------------------------------------------------------------

_FRACTION_RE = re.compile(
    r"(?<![\w/.,])(?:(?P<whole>\d{1,3})(?:[- ](?=\d/)|(?=[½¼¾⅓⅔⅛⅜⅝⅞])))?"
    r"(?P<frac>\d{1,2}/\d{1,2}|[½¼¾⅓⅔⅛⅜⅝⅞])(?![\d/])" + _POINT
)
_FRACTION_WORD_RE = re.compile(
    r"\b(?:(?:a|an|one)[- ])?(?P<word>three[- ]quarters?|two[- ]thirds|quarter|half|third|eighth)"
    r"(?:[- ](?:of[- ])?an?)?[- ](?:percentage[- ]points?|percent\b|points?\b)",
    re.IGNORECASE,
)
_SPELLED_RE = re.compile(
    rf"\b(?P<num>{_NUMBER_WORD}(?:[- ](?:and[- ])?(?:{_NUMBER_WORD}|hundred))*"
    r"(?:[- ]and[- ](?:a|one|two)[- ](?:half|quarter|third|thirds))?)"
    rf"[- ](?P<unit>{_SPELLED_UNITS})\b",
    re.IGNORECASE,
)
_INDEX_BASE_RE = re.compile(
    r"\b(?:19|20)\d{2}(?:\s*[-–]\s*(?:(?:19|20)\d{2}|\d{2}))?\s*=\s*100\b(?![.,]\d)"
)
_OTHER_BASE_RE = re.compile(r"=\s*100\b(?![.,]\d)")
_RANGE_RE = re.compile(
    r"(?<![\w.,/-])(?P<lo>\d{1,3}(?:,\d{3})*(?:\.\d+)?)%?\s?(?:[-–—]|\s+to\s+)\s?"
    r"(?P<hi>\d{1,3}(?:,\d{3})*(?:\.\d+)?)%?(?![\w.,/-])(?!\s*" + "=" + r")"
)
_FOOTNOTE_RE = re.compile(
    r"(?<![\w.,])(?P<amount>\d{1,3}(?:,\d{3})*\.\d+)(?:"
    r"(?P<percent_digit>%\d\b)|(?P<superscript>%?[¹²³⁴⁵⁶⁷⁸⁹⁰⁽⁾]+)|(?P<star>%?\*+))"
)
_GLUED_RE = re.compile(
    r"(?<![\w.,])(?P<cur>[$€£¥])?(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suffix>[A-Za-z]+)\b"
)
_TOKEN_RE = re.compile(r"\S*\d\S*")

#: Every other token that carries a digit and reads as nothing: the classes that explain it,
#: tried in order on the token with its superscript footnote markers removed.
_SUPERSCRIPTS = str.maketrans("", "", "¹²³⁴⁵⁶⁷⁸⁹⁰⁽⁾")
_UNREAD_CLASSES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("index_base", re.compile(r"=\s*100\b")),
    (
        "year_or_period",
        re.compile(
            r"(?i)^\W*(?:FY|CY|Q[1-4]|[1-4]Q|H[12]|[12]H)?[-'’]?(?:19|20)?\d{2,4}(?:[-/–]\d{2,4})*"
            r"(?:Q[1-4]|[1-4]Q|[AEFY]|s|['’]s)?(?:[,;]\s*(?:[1-4]Q|Q[1-4])\d{2})*\W*$"
        ),
    ),
    ("year_then_word", re.compile(r"^\W*(?:19|20)\d{2}\W+[A-Za-z(]")),
    (
        "date",
        re.compile(
            r"^\W*(?:\d{1,4}[-/.]){1,2}\d{1,4}\W*(?:->|\.\.|–|-|to)?\W*"
            r"(?:(?:\d{1,4}[-/.]){0,2}\d{1,4})?\W*$"
        ),
    ),
    ("identifier", re.compile(r"^\W*[^\W\d_]")),
    ("period_with_fraction", re.compile(r"^\d+[½¼¾⅓⅔⅛⅜⅝⅞]$")),
    ("number_then_word", re.compile(r"^\W*[\d.,%$]+[A-Za-z]")),
    ("id_with_punctuation", re.compile(r"^\W*\w*\d\w*[-‑–./:#()_—+]")),
    ("marker_only", re.compile(r"^\W*\d{1,2}\W*$")),
    ("hex_colour", re.compile(r"^#[0-9A-Fa-f]{3,8}\W*$")),
    ("path_or_url", re.compile(r"[/_]|https?:|\.htm")),
    ("glued_values", re.compile(r"^\W*[$\d.,%()\-]+\W*$")),
)
_UNREAD_REASONS = {
    "index_base": 'an index base ("2017=100") is not an amount',
    "year_or_period": "a year, fiscal period or quarter label is not an amount",
    "year_then_word": 'a year glued to the next word ("2027—filed")',
    "date": "a date or date range is not an amount",
    "period_with_fraction": 'a mixed fraction not before a percent or point word ("6½ years")'
    ' is a count of periods, like "13 weeks"',
    "identifier": "letters then digits: a series id, a footnoted word, a label, or text whose"
    " font the extraction garbled (a run of shifted capitals and digits)",
    "number_then_word": "a footnote marker, an id or several table cells glued to a word"
    ' ("2Notes", "9Z"); a readable amount glued to a label is the'
    " glued/value_then_label class",
    "id_with_punctuation": "an exhibit, rule, form or section number",
    "marker_only": "a footnote or list marker",
    "hex_colour": "a colour code",
    "path_or_url": "a file name, field name or URL",
    "glued_values": "table cells glued with no separator by text extraction; no rule can"
    " recover the cell boundaries",
}


def _unread_class(token: str) -> str | None:
    bare = token.translate(_SUPERSCRIPTS)
    if not any(ch.isdigit() for ch in bare):
        return "marker_only"
    for name, pattern in _UNREAD_CLASSES:
        if pattern.search(bare):
            return name
    return None


def _texts(unit_dir: pathlib.Path) -> Iterator[tuple[str, str]]:
    for path in sorted((unit_dir / "corpus").glob("*.json")):
        if path.name == "manifest.json":
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(document, dict):
            yield path.stem, str(document.get("text", ""))
    task = unit_dir / "task.json"
    if task.is_file():
        yield "task", task_table_text(json.loads(task.read_text(encoding="utf-8")))[0]


def scan_text(
    text: str, report: Report, *, where: str = "", show: str | None = None
) -> None:
    """Add every finding in `text` to `report`."""

    def note(finding: Finding, m: re.Match[str]) -> None:
        report.add(finding)
        if show is not None and finding.verdict == show:
            ctx = text[max(0, m.start() - 60) : m.end() + 40].replace("\n", " ")
            report.shown.append(f"{finding.family}/{finding.kind} {where}: ...{ctx}...")

    for m in _FRACTION_RE.finditer(text):
        note(_check_fraction(m), m)
    for m in _FRACTION_WORD_RE.finditer(text):
        note(_check_fraction_word(m), m)
    for m in _SPELLED_RE.finditer(text):
        note(_check_spelled(m), m)
    for m in _INDEX_BASE_RE.finditer(text):
        note(_check_index_base(m), m)
    for m in _OTHER_BASE_RE.finditer(text):
        if not _INDEX_BASE_RE.search(text[max(0, m.start() - 12) : m.end()]):
            note(
                Finding(
                    "index_base",
                    "other_base",
                    "left_alone",
                    '"=100" after a label, not a year, is a table value',
                ),
                m,
            )
    for kind, pattern, reason in _DATE_KINDS:
        for m in pattern.finditer(text):
            note(_check_date(kind, m, reason), m)
    for m in _MONTH_DAY_RE.finditer(text):
        note(_check_month_day(m), m)
    for m in _RANGE_RE.finditer(text):
        note(_check_range(m), m)
    for m in _FOOTNOTE_RE.finditer(text):
        note(_check_footnote(m), m)
    for m in _GLUED_RE.finditer(text):
        note(_check_glued(m), m)
    for m in _TOKEN_RE.finditer(text):
        token = m.group(0)
        if figures(token):
            continue
        name = _unread_class(token)
        if name is None:
            note(Finding("unread", "unclassified", "gap"), m)
        else:
            report.add(Finding("unread", name, "left_alone", _UNREAD_REASONS[name]))


def scan_units(unit_dirs: Iterable[pathlib.Path], *, show: str | None = None) -> Report:
    report = Report()
    for unit_dir in unit_dirs:
        report.units += 1
        for doc_id, text in _texts(unit_dir):
            report.documents += 1
            scan_text(text, report, where=f"{unit_dir.name}/{doc_id}", show=show)
    return report


def unit_dirs_under(paths: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for path in paths:
        if (path / "corpus").is_dir():
            out.append(path)
        else:
            out.extend(sorted(p.parent for p in path.glob("*/corpus") if p.is_dir()))
    return out


def _rows(report: Report) -> list[dict[str, str | int]]:
    return [
        {"family": f, "kind": k, "verdict": v, "reason": r, "count": n}
        for (f, k, v, r), n in sorted(report.counts.items())
    ]


def summary(report: Report) -> dict[str, object]:
    return {
        "units": report.units,
        "documents": report.documents,
        "gaps": report.failures,
        "findings": _rows(report),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "paths",
        nargs="+",
        type=pathlib.Path,
        help="unit directories, or directories that hold them",
    )
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument(
        "--show",
        choices=("gap", "left_alone", "covered"),
        help="print the text around each finding of this verdict (organiser only)",
    )
    args = parser.parse_args(argv)
    units = unit_dirs_under(args.paths)
    if not units:
        parser.error("no unit directory (one with corpus/) under the given paths")
    report = scan_units(units, show=args.show)
    result = summary(report)
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        print(f"{report.units} units, {report.documents} documents")
        for row in _rows(report):
            reason = f"  ({row['reason']})" if row["reason"] else ""
            print(
                f"{row['count']:8}  {row['verdict']:10} {row['family']}/{row['kind']}{reason}"
            )
        print(f"gaps: {report.failures}")
    for line in report.shown:
        print(line)
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
