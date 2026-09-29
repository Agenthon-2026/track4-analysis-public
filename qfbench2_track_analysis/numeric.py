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
"""

from __future__ import annotations

import bisect
import functools
import re
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
    # 10-K, 8-K, 10-year, 2-Year; 5.2.0: never the decimals of an amount ("2.6-percent" read as 2)
    re.compile(r"(?<![\d.,])\b\d{1,3}-[A-Za-z]"),
    # S000002564, CIK0001033225, T10Y2Y; 5.2.x: dotted labels too ("E3.3" read as 0.3 before)
    re.compile(r"\b[A-Z]{1,5}\d+(?:\.\d+)*[A-Z0-9]*\b"),
    re.compile(r"\b(?:CERT|CIK|No\.?|#|ticker|Form)\s*\d+\b", re.IGNORECASE),
)

#: 4-1/4 -> 4.25 before anything else is read; FOMC statements write the target range this way.
_FRACTION = re.compile(r"(?<!\d)(\d+)[-‑–](\d)/(\d)\b")

#: One figure: optional sign, digits with optional thousands separators, optional decimals,
#: optional scale suffix. A leading-dot decimal (".28") is a figure. The trailing lookahead
#: refuses a figure glued to a letter it did not consume, so "1Q" and "CIK123" do not yield 1
#: and 123, while a trailing period ("... $58,428,612.00.") does not break the match.
_FIGURE = re.compile(
    r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?"
    r"(?:(?:[kKmMbBtT]n?|bn|mn|tn|MM|bps?)\b)?(?![\w]|\.\d|,\d)"
)
#: 5.2.0: "MM" (millions, upper case only: "300mm" is a length) and basis points glued to the
#: number ("25bp", "-5bps") are figures too; before, the glued letters hid the amount entirely.
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
}

#: Scale steps a span figure may be re-based by to match a claim figure: powers of a thousand
#: (table headed "in thousands" / "in millions", or a claim written in billions), plus the two
#: steps that relate a percent to its ratio.
_SCALE_STEPS: tuple[int, ...] = tuple(range(-12, 13, 3)) + (-2, 2)

#: The longest text the check will read. A cited span can be a whole filing; the claim is capped
#: upstream. Nothing beyond this many characters is examined, in either direction.
_MAX_CHARS = 200_000


def _normalise(text: str) -> str:
    text = text[:_MAX_CHARS]
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
        raw = match.group(0)
        token = raw.replace(",", "").lstrip("+-")
        exponent = 0
        suffix = re.search(r"[A-Za-z]+$", token)
        if suffix:
            # An unknown suffix ("12kn") is no scale, never an exception (5.1.2): a KeyError
            # here aborted a whole evaluation on one claim's prose.
            exponent = _SUFFIX_EXPONENT.get(suffix.group(0).lower(), 0)
            token = token[: suffix.start()]
        if not token or not any(ch.isdigit() for ch in token):
            continue
        try:
            value = Decimal(token).scaleb(exponent)
        except (
            InvalidOperation
        ):  # pragma: no cover - the regex admits only valid decimals
            continue
        out.append(abs(value))
    return tuple(out)


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
) -> tuple[tuple[Decimal, ...], int]:
    """The claim's figures that no passage in `span_texts` carries, and how many figures the claim
    states in all (both after the exact own-value exemption `claim_number_status` applies).

    `names`: the unit's own entity names and tickers; a number written as part of one of
    them ("Phillips 66", "S&P 500", "3M") is not a figure (`name_fragments`). `span_cap`:
    a cited span longer than `span_cap` characters anchors no figure (`_readable`), so a
    whole-document citation cannot anchor a figure by coincidence somewhere in a long filing: a
    claim with figures cites the passage that states them. The verbatim-quote tests still read
    the whole span (a verbatim quote cannot misstate a figure)."""
    spans = list(span_texts)
    own = {abs(Decimal(str(v))) for v in submitted if v is not None}
    fragments = name_fragments(tuple(names))
    read = _blank_names(claim_text, fragments) if fragments else claim_text
    claim_figures = [f for f in figures(read) if f not in own]
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
        claim_figures = [f for f in figures(read_label) if f not in own]
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
) -> ClaimNumberStatus:
    """Classify one claim against the passages it cites.

    ``no_figures``  the claim states no amount (after the exclusions above and after
        removing any figure exactly equal to one of `submitted`, the participant's own scored
        values); nothing to anchor, the judge decides.
    ``anchored``    rule "any": at least one figure in the claim appears in at least one cited
        passage. Rule "every": every figure in the claim appears in at least one cited passage.
    ``unanchored``  rule "any": the claim carries figures and none of them appears in any cited
        passage. Rule "every": at least one of the claim's figures appears in no cited passage.
    """
    if rule not in FIGURE_RULES:
        raise ValueError(f"figure rule {rule!r} is not one of {list(FIGURE_RULES)}")
    if rule == "every":
        missing, stated = missing_figures(
            claim_text, span_texts, submitted=submitted, names=names, span_cap=span_cap
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
