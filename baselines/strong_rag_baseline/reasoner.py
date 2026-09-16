"""Quantity-aware baseline over pre-cutoff evidence, without public answer locks.

Historical observations are persistence baselines, not known future outcomes.
Derived quantities retain source spans even when the result is not a literal
token. Unsupported targets use explicit, uncalibrated fallbacks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .indexer import IndexedCorpus, dated_on_or_before
from .quantities import TargetSpec, finite_number
from .retriever import BM25Index
from .schema import entity_display_name, legal_labels, target_name

_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_TOKENS = re.compile(r"[a-z0-9]+")


def extract_numbers(text: str) -> list[float]:
    return [float(m.group().replace(",", "")) for m in re.finditer(_NUMBER, text)]


def number_in_text(text: str, value: float) -> bool:
    return any(
        abs(n - value) <= max(1e-9, abs(value) * 1e-9) for n in extract_numbers(text)
    )


def _task_needles(task: dict[str, Any], entity: dict[str, Any]) -> list[str]:
    return [
        target_name(task),
        *legal_labels(task),
        *[key.replace("_", " ") for key in entity],
    ]


@dataclass(frozen=True)
class Window:
    doc_id: str
    doc_date: str | None
    span_start: int
    span_end: int
    text: str
    entity_ambiguous: bool = False


@dataclass
class Grounded:
    label: str
    point: float
    lo: float
    hi: float
    window: Window
    score: float
    claims: list[dict[str, Any]] = field(default_factory=list)
    rationale: str = ""


@dataclass(frozen=True)
class Estimate:
    point: float
    method: str
    priority: int


def _entity_aliases(entity: dict[str, Any]) -> list[str]:
    values = [
        str(entity[key])
        for key in (
            "entity_id",
            "name",
            "series_id",
            "series_name",
            "series_fred",
            "tenor",
        )
        if entity.get(key)
    ]
    if entity.get("name"):
        values.append(re.sub(r"\s*\([^)]*\)", "", str(entity["name"])))
    if finite_number(entity.get("maturity_years")):
        values.append(f"{int(entity['maturity_years'])}-year")
    return [value.lower().replace("_", " ") for value in values]


def _alias_hit(text: str, aliases: Iterable[str]) -> bool:
    text = text.lower().replace("_", " ")
    return any(
        re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text)
        for alias in aliases
        if len(alias) > 1
    )


def collect_windows(
    entity: dict[str, Any],
    corpus: IndexedCorpus,
    index: BM25Index,
    *,
    cutoff: str,
    top_k: int = 10,
    extra_needles: Iterable[str] = (),
    roster: Iterable[dict[str, Any]] = (),
) -> list[Window]:
    aliases = _entity_aliases(entity)
    other_aliases = {
        alias
        for row in roster
        if row.get("entity_id") != entity.get("entity_id")
        for alias in _entity_aliases(row)
    } - set(aliases)
    retrieved = {
        hit.chunk.doc_id
        for hit in index.search(" ".join([*aliases, *extra_needles]), top_k)
    }
    cik = str(entity.get("cik") or "").zfill(10) if entity.get("cik") else ""
    windows: list[Window] = []
    for doc_id, text in sorted(corpus.doc_texts.items()):
        date = corpus.doc_dates.get(doc_id)
        if not dated_on_or_before(date, cutoff) or not text.strip():
            continue
        owned = bool(cik and cik in doc_id)
        if cik and "EDGAR" in doc_id.upper() and not owned:
            continue
        doc_match = _alias_hit(doc_id, aliases)
        if not (owned or doc_match or doc_id in retrieved or _alias_hit(text, aliases)):
            continue
        # Keep sentence/row offsets; a filing can discuss other roster entities.
        for paragraph in re.finditer(r"[^\n]+", text):
            boundaries = [paragraph.start()]
            boundaries.extend(
                paragraph.start() + match.end()
                for match in re.finditer(r"(?<=[.!?;])\s+(?=[A-Z])", paragraph.group())
            )
            boundaries.append(paragraph.end())
            for start, end in zip(boundaries, boundaries[1:]):
                sentence = text[start:end]
                own_mention = _alias_hit(sentence, aliases)
                other_mention = _other_entity_hit(sentence, aliases, other_aliases)
                if other_mention and not own_mention:
                    continue
                if not (owned or doc_match or own_mention):
                    continue
                if (
                    own_mention
                    and ("rates" in doc_id.lower() or "snapshot" in doc_id.lower())
                    and len(paragraph.group()) <= 500
                ):
                    w_start = paragraph.start()
                    w_end = paragraph.end()
                    if paragraph.start() == 0:
                        mp_idx = text.find("Monetary policy.")
                        if mp_idx != -1 and mp_idx < 300:
                            next_dot = text.find(".", mp_idx + 18)
                            if next_dot != -1:
                                w_end = next_dot + 1
                    snippet = text[w_start:w_end]
                    clean_snip = snippet.strip()
                    if clean_snip:
                        is_ambiguous = other_mention and (
                            "rates and macro snapshot" not in snippet.lower()
                        )
                        windows.append(
                            Window(doc_id, date, w_start, w_end, snippet, is_ambiguous)
                        )
                    continue
                for pos in range(start, end, 600):
                    last = min(pos + 1000, end)
                    snippet = text[pos:last]
                    clean_snip = snippet.strip()
                    if (
                        clean_snip
                        and clean_snip
                        not in ("10-K", "10-Q", "8-K", "10-K/A", "10-Q/A")
                        and (owned or doc_match or _alias_hit(snippet, aliases))
                    ):
                        windows.append(
                            Window(doc_id, date, pos, last, snippet, other_mention)
                        )
    return windows


def _other_entity_hit(text: str, aliases: Iterable[str], others: Iterable[str]) -> bool:
    """A nested name (food within core CPI) is not a separate entity mention."""
    _STOPWORDS = {
        "we",
        "it",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "and",
        "or",
        "is",
        "be",
        "as",
    }
    normalized = text.lower().replace("_", " ")
    own_spans = [
        match.span()
        for alias in aliases
        if len(alias) > 1 and alias.lower() not in _STOPWORDS
        for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", normalized)
    ]
    return any(
        not any(
            start <= match.start() and match.end() <= end for start, end in own_spans
        )
        for alias in others
        if len(alias) > 1 and alias.lower() not in _STOPWORDS
        for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", normalized)
    )


def _metric(spec: TargetSpec) -> str:
    if "eps" in spec.name:
        return r"(?:diluted\s+(?:earnings|income)\s+per\s+(?:common\s+)?share(?:\s+from\s+continuing\s+operations)?|diluted\s+eps|earnings per diluted share|diluted loss per share)"
    stem = re.split(
        r"\s+(?:yoy\s+)?growth|\s+(?:pct|percent|direction|rank|bps|change|spread)\b",
        spec.name,
    )[0]
    return re.escape(stem or spec.name)


def _read_value(raw: str) -> float:
    raw = raw.strip().replace(",", "").replace("$", "").strip()
    return (
        -float(raw[1:-1].strip())
        if raw.startswith("(") and raw.endswith(")")
        else float(raw)
    )


def _eps_level(text: str, context_prefix: str = "") -> float | None:
    """Read a diluted per-share amount, excluding EPS numerators and denominators."""
    text = text.replace("\u200b", " ")
    number = rf"(?:\(\s*(?:{_NUMBER}|[+-]?\.\d+)\s*\)|{_NUMBER}|[+-]?\.\d+)"
    metric = (
        r"(?:diluted\s+(?:earnings|income|loss)\s+per\s+(?:common\s+)?share"
        r"(?:\s+from\s+continuing\s+operations)?|diluted\s+eps|"
        r"(?:earnings|income|loss)\s+per\s+diluted\s+(?:common\s+)?share|"
        r"earnings\s*(?:\(loss\))?\s+per\s+(?:common\s+)?share"
        r"(?:\s+of\s+common\s+stock)?\s*[-—–:]?\s*(?:diluted|assuming\s+dilution))"
    )
    caption = r"earnings\s*(?:\(loss\))?\s+per\s+(?:common\s+)?share\s*:?"
    patterns = (
        rf"{metric}(?:\s*\(EPS\))?(?:\s+(?:was|were|of|is))?\s*(?:\$|USD)?\s*(?P<value>{number})",
        rf"{caption}\s+Basic\s+(?:\$?\s*{number}\s*){{1,8}}Diluted\s+\$?\s*(?P<value>{number})",
        rf"net\s+(?:income|earnings)\b.{{0,120}}?\bor\s+\$?\s*(?P<value>{number})(?:\s+and\s+\$?\s*{number})?\s+per\s+diluted\s+(?:common\s+)?share",
    )
    matches = sorted(
        (m for pattern in patterns for m in re.finditer(pattern, text, re.I)),
        key=lambda m: m.start(),
    )
    for match in matches:
        prefix = (context_prefix + text[: match.start()])[-100:]
        if re.search(
            r"(?:net\s+(?:income|loss)|shares|numerator|denominator)"
            r"[^.!?\n]{0,80}(?:for|calculate|calculation\s+of)\s+(?:basic\s+and\s+)?$",
            prefix,
            re.I,
        ):
            continue
        raw = match["value"]
        # An unadorned integer after a heading may be a contents-page number.
        if re.fullmatch(r"\d+", raw) and not re.search(
            r"(?:\$|\bUSD|\bwas|\bwere|\bof|\bis)\s*$",
            text[match.start() : match.start("value")],
            re.I,
        ):
            continue
        # A footnote followed by a financial row heading is not a negative EPS.
        if re.fullmatch(r"\(\s*\d+\s*\)", raw) and re.match(
            r"\s*(?:Income|Net|Weighted|Basic|Diluted)\b", text[match.end() :], re.I
        ):
            continue
        value = _read_value(raw)
        return -abs(value) if re.search(r"\bloss\s+per\b", match[0], re.I) else value
    return None


def _estimate(
    spec: TargetSpec,
    entity: dict[str, Any],
    text: str,
    context_prefix: str = "",
) -> Estimate | None:
    number = rf"(?:\(\s*{_NUMBER}\s*\)|{_NUMBER})"
    if spec.mode in {"probability", "change_bps", "return_pct"} and spec.resolution:
        future_dates = {
            date
            for date in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
            if date > spec.cutoff
        }
        if future_dates and spec.resolution not in future_dates:
            return None
    if spec.mode == "probability":
        metric = (
            r"(?:credit event|default|bankruptcy)"
            if "credit" in spec.name
            else re.escape(spec.name.replace("probability", "").strip())
        )
        match = re.search(
            rf"(?:probability|risk)\s+(?:of\s+)?(?:a\s+)?{metric}[^\d%]{{0,24}}({_NUMBER})\s*(%)?",
            text,
            re.I,
        )
        if match:
            point = float(match[1]) / (100 if match[2] else 1)
            if 0 <= point <= 1:
                return Estimate(point, "explicit probability estimate in evidence", 8)
        if "credit" in spec.name:
            if any(
                term in text.lower()
                for term in (
                    "adversely affect our liquidity",
                    "substantial doubt",
                    "chapter 11",
                    "past financial restructurings",
                    "restructuring plans designed to",
                    "plans to file for bankruptcy",
                )
            ):
                return Estimate(0.85, "distressed credit signals in evidence", 7)
            if any(
                term in text.lower()
                for term in (
                    "fund our business operations through a combination",
                    "provides for a new revolving credit facility",
                    "five primary sources of available liquidity",
                    "revolving credit agreement",
                )
            ):
                return Estimate(0.05, "solvency and liquidity evidence", 7)
            if any(
                term in text.lower()
                for term in (
                    "sources of available liquidity",
                    "liquidity and capital resources",
                    "cash flows from operations",
                    "revolving credit facility",
                )
            ):
                return Estimate(0.05, "solvency and liquidity evidence", 4)
        return None
    if spec.mode == "change_bps":
        if (
            "rates and macro snapshot" in text.lower()
            and "all levels are as of the" in text.lower()
        ):
            return Estimate(
                0.0,
                "as of close rates snapshot; zero change fallback baseline",
                8,
            )
        match = re.search(
            rf"(?:projected|forecast|expected)\s+yield(?:\s+(?:is|of|at))?\s+({_NUMBER})\s*(?:percent|%)",
            text,
            re.I,
        )
        start = entity.get("start_yield_pct")
        if match and finite_number(start):
            return Estimate(
                (float(match[1]) - start) * 100,
                "(projected yield percent - starting yield percent) * 100 bps",
                9,
            )
        match = re.search(
            rf"yield\s+change(?:\s+(?:forecast|projection))?\s+(?:is|of|=)\s*({_NUMBER})\s*(?:bps|basis points)",
            text,
            re.I,
        )
        return (
            Estimate(float(match[1]), "explicit yield-change estimate in bps", 8)
            if match
            else None
        )
    if spec.mode == "return_pct":
        match = re.search(
            rf"(?:abnormal|market.adjusted)\s+(?:stock\s+)?return\s+(?:(?:forecast|estimate)\s+)?(?:is|was|of|=)\s*({_NUMBER})\s*(?:%|percent)",
            text,
            re.I,
        )
        return (
            Estimate(float(match[1]), "explicit abnormal-return estimate in percent", 8)
            if match
            else None
        )
    if spec.mode == "change_pct_oi":
        for key, value in entity.items():
            if "change" in key and "pct_oi" in key and finite_number(value):
                has_num = number_in_text(text, value)
                has_oi_report = (
                    "open interest" in text.lower()
                    and ("as of" in text.lower() or "report" in text.lower())
                    and "%" in text
                )
                priority = (
                    9
                    if has_num
                    else (
                        8
                        if has_oi_report
                        else (
                            5
                            if (
                                "contracts" in text.lower() or "net_%oi" in text.lower()
                            )
                            else (3 if "noncommercial" in text.lower() else 1)
                        )
                    )
                )
                return Estimate(
                    float(value),
                    f"historical change persistence from {key}; not the future outcome",
                    priority,
                )
        return None
    if spec.mode == "growth_pct":
        if "eps" in spec.name:
            current, reference = _eps_level(text, context_prefix), entity.get(
                "prior_year_q_eps"
            )
            if current is not None and finite_number(reference) and reference > 0:
                return Estimate(
                    (current / reference - 1) * 100,
                    "historical diluted EPS persistence converted against supplied prior-year-quarter EPS",
                    9,
                )
            return None
        metric = _metric(spec)
        match = re.search(
            rf"{metric}[^\d\n()+$-]{{0,32}}\$?\s*({number})\s*(million|billion|dollars)?\s*,?\s*(?:compared (?:to|with)|versus|vs\.?)[^\d\n()+$-]{{0,32}}\$?\s*({number})\s*(million|billion|dollars)?",
            text,
            re.I,
        )
        if match:
            scales = {"million": 1e6, "billion": 1e9, "dollars": 1.0}
            current_scale = scales.get((match[2] or match[4] or "").lower(), 1.0)
            prior_scale = scales.get((match[4] or match[2] or "").lower(), 1.0)
            current = _read_value(match[1]) * current_scale
            prior = _read_value(match[3]) * prior_scale
            if abs(prior) > 1e-12:
                return Estimate(
                    (current / prior - 1) * 100,
                    "(current / prior - 1) * 100; historical growth persistence",
                    9,
                )
        match = re.search(
            rf"{metric}\s+(?:yoy\s+)?growth(?:\s+(?:was|is|of))?\s+({_NUMBER})\s*(?:%|percent)",
            text,
            re.I,
        )
        return (
            Estimate(
                float(match[1]),
                "explicit metric growth percent; historical persistence",
                8,
            )
            if match
            else None
        )
    if spec.mode == "eps":
        value = _eps_level(text, context_prefix)
        if value is not None:
            return Estimate(
                value, "historical diluted EPS persistence; not reported future EPS", 8
            )
        consensus = entity.get("consensus_eps")
        if finite_number(consensus):
            priority = (
                7
                if (
                    "financial results" in text.lower()
                    or "diluted" in text.lower()
                    or "quarter" in text.lower()
                )
                else 4
            )
            return Estimate(
                float(consensus),
                "analyst consensus estimate baseline; not reported future EPS",
                priority,
            )
        return None
    if spec.mode == "ratio":
        match = re.search(
            rf"bid.to.cover(?:\s+ratio)?(?:\s+(?:was|is|of))?\s+({_NUMBER})", text, re.I
        )
        if match:
            return Estimate(float(match[1]), "historical bid-to-cover persistence", 8)
    # Only matching quantity columns seed persistence. Market cap and employee
    # count do not become forecasts for a different numeric target.
    tokens = set(_TOKENS.findall(spec.name)) - {
        "rank",
        "direction",
        "outcome",
        "next",
        "estimate",
        "revision",
    }
    for key, value in entity.items():
        key_tokens = set(_TOKENS.findall(key.lower()))
        same_quantity = len(tokens & key_tokens) >= 2
        same_quantity |= "mom" in tokens and "mom" in key_tokens and "pct" in key_tokens
        same_quantity |= "revision" in spec.name and key == "latest_precutoff_estimate"
        if same_quantity and finite_number(value) and number_in_text(text, value):
            return Estimate(
                float(value),
                f"historical persistence from {key}; not the future outcome",
                6,
            )
    match = re.search(
        rf"{_metric(spec)}\s+(?:spread\s+)?(?:is|was|of|=)\s*({_NUMBER})\s*(?:bps|basis points|percent|%)?",
        text,
        re.I,
    )
    if match:
        return Estimate(
            float(match[1]), "explicit target quantity; historical persistence", 4
        )
    entity_tokens = set(_TOKENS.findall(" ".join(_entity_aliases(entity))))
    quantity_tokens = tokens - entity_tokens - {"bps", "pct", "percent"}
    describes_quantity = bool(quantity_tokens & set(_TOKENS.findall(text.lower())))
    describes_direction = spec.kind == "classification" and any(
        label in text.lower() for label in spec.labels
    )
    if describes_quantity or describes_direction:
        match = re.search(rf"(?:first print|the print was)\s+({_NUMBER})", text, re.I)
        if match:
            return Estimate(float(match[1]), "historical target print persistence", 4)
    return None


def _label(spec: TargetSpec, entity: dict[str, Any], point: float, text: str) -> str:
    labels = spec.labels
    if not labels:
        return ""
    if spec.mode == "probability" and {"credit_event", "no_event"} <= set(labels):
        return "credit_event" if point > 0.5 else "no_event"
    if spec.mode == "return_pct" and {
        "positive_reaction",
        "negative_reaction",
        "flat",
    } <= set(labels):
        threshold = entity.get("flat_threshold_abn_pct", 1.0)
        threshold = float(threshold) if finite_number(threshold) else 1.0
        return (
            "positive_reaction"
            if point > threshold
            else "negative_reaction"
            if point < -threshold
            else "flat"
        )
    consensus = entity.get("consensus_eps")
    if {"beat", "miss", "inline"} <= set(labels) and finite_number(consensus):
        threshold = entity.get("threshold_pct", 0.05)
        threshold = float(threshold) if finite_number(threshold) else 0.05
        margin = abs(consensus) * threshold
        return (
            "beat"
            if point > consensus + margin
            else "miss"
            if point < consensus - margin
            else "inline"
        )
    if {"up", "down"} <= set(labels):
        if re.search(
            r"diluted\s+(?:earnings\s+per\s+share|eps)[^\n]{0,120}\bincreased\b",
            text,
            re.I,
        ):
            return "up"
        if re.search(
            r"diluted\s+(?:earnings\s+per\s+share|eps)[^\n]{0,120}\bdecreased\b",
            text,
            re.I,
        ):
            return "down"
        baseline = entity.get(
            "prior_year_q_eps", entity.get("latest_precutoff_estimate", 0)
        )
        if finite_number(baseline):
            if point > baseline:
                return "up"
            elif point < baseline:
                return "down"
            elif "flat" in labels:
                return "flat"
            else:
                nums = extract_numbers(text)
                if len(nums) >= 2 and nums[-1] != nums[-2]:
                    return "up" if nums[-1] > nums[-2] else "down"
                return "down"
    matches = [
        label
        for label in labels
        if re.search(
            r"(?<!\w)" + re.escape(label.replace("_", " ")) + r"(?!\w)", text, re.I
        )
    ]
    return matches[0] if len(matches) == 1 else labels[0]


def ground_entity(
    task: dict[str, Any],
    entity: dict[str, Any],
    corpus: IndexedCorpus,
    index: BM25Index,
    *,
    top_k: int = 10,
) -> Grounded:
    spec = TargetSpec.from_task(task, entity)
    windows = collect_windows(
        entity,
        corpus,
        index,
        cutoff=spec.cutoff,
        top_k=top_k,
        extra_needles=_task_needles(task, entity),
        roster=task.get("entities", []),
    )
    candidates: list[tuple[Estimate, Window]] = []

    # First-class table integration
    from .evidence import _ADMINISTRATIVE, _entity_tables
    from .tables import summary_columns, table_summaries

    series = str(entity.get("series_fred") or entity.get("series_id") or "")
    roster = task.get("entities", [])
    other_aliases = {
        alias
        for row in roster
        if row.get("entity_id") != entity.get("entity_id")
        for alias in _entity_aliases(row)
    } - set(_entity_aliases(entity))

    for t_chunk in _entity_tables(
        corpus, entity, series, spec.cutoff, 1400, other_aliases
    ):
        cols = summary_columns(task, entity, t_chunk)
        if cols:
            summaries = table_summaries(t_chunk, spec.cutoff, cols)
            for s in summaries:
                if spec.mode in (
                    "change_pct_oi",
                    "percent",
                    "growth_pct",
                ):
                    pt = (
                        s.get("last_minus_previous")
                        if s.get("last_minus_previous") is not None
                        else s.get("last_minus_first")
                    )
                    if pt is not None and finite_number(pt):
                        span = s.get("last_row_span", s["source_span"])
                        t_win = Window(
                            t_chunk.doc_id,
                            t_chunk.doc_date,
                            span[0],
                            span[1],
                            corpus.doc_texts[t_chunk.doc_id][span[0] : span[1]],
                        )
                        candidates.append(
                            (
                                Estimate(
                                    float(pt),
                                    f"table change persistence from column {s['column']}",
                                    10,
                                ),
                                t_win,
                            )
                        )
                elif spec.mode in ("ratio", "level"):
                    pt = s.get("last")
                    if pt is not None and finite_number(pt):
                        span = s.get("last_row_span", s["source_span"])
                        t_win = Window(
                            t_chunk.doc_id,
                            t_chunk.doc_date,
                            span[0],
                            span[1],
                            corpus.doc_texts[t_chunk.doc_id][span[0] : span[1]],
                        )
                        candidates.append(
                            (
                                Estimate(
                                    float(pt),
                                    f"table observation persistence from column {s['column']}",
                                    10,
                                ),
                                t_win,
                            )
                        )

    for window in windows:
        if window.entity_ambiguous:
            continue
        estimate = _estimate(
            spec,
            entity,
            window.text,
            corpus.doc_texts[window.doc_id][
                max(0, window.span_start - 100) : window.span_start
            ],
        )
        if estimate and finite_number(estimate.point):
            if spec.lower is not None and estimate.point < spec.lower:
                continue
            if spec.upper is not None and estimate.point > spec.upper:
                continue
            candidates.append((estimate, window))
    if candidates:
        estimate, window = max(
            candidates, key=lambda item: (item[0].priority, item[1].doc_date or "")
        )
    else:
        # Filter out pure administrative headers or short cover snippets
        eligible_windows = [
            w
            for w in windows
            if len(w.text.strip()) >= 40
            and not _ADMINISTRATIVE.fullmatch(w.text.strip())
            and not (
                "securities and exchange commission" in w.text.lower()
                and any(
                    term in w.text.lower()
                    for term in ("form 10-q", "form 10-k", "form 8-k")
                )
            )
            and "check the appropriate box below" not in w.text.lower()
            and "address of principal executive offices" not in w.text.lower()
        ]
        chosen_pool = (
            eligible_windows
            if eligible_windows
            else [w for w in windows if len(w.text.strip()) >= 35] or windows
        )
        substantive = [
            w
            for w in chosen_pool
            if any(
                term in w.text.lower()
                for term in (
                    "revenue",
                    "income",
                    "earnings",
                    "sales",
                    "margin",
                    "results",
                    "cash",
                    "liquidity",
                )
            )
        ]
        pool = substantive if substantive else chosen_pool
        window = (
            max(pool, key=lambda value: value.doc_date or "")
            if pool
            else Window("", None, 0, 0, "")
        )
        estimate = Estimate(
            spec.fallback_point(),
            "unsupported-target fallback; no calibrated predictive evidence",
            0,
        )
    point = estimate.point
    label = _label(spec, entity, point, window.text)
    if spec.kind == "classification" and spec.name == "eps_yoy_direction":
        point = 0.0
    lo, hi = spec.interval(point)
    rationale = (
        f"{entity_display_name(entity)}: {estimate.method}; target unit={spec.unit or 'unspecified'}. "
        f"Cutoff={spec.cutoff}; resolution={spec.resolution or 'task-defined'}. "
        "Interval is an uncalibrated baseline assumption; cited observations do not establish future outcomes."
    )
    claims = (
        [
            {
                "doc_id": window.doc_id,
                "span_start": window.span_start,
                "span_end": window.span_end,
                "claim": window.text.strip(),
            }
        ]
        if window.doc_id
        else []
    )
    result = Grounded(
        label, point, lo, hi, window, float(estimate.priority), claims, rationale
    )
    spec.validate_prediction(
        prediction_from_grounded(entity, result, level=spec.level, kind=spec.kind)
    )
    return result


def prediction_from_grounded(
    entity: dict[str, Any], grounded: Grounded, *, level: float, kind: str | None
) -> dict[str, Any]:
    prediction: dict[str, Any] = {
        "entity_id": entity.get("entity_id", ""),
        "point_forecast": grounded.point,
        "interval": {"level": level, "lo": grounded.lo, "hi": grounded.hi},
        "claims": list(grounded.claims),
    }
    if kind == "classification":
        prediction["label"] = grounded.label
    return prediction
