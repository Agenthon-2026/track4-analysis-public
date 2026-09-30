"""Pre-submission citation rail for Track 4 answers.

Two local checks an agent can run on its OWN draft answer before submitting:

1. **Date rail** — every cited ``doc_id`` must resolve to a corpus document whose
   ``doc_date`` is on or before the task ``cutoff_date``. Citing a document that
   is missing from the frozen corpus, or dated after the cutoff, is flagged.
2. **Shape rail** — every claim must carry a well-formed ``(doc_id, span_start,
   span_end)`` triple whose offsets resolve inside the cited document's text.

These are the two mistakes the adversarial variants (stale-filing traps,
Family 5) are designed to elicit. A claim may also cite the unit's task table
(``doc_id: "task"``); pass ``task=`` (the parsed task.json) and the span is
checked against the task-table text the scorer builds, inside the citing
entity's own row.

:func:`check_claim_rules` (scorer 5.2.0) runs the DETERMINISTIC per-claim rules
of the analysis scorer on a whole answer, with the scorer's own code (it imports
``qfbench2_track_analysis``): wrong-entity citations, out-of-range offsets,
malformed claims (empty, too long, over the judge-token cap when a tokenizer is
available) and the every-figure rule (whole cited span, verbatim-quote pass,
the unit's own names and tickers exempt), a citation over 8,000 characters and a
content-free claim (both false from scorer 5.2.2). Only the NLI contradiction check is left out.

A third check, :func:`check_submitted_reasons`, covers the optional top-level
``submitted_reasons`` field that the reasoning grader reads: its shape, its
citations, the published caps and the deny list. Running the rail locally lets an agent drop or
repair a bad claim before it ever reaches the organizer's scoring pipeline.

The rail is advisory and participant-side only: it reduces YOUR gate failures.
The competition's embargo and faithfulness gates are deterministic organizer
code and are the authority on every submission.

Text/offset convention mirrors the scorer (and ``baseline_agent.indexer``): a
document's text is its flat ``text`` field if present, else its ``spans[].text``
values joined with a single space; span offsets are global character offsets
into that string.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_MANIFEST_NAME = "manifest.json"


@dataclass
class CorpusDoc:
    doc_id: str
    text: str
    doc_date: str | None


@dataclass
class RailFinding:
    """One problem the rail found in a draft answer."""

    entity_id: str
    claim_index: int
    # check_answer: unknown_doc | stale_doc | missing_field | bad_span | empty_claim |
    #   task_row | task_unchecked | claim_citations
    # check_claim_rules: claim_wrong_entity | claim_out_of_range | claim_malformed |
    #   claim_unanchored | claim_over_cap | claim_content_free |
    #   claim_tokens_unchecked | unit_refused
    # check_submitted_reasons: reasons_shape | reason_citation | cap_citation_chars |
    #   cap_answer_bytes | cap_reason_bytes | cap_evidence_bytes | deny_list | duplicate_reason
    code: str
    message: str

    def __str__(self) -> str:
        if self.entity_id == "submitted_reasons":
            where = "whole unit" if self.claim_index < 0 else f"reason#{self.claim_index}"
            return f"[{self.code}] submitted_reasons {where}: {self.message}"
        return f"[{self.code}] entity={self.entity_id} claim#{self.claim_index}: {self.message}"


def _doc_text(doc: dict) -> str:
    """Concatenate spans (or use a flat ``text`` field) — mirrors the scorer."""
    if isinstance(doc.get("text"), str):
        return doc["text"]
    spans = doc.get("spans")
    if isinstance(spans, list):
        return " ".join(sp.get("text", "") for sp in spans if isinstance(sp, dict))
    return ""


def load_corpus(corpus_dir: str | Path) -> dict[str, CorpusDoc]:
    """Load every corpus document keyed by ``doc_id`` (skips the manifest)."""
    corpus_dir = Path(corpus_dir)
    docs: dict[str, CorpusDoc] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        if path.name == _MANIFEST_NAME:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        doc_id = raw.get("doc_id", path.stem)
        docs[doc_id] = CorpusDoc(
            doc_id=doc_id, text=_doc_text(raw), doc_date=raw.get("doc_date")
        )
    return docs


def filter_retrieved(
    docs: list[CorpusDoc], cutoff_date: str
) -> tuple[list[CorpusDoc], list[CorpusDoc]]:
    """Split a retrieval pool into (usable, stale) by ``doc_date <= cutoff_date``.

    Use this rail at retrieval time so post-cutoff material never reaches the
    reasoning step. Documents with no ``doc_date`` are treated as stale — an
    undatable document cannot be shown to be embargo-safe.
    """
    usable: list[CorpusDoc] = []
    stale: list[CorpusDoc] = []
    for doc in docs:
        if doc.doc_date is not None and doc.doc_date <= cutoff_date:
            usable.append(doc)
        else:
            stale.append(doc)
    return usable, stale


#: The reserved doc_id of the unit's task table (``qfbench2_track_analysis.corpus.TASK_DOC_ID``).
TASK_DOC_ID = "task"


#: The claim-level citations list, removed in scorer 5.2.2 (a claim cites one span of one
#: document through its own doc_id, span_start and span_end).
NESTED_CITATIONS_KEY = "citations"
_CITATIONS_TEXT = (
    "carries a claim-level `citations` list, a shape removed in scorer 5.2.2: the scorer does "
    "not read the list, counts the claim as false (malformed) and never puts it to the judge; "
    "cite one span per claim with the claim's own doc_id, span_start and span_end"
)


def check_answer(
    answer: dict, corpus: dict[str, CorpusDoc], cutoff_date: str, *, task: dict | None = None
) -> list[RailFinding]:
    """Run both rails over a draft answer; return every finding (empty = clean).

    ``task`` (the parsed task.json) lets a ``doc_id: "task"`` citation be checked: its span must
    lie inside the citing entity's own row of the task table (``task_row`` otherwise, the
    scorer's wrong-entity rule) and inside the table (``bad_span``). Without ``task`` such a
    citation is reported ``task_unchecked``, never ``unknown_doc``."""
    findings: list[RailFinding] = []
    table = None
    for entity in answer.get("entity_predictions", []):
        entity_id = str(entity.get("entity_id", "?"))
        for i, claim in enumerate(entity.get("claims", [])):
            if isinstance(claim, dict) and NESTED_CITATIONS_KEY in claim:
                findings.append(RailFinding(entity_id, i, "claim_citations", _CITATIONS_TEXT))
            if isinstance(claim, dict) and claim.get("doc_id") == TASK_DOC_ID and task is not None:
                if table is None:
                    table = _task_table(task)
                findings.extend(_check_task_claim(entity_id, i, claim, table))
                continue
            findings.extend(_check_claim(entity_id, i, claim, corpus, cutoff_date))
    return findings


def _task_table(task: dict) -> tuple[str, dict[str, tuple[int, int]]]:
    """The task-table text and each entity's row, from the scorer's own renderer."""
    from qfbench2_track_analysis.corpus import task_table_text

    return task_table_text(task)


def _check_task_claim(
    entity_id: str, index: int, claim: dict, table: tuple[str, dict[str, tuple[int, int]]]
) -> list[RailFinding]:
    findings: list[RailFinding] = []
    text, rows = table
    start, end = claim.get("span_start"), claim.get("span_end")
    if not str(claim.get("claim", "")).strip():
        findings.append(RailFinding(entity_id, index, "empty_claim", "claim text is empty"))
    if not (_is_offset(start) and _is_offset(end)) or not start < end <= len(text):
        findings.append(RailFinding(
            entity_id, index, "bad_span",
            f"task span [{start}, {end}) is not a slice of the task table (length {len(text)})"))
        return findings
    row = rows.get(entity_id)
    if row is None or not row[0] <= start < end <= row[1]:
        findings.append(RailFinding(
            entity_id, index, "task_row",
            f"task span [{start}, {end}) is not inside {entity_id}'s own row "
            f"{list(row) if row else 'none'}; the scorer counts the claim false (wrong entity)"))
    return findings


def _check_claim(
    entity_id: str,
    index: int,
    claim: dict,
    corpus: dict[str, CorpusDoc],
    cutoff_date: str,
) -> list[RailFinding]:
    findings: list[RailFinding] = []

    def flag(code: str, message: str) -> None:
        findings.append(RailFinding(entity_id, index, code, message))

    # Shape rail: required fields present and well-typed.
    missing = [k for k in ("doc_id", "span_start", "span_end", "claim") if k not in claim]
    if missing:
        flag("missing_field", f"claim is missing field(s): {', '.join(missing)}")
        return findings  # nothing further is checkable

    if not str(claim["claim"]).strip():
        flag("empty_claim", "claim text is empty")

    start, end = claim["span_start"], claim["span_end"]
    if not isinstance(start, int) or not isinstance(end, int):
        flag("bad_span", f"span offsets must be integers (got {start!r}, {end!r})")
        return findings
    if start < 0 or end <= start:
        flag("bad_span", f"span [{start}, {end}) is not a valid half-open range")
        return findings

    # Date rail: the cited document must exist in the frozen corpus and pre-date
    # the cutoff. A doc_id the corpus does not contain usually means the agent
    # cited something it fetched live — exactly the stale-evidence mistake.
    if claim["doc_id"] == TASK_DOC_ID:
        flag("task_unchecked", 'doc_id "task" cites the task table; pass task=<task.json> to check the span')
        return findings
    doc = corpus.get(claim["doc_id"])
    if doc is None:
        flag(
            "unknown_doc",
            f"cited doc_id {claim['doc_id']!r} is not in the frozen corpus",
        )
        return findings
    if doc.doc_date is None or doc.doc_date > cutoff_date:
        flag(
            "stale_doc",
            f"cited doc {doc.doc_id!r} has doc_date={doc.doc_date!r}, "
            f"after cutoff {cutoff_date!r}",
        )

    # Shape rail, continued: offsets must resolve inside the document text.
    if end > len(doc.text):
        flag(
            "bad_span",
            f"span [{start}, {end}) exceeds document length {len(doc.text)}",
        )

    return findings


# --------------------------------------------------------------------------- #
# submitted_reasons: the field the reasoning grader reads                      #
# --------------------------------------------------------------------------- #

#: Published caps (scorer 5.2.2). Reasons are checked in the order they are
#: submitted: a reason is judged only if every citation in it is at most MAX_CITATION_CHARS and,
#: together with the reasons already judged, the reasons stay within MAX_REASON_BYTES and their
#: cited evidence within MAX_EVIDENCE_BYTES. A reason that does not fit is not judged and scores
#: 0; later reasons are still checked. MAX_ANSWER_BYTES applies to the whole unit.
MAX_REASONS = 3
MAX_CITATION_CHARS = 8_000
#: Per-unit caps in UTF-8 bytes of the compact JSON the judge reads (the grader's measure).
#: They sum to MAX_JUDGE_BYTES, so an answer within them never reaches that limit.
MAX_ANSWER_BYTES = 3_000
MAX_REASON_BYTES = 6_500
MAX_EVIDENCE_BYTES = 46_500
MAX_JUDGE_BYTES = MAX_ANSWER_BYTES + MAX_REASON_BYTES + MAX_EVIDENCE_BYTES  # 56,000

#: Published deny list (case-insensitive substrings of participant-written text). From scorer
#: 5.2.2 "://" is not on it: URLs in reasons are masked (`mask_reason_uris`), not refused. The
#: list still runs on the text as written, so a URL that carries a listed token is refused, and
#: so is a "://" left over after masking (one with no scheme letters before it).
DENY_LIST = (
    "leaderboard",
    "canary",
    "/home/",
    "units/",
    "reference/",
    "outcome.json",
    "team_id",
    "team name",
    "participant_id",
    "participant name",
    "submission_id",
    "other submission",
)

_REASON_TEXT_FIELDS = ("premise", "mechanism", "answer_implication")
_REASON_REQUIRED = ("reason_id",) + _REASON_TEXT_FIELDS
_ANSWER_FIELDS = ("label", "point_forecast", "interval", "label_probs")
_REASONS_ENTITY = "submitted_reasons"


def _duplicate_key(text: str) -> str:
    """The grader's content-duplicate comparison: NFC, format (Cf) characters removed, casefolded,
    whitespace runs collapsed; exact equality after that."""
    visible = "".join(
        ch for ch in unicodedata.normalize("NFC", text) if unicodedata.category(ch) != "Cf"
    )
    return " ".join(visible.casefold().split())


def _is_offset(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


#: The grader masks every URI in cited corpus text with the same number of this character
#: (3 UTF-8 bytes each) before the judge reads it; the byte caps count the masked text.
_URI_MASK = "\u2588"
_URI_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s,;()\[\]{}<>\"']*")


def _masked(text: str) -> str:
    return _URI_PATTERN.sub(lambda m: _URI_MASK * (m.end() - m.start()), text)


def _compact_bytes(items: list) -> int:
    text = json.dumps(items, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return len(text.encode("utf-8")) - 2  # the list's own brackets


#: A URL in a reason field, as the reasoning grader finds it (scorer 5.2.2): a scheme URL, a scheme-less "//host.tld" or a "www." host, matched in a
#: FOLDED view of the text (`fold_for_detection`): each character NFKC-normalised, the slash and
#: colon look-alikes NFKC keeps apart read as "/" and ":", and format characters (Unicode Cf) and
#: Hangul fillers removed. The same detection is `qfbench2_track_analysis.numeric`'s (a test keeps
#: the two equal); this module stays standard library only.
_SLASH_LOOKALIKES = ("\u2044", "\u2215", "\u29f8", "\u2571", "\u27cb", "\u3033", "\u1735", "\u2cfa",
                     "\ufe68", "\u2e4a", "\u0338")
_COLON_LOOKALIKES = ("\u2236", "\ua789", "\u02d0")
_LOOKALIKES = {**{c: "/" for c in _SLASH_LOOKALIKES}, **{c: ":" for c in _COLON_LOOKALIKES}}
_INVISIBLE_FILLERS = frozenset("\u115f\u1160\u3164\uffa0")
_URL_TAIL = r"[^\s,;()\[\]{}<>\"']*"
_URI = re.compile(
    r"[A-Za-z][A-Za-z0-9+.\-]{0,63}://" + _URL_TAIL
    + r"|(?<![\w/:])//[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9-]+)+" + _URL_TAIL
    + r"|(?<![\w.@/])[Ww]{3}\.[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*" + _URL_TAIL
)
#: The grader's premise filler: one "#" per masked character.
PREMISE_URI_FILLER = "#"


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
#: A premise is a quote only with at least this many words once its URLs are masked.
MIN_PREMISE_QUOTE_WORDS = 3


def _mask_premise_uris(text: str) -> str:
    """`text` with every character of every URL found in its folded view replaced by "#": from the
    first to the last original character of each match, format characters included."""
    folded, index = fold_for_detection(text)
    ranges: list[list[int]] = []
    for match in _URI.finditer(folded):
        start = match.start() if index is None else index[match.start()]
        end = match.end() if index is None else index[match.end() - 1] + 1
        if ranges and start <= ranges[-1][1]:
            ranges[-1][1] = max(ranges[-1][1], end)
            continue
        ranges.append([start, end])
    if not ranges:
        return text
    chars = list(text)
    for start, end in ranges:
        chars[start:end] = PREMISE_URI_FILLER * (end - start)
    return "".join(chars)


def mask_reason_uris(text: str) -> str:
    """A reason field as the grader sends it: every URL replaced, equal length, by "#" (1 byte
    per character, so a masked reason is never larger than the reason as submitted)."""
    return _mask_premise_uris(text)


def premise_is_corpus_quote(premise: str, masked_corpus_texts: list[str]) -> bool:
    """The grader's rule for a premise that is exempt from the deny list: with every URL masked
    by "#" (one per character), it has at least `MIN_PREMISE_QUOTE_WORDS` words (the filler and
    bare punctuation count as no word) and is a substring of one corpus document masked the same
    way. A bare URL, a bare deny-list token ("units/", "canary", "/home/") or a two-word quote is
    not a quote, and a premise that adds a word of your own is not either."""
    quote = _mask_premise_uris(premise.strip())
    words = [w for w in quote.replace(PREMISE_URI_FILLER, " ").split() if any(c.isalnum() for c in w)]
    return len(words) >= MIN_PREMISE_QUOTE_WORDS and any(quote in text for text in masked_corpus_texts)


def check_submitted_reasons(
    answer: dict, corpus: dict[str, CorpusDoc], cutoff_date: str
) -> list[RailFinding]:
    """Check the optional top-level ``submitted_reasons`` field; return every finding
    (`_check_reasons` documents the codes). `reasons_judged` gives the per-reason plan."""
    return _check_reasons(answer, corpus, cutoff_date)[0]


def _check_reasons(
    answer: dict, corpus: dict[str, CorpusDoc], cutoff_date: str
) -> tuple[list[RailFinding], list[dict]]:
    """Check the optional top-level ``submitted_reasons`` field; return every finding.

    An absent field is clean (no reasons are submitted and none are judged). Findings use
    ``entity_id="submitted_reasons"`` and ``claim_index`` = the reason's position, or -1 for
    a finding about the whole unit (a cap, or the field itself).

    Codes, and what the grader does about each:

    - ``reasons_shape`` -- the field is not a list of 1..3 objects, a reason lacks
      ``reason_id``/``premise``/``mechanism``/``answer_implication`` or one of them is not a
      string, or ``scope``/``scope.entities``/``citations`` has the wrong shape. The answer
      fails the published schema.
    - ``reason_citation`` -- a citation does not resolve in the frozen corpus (unknown
      document, empty or out-of-range span) or its document is dated after the cutoff. The
      judge never sees that passage; the rest of the reason is still judged.
    - Caps, checked reason by reason in submitted order (scorer 5.2.2): ``cap_citation_chars``
      (a citation in this reason > 8,000 characters), ``cap_reason_bytes`` (this reason with
      the reasons judged before it > 6,500 UTF-8 bytes of compact JSON: id, premise, mechanism,
      answer_implication, URLs masked by "#") and ``cap_evidence_bytes`` (their resolved cited
      passages > 46,500 bytes). That reason is not judged and scores 0; later reasons are still
      checked. ``cap_answer_bytes`` (your per-entity answer > 3,000) stops the whole unit's
      reasoning. Nothing is clipped. The three byte caps sum to the grader's 56,000-byte limit.
    - ``deny_list`` -- a deny-list phrase (on the text as written, URLs included, or a "://"
      left after masking) in ``mechanism`` or ``answer_implication``, or in a ``premise`` that is
      not a verbatim corpus quote (`premise_is_corpus_quote`: at least 3
      words once URLs are masked, and, URLs masked on both sides, a substring of one corpus
      document). The grader refuses that unit's reasoning, which scores 0; the analysis score is
      unaffected.

    The byte backstop is computed as the compact-JSON UTF-8 size (``ensure_ascii=False``,
    ``separators=(",", ":")``, ``sort_keys=True``, minus two bytes per list for its
    brackets) of: your per-entity answer projected to ``entity_id`` plus whichever of
    ``label``/``point_forecast``/``interval``/``label_probs`` each row carries; your reasons
    projected to ``reason_id``/``premise``/``mechanism``/``answer_implication``; and one
    ``{doc_id, span_start, span_end, trusted_text, reason_id}`` item per resolving
    pre-cutoff citation. It approximates the grader to within the grader's renumbering of
    reason ids to ``r1``..``r3`` (and the grader keeps only the answer fields the unit
    declares, so this count can only be the larger).

    Advisory, like the rest of this module: the organiser-side grader is the authority.
    """
    if _REASONS_ENTITY not in answer:
        return [], []
    findings: list[RailFinding] = []

    def flag(index: int, code: str, message: str) -> None:
        findings.append(RailFinding(_REASONS_ENTITY, index, code, message))

    reasons = answer[_REASONS_ENTITY]
    if not isinstance(reasons, list):
        flag(-1, "reasons_shape", f"submitted_reasons must be a list (got {type(reasons).__name__})")
        return findings, []
    if not 1 <= len(reasons) <= MAX_REASONS:
        flag(
            -1,
            "reasons_shape",
            f"submitted_reasons must hold 1 to {MAX_REASONS} reasons (got {len(reasons)}); "
            "omit the field to submit none",
        )

    quotable = [_mask_premise_uris(doc.text) for doc in corpus.values()]
    projected_reasons: list[dict] = []
    reason_trusted: list[list[dict]] = []
    reason_over_cap: list[bool] = []
    reason_duplicate: list[bool] = []
    reason_index: list[int] = []
    seen_content: dict[str, int] = {}

    for i, reason in enumerate(reasons):
        if not isinstance(reason, dict):
            flag(i, "reasons_shape", "a reason must be an object")
            continue
        reason_duplicate_now = False
        missing = [k for k in _REASON_REQUIRED if k not in reason]
        if missing:
            flag(i, "reasons_shape", f"reason is missing field(s): {', '.join(missing)}")
        wrong = [k for k in _REASON_REQUIRED if k in reason and not isinstance(reason[k], str)]
        if wrong:
            flag(i, "reasons_shape", f"field(s) must be strings: {', '.join(wrong)}")
        if all(isinstance(reason.get(k), str) for k in _REASON_TEXT_FIELDS):
            key = "\x1f".join(_duplicate_key(reason[k]) for k in _REASON_TEXT_FIELDS)
            if key in seen_content:
                reason_duplicate_now = True
                flag(
                    i,
                    "duplicate_reason",
                    f"same premise, mechanism and answer_implication as reason#{seen_content[key]} "
                    "(case, whitespace and invisible characters ignored); the grader does not judge "
                    "it, so it can match no target reason",
                )
            else:
                seen_content[key] = i

        projected = {k: reason[k] for k in _REASON_REQUIRED if isinstance(reason.get(k), str)}
        # the grader renumbers reasons r1, r2, r3 before the judge reads them, and measures the
        # text with every URL masked by the 1-byte "#" filler
        sent = {
            k: (mask_reason_uris(v) if k in _REASON_TEXT_FIELDS else v)
            for k, v in projected.items()
        }
        projected_reasons.append({**sent, "reason_id": f"r{i + 1}"})
        reason_trusted.append([])
        reason_over_cap.append(False)
        reason_duplicate.append(False)

        # Deny list: mechanism and answer_implication always; the premise unless it is a
        # whole verbatim quote of a corpus document.
        premise = projected.get("premise", "")
        for field in _REASON_TEXT_FIELDS:
            value = projected.get(field)
            if not value:
                continue
            if field == "premise" and premise_is_corpus_quote(premise, quotable):
                continue
            # the grader scans the folded view, so a disguised phrase ("u\u200bnits/") is refused
            hits = [p for p in DENY_LIST if p in fold_for_detection(value)[0].lower()]
            if "://" in fold_for_detection(mask_reason_uris(value))[0]:
                hits.append("://")  # a "://" the URL detection does not mask
            if hits:
                flag(
                    i,
                    "deny_list",
                    f"{field} contains deny-list phrase(s) {hits}; the grader refuses the unit",
                )

        reason_index.append(i)
        reason_duplicate[-1] = reason_duplicate_now

        if "scope" in reason:
            scope = reason["scope"]
            if not isinstance(scope, dict):
                flag(i, "reasons_shape", "scope must be an object")
            elif "entities" in scope and not (
                isinstance(scope["entities"], list)
                and all(isinstance(e, str) for e in scope["entities"])
            ):
                flag(i, "reasons_shape", "scope.entities must be a list of entity_id strings")

        if "citations" not in reason:
            continue
        citations = reason["citations"]
        if not isinstance(citations, list):
            flag(i, "reasons_shape", "citations must be a list")
            continue
        for j, cit in enumerate(citations):
            where = f"citation #{j}"
            if not isinstance(cit, dict):
                flag(i, "reasons_shape", f"{where} must be an object")
                continue
            absent = [k for k in ("doc_id", "span_start", "span_end") if k not in cit]
            if absent:
                flag(i, "reasons_shape", f"{where} is missing field(s): {', '.join(absent)}")
                continue
            start, end = cit["span_start"], cit["span_end"]
            if not isinstance(cit["doc_id"], str) or not (_is_offset(start) and _is_offset(end)):
                flag(
                    i,
                    "reasons_shape",
                    f"{where} needs a string doc_id and integer offsets >= 0 "
                    f"(got {cit['doc_id']!r}, {start!r}, {end!r})",
                )
                continue

            length = end - start
            if length > MAX_CITATION_CHARS:
                reason_over_cap[-1] = True
                flag(
                    i,
                    "cap_citation_chars",
                    f"{where} spans {length:,} characters (cap {MAX_CITATION_CHARS:,}); "
                    "this reason is not judged and scores 0; cite the passage, not the document",
                )

            if cit["doc_id"] == TASK_DOC_ID:
                flag(
                    i,
                    "reason_citation",
                    f'{where}: doc_id "task" is valid in a claim, not in a reason: the grader resolves '
                    "reason citations against the corpus only, so the judge never sees this passage "
                    "(it already reads the task table's entity list)",
                )
                continue
            # Resolution: the same rules check_answer applies to claims.
            doc = corpus.get(cit["doc_id"])
            if doc is None:
                flag(i, "reason_citation", f"{where}: doc_id {cit['doc_id']!r} is not in the frozen corpus")
                continue
            if doc.doc_date is None or doc.doc_date > cutoff_date:
                flag(
                    i,
                    "reason_citation",
                    f"{where}: doc {doc.doc_id!r} has doc_date={doc.doc_date!r}, after cutoff "
                    f"{cutoff_date!r}; the judge never sees it",
                )
                continue
            if end <= start or end > len(doc.text):
                flag(
                    i,
                    "reason_citation",
                    f"{where}: span [{start}, {end}) does not resolve in a document of "
                    f"length {len(doc.text)}",
                )
                continue
            reason_trusted[-1].append(
                {
                    "doc_id": doc.doc_id,
                    "span_start": start,
                    "span_end": end,
                    "trusted_text": _masked(doc.text[start:end]),
                    "reason_id": f"r{i + 1}",  # the grader's renumbered id
                }
            )

    entity_answer = []
    for row in answer.get("entity_predictions", []) or []:
        if isinstance(row, dict):
            entity_answer.append(
                {"entity_id": row.get("entity_id"), **{k: row[k] for k in _ANSWER_FIELDS if k in row}}
            )
    try:
        answer_size = _compact_bytes(entity_answer)
    except (TypeError, ValueError):
        answer_size = 0  # not JSON-serialisable: the schema findings above already say why
    if answer_size > MAX_ANSWER_BYTES:
        flag(
            -1,
            "cap_answer_bytes",
            f"your per-entity answer comes to {answer_size:,} UTF-8 bytes as compact JSON (cap "
            f"{MAX_ANSWER_BYTES:,} per unit); the unit's reasoning is not judged and scores 0",
        )
    plan = _judged_plan(
        projected_reasons, reason_trusted, reason_over_cap, reason_duplicate, reason_index
    )
    for entry in plan:
        if entry["judged"] or entry["why"] in ("cap_citation_chars", "duplicate_reason"):
            continue  # judged, or already reported above
        flag(entry["index"], entry["why"], entry["message"])
    return findings, plan


def _judged_plan(
    projected: list[dict],
    trusted: list[list[dict]],
    over_cap: list[bool],
    duplicate: list[bool],
    index: list[int],
) -> list[dict]:
    """Which reasons the grader judges, in submitted order (skip and continue)."""
    plan: list[dict] = []
    judged_reasons: list[dict] = []
    judged_evidence: list[dict] = []
    for k, reason in enumerate(projected):
        entry: dict = {"index": index[k], "reason_id": reason.get("reason_id"), "judged": False}
        try:
            reason_bytes = _compact_bytes(judged_reasons + [reason])
            evidence_bytes = _compact_bytes(judged_evidence + trusted[k])
        except (TypeError, ValueError):
            entry.update(why="reasons_shape", message="not JSON-serialisable")
            plan.append(entry)
            continue
        entry.update(reason_bytes=reason_bytes, evidence_bytes=evidence_bytes)
        if duplicate[k]:
            entry.update(why="duplicate_reason", message="a content duplicate of an earlier reason")
        elif over_cap[k]:
            entry.update(why="cap_citation_chars", message="a citation spans over 8,000 characters")
        elif reason_bytes > MAX_REASON_BYTES:
            entry.update(
                why="cap_reason_bytes",
                message=f"with the reasons judged before it, the reasons come to {reason_bytes:,} "
                f"UTF-8 bytes (cap {MAX_REASON_BYTES:,}); this reason is not judged and scores 0, "
                "later reasons are still checked",
            )
        elif evidence_bytes > MAX_EVIDENCE_BYTES:
            entry.update(
                why="cap_evidence_bytes",
                message=f"with the reasons judged before it, the cited evidence comes to "
                f"{evidence_bytes:,} UTF-8 bytes (cap {MAX_EVIDENCE_BYTES:,}); this reason is not "
                "judged and scores 0, later reasons are still checked",
            )
        else:
            entry.update(judged=True, why="", message="judged")
            judged_reasons.append(reason)
            judged_evidence.extend(trusted[k])
        plan.append(entry)
    return plan


def reasons_judged(answer: dict, corpus: dict[str, CorpusDoc], cutoff_date: str) -> list[dict]:
    """The grader's plan for `answer`'s ``submitted_reasons``, one entry per reason in submitted
    order: ``index``, ``judged`` (True or False), ``why`` (the finding code that stops it, or ""),
    ``message``, and the cumulative ``reason_bytes`` / ``evidence_bytes`` measured with it. Run it
    before you write answer.json and put your strongest reason first. Advisory; the grader decides.
    A deny-list hit or an answer over the 3,000-byte cap stops the whole unit's reasoning; see
    `check_submitted_reasons` for those."""
    return _check_reasons(answer, corpus, cutoff_date)[1]


# --------------------------------------------------------------------------- #
# The 5.2.0 deterministic claim rules, with the scorer's own code             #
# --------------------------------------------------------------------------- #

#: A counter of judge tokens, or "auto" (the pinned judge tokenizers when installed), or None.
TokenCounter = Callable[[str], int]


class _DeterministicJudge:
    """What `evaluate_claims` needs from a judge for the deterministic rules only: it exposes no
    `contradiction` (so the NLI question is not asked) and no window (the every-figure rule reads
    the whole cited span anyway); `claim_tokens` only when a tokenizer is available."""

    def __init__(self, counter: TokenCounter | None) -> None:
        if counter is not None:
            self.claim_tokens = counter

    def entail(self, premise: str, hypothesis: str) -> float:  # never asked for a verdict
        return 0.0


def judge_token_counter() -> TokenCounter | None:
    """The judge's claim-length measure (the longest count over the ensemble members, special
    tokens excluded), from tokenizers already on this machine, or None.

    Tries the pinned judge spec (``QFBENCH2_T4_JUDGE_SPEC``, revisions and cache) and then the
    published model ids in the default caches, local files only: nothing is downloaded, and a
    missing tokenizer is reported as such rather than guessed."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    candidates: list[list[dict]] = []
    try:
        import os

        from qfbench2_track_analysis.judge_factory import ENV_JUDGE_CACHE_DIR, load_judge_spec

        spec = load_judge_spec()
        cache = os.environ.get(ENV_JUDGE_CACHE_DIR, spec.cache_dir)
        candidates.append([{"pretrained_model_name_or_path": m, "revision": spec.model_revisions[m],
                            "cache_dir": cache} for m in spec.model_ids])
    except Exception:  # noqa: BLE001 - no spec configured: try the published ids
        pass
    try:
        from faithfulness.judge import DEFAULT_CACHE_DIR, NLI_MODEL_IDS

        candidates.append([{"pretrained_model_name_or_path": m, "cache_dir": DEFAULT_CACHE_DIR}
                           for m in NLI_MODEL_IDS])
        candidates.append([{"pretrained_model_name_or_path": m} for m in NLI_MODEL_IDS])
    except ImportError:
        pass
    for members in candidates:
        try:
            toks = [AutoTokenizer.from_pretrained(local_files_only=True, **kw) for kw in members]
        except Exception:  # noqa: BLE001 - not on this machine
            continue
        return lambda text, _t=toks: max(
            len(t(text, add_special_tokens=False)["input_ids"]) for t in _t
        )
    return None


def check_claim_rules(
    answer: dict, unit_dir: str | Path, *, token_counter: TokenCounter | str | None = "auto"
) -> list[RailFinding]:
    """The analysis scorer's deterministic claim verdicts (scorer 5.2.2) for ``answer``, by the scorer's
    own code (`qfbench2_track_analysis.scoring.evaluate_claims` over the unit as `hydrate` reads
    it), so this check and the scorer cannot disagree. Only the NLI contradiction check is left
    out. Findings, one per false reason of a claim:

    - ``claim_wrong_entity`` -- a citation of a document the unit manifest does not label for
      the claim's entity (nor mark shared), or a ``"task"`` span outside the entity's own row;
    - ``claim_out_of_range`` -- offsets that are not a slice of the cited document;
    - ``claim_malformed`` -- empty, over 4,000 characters, or over 400 judge tokens;
      from scorer 5.2.2 also a claim carrying a claim-level ``citations`` list, a removed shape
      (the scorer does not read the list; the finding's message says so);
    - ``claim_over_cap`` -- a citation spans more than 8,000 characters (false whatever the
      claim states, a verbatim quote included);
    - ``claim_unanchored`` -- a figure no cited span carries (the every-figure rule: whole
      span; a word-for-word quote of a cited span passes; the unit's own entity names and
      tickers and your scored values are exempt);
    - ``claim_content_free`` -- no figure, nothing but function and evidence/meta words, and
      either nothing but function words or a filler word about the evidence ("evidence",
      "passage", "cited", ...).

    Each makes the claim false; `claim_penalty_preview` gives the resulting factor (the soft
    floor: each false claim costs a share of the unit, and other claims beyond 3 x E in total do not
    dilute it). Unit-level findings
    (``claim_index`` -1): ``unit_refused`` -- the scorer refuses the whole unit before any claim
    rule (roster, schema values, an unresolved, undated or post-cutoff citation); and
    ``claim_tokens_unchecked`` -- no judge tokenizer is installed, so the 400-token cap was not
    checked (install ``transformers`` and the judge models, or pass ``token_counter``).

    Needs this repository's ``qfbench2_track_analysis`` (run from the repo root)."""
    counter = judge_token_counter() if token_counter == "auto" else token_counter
    claims, refused, _entities = _claim_report(answer, unit_dir, counter)
    if refused:
        return refused
    findings: list[RailFinding] = []
    position: dict[str, int] = {}
    nested = {
        (str(row.get("entity_id")), i)
        for row in answer.get("entity_predictions", [])
        if isinstance(row, dict)
        for i, claim in enumerate(row.get("claims") or [])
        if isinstance(claim, dict) and NESTED_CITATIONS_KEY in claim
    }
    for verdict in claims.verdicts:
        index = position.get(verdict.entity_id, 0)
        position[verdict.entity_id] = index + 1
        for reason in verdict.reasons:
            if reason == "contradicted":
                continue
            if reason == "malformed" and (verdict.entity_id, index) in nested:
                # The scorer's reason (malformed), with why: not "empty or too long".
                findings.append(RailFinding(
                    verdict.entity_id, index, "claim_malformed", _CITATIONS_TEXT))
                continue
            findings.append(RailFinding(
                verdict.entity_id, index, f"claim_{reason}", _CLAIM_REASON_TEXT[reason]))
    if counter is None:
        findings.append(RailFinding(
            "unit", -1, "claim_tokens_unchecked",
            "needs tokenizer: no judge tokenizer is installed here, so the 400-judge-token claim "
            "cap was not checked"))
    return findings


def _claim_report(answer: dict, unit_dir: str | Path, counter: TokenCounter | None):
    """(the scorer's ClaimReport, [], roster count), or (None, [unit_refused finding], 0)."""
    from qfbench2_track_analysis.alignment import align_predictions
    from qfbench2_track_analysis.codes import T4ParticipantFailure
    from qfbench2_track_analysis.scoring import (
        _entity_admits,
        _entity_bound_citations,
        claim_interval_scored,
        evaluate_claims,
        hydrate,
        unit_entity_names,
    )

    ctx: dict = {"unit_dir": Path(unit_dir)}
    hydrate(ctx)
    params, corpus = ctx["_params"], ctx["_corpus"]
    try:
        aligned = align_predictions(
            answer, ctx["_roster"], target_type=params.target_type,
            interval_level=params.interval_level,
        )
    except T4ParticipantFailure as failure:
        return None, [RailFinding("unit", -1, "unit_refused", f"{failure.reason.value}: {failure}")], 0
    report = corpus.embargo_report(aligned.all_citations(), ctx["_cutoff"])
    if not report.clean:
        return None, [RailFinding(
            "unit", -1, "unit_refused",
            f"{report.violation_count} citation(s) unresolved, undated or post-cutoff: the scorer "
            "refuses the whole unit (see check_answer for which)")], 0
    _entity_bound_citations(corpus, aligned)
    claims = evaluate_claims(
        aligned,
        corpus.lookup(),
        _DeterministicJudge(counter),
        target_type=params.target_type,
        interval_scored=claim_interval_scored(ctx),
        contradiction_bar=float(params.contradiction_bar),
        entity_admits=_entity_admits(corpus),
        entity_names=unit_entity_names(ctx["_task"]),
        interval_level=params.interval_level,
    )
    return claims, [], ctx["_roster"].count


def claim_penalty_preview(
    answer: dict, unit_dir: str | Path, *, token_counter: TokenCounter | str | None = "auto"
) -> dict:
    """The unit's faithfulness factor from the deterministic rules alone, by the scorer's own
    `ClaimReport.penalty_factor` (the soft floor ``1 - F/(F + min(T, 3E))``, E the roster count),
    so it equals the scorer's factor whenever no claim is contradicted. Returns
    ``{"refused": bool, "claims": N, "false": F, "entities": E, "factor": float | None}``."""
    counter = judge_token_counter() if token_counter == "auto" else token_counter
    claims, refused, entities = _claim_report(answer, unit_dir, counter)
    if refused:
        return {"refused": True, "claims": None, "false": None, "entities": None, "factor": None}
    from qfbench2_track_analysis.scoring import DEFAULT_PENALTY_K

    false = sum(1 for v in claims.verdicts if any(r != "contradicted" for r in v.reasons))
    return {
        "refused": False,
        "claims": claims.claim_count,
        "false": false,
        "entities": entities,
        "factor": claims.penalty_factor(DEFAULT_PENALTY_K, entity_count=entities, judge_verdicts=False),
    }


_CLAIM_REASON_TEXT = {
    "wrong_entity": "cites a document (or task row) not labelled for this entity; the claim is false",
    "out_of_range": "a citation's offsets are not a slice of its document; the claim is false",
    "malformed": "empty, over 4,000 characters or over 400 judge tokens; the claim is false",
    "unanchored": "states a figure no cited span carries (every-figure rule); the claim is false",
    "over_cap": "cites a span over 8,000 characters (cite the passage, not the document); the "
    "claim is false",
    "content_free": "states nothing but evidence or meta words with a filler word such as "
    "'evidence' or 'passage', or nothing at all (no figure, no content word); the claim is false",
}
