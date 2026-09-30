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
the unit's own names and tickers exempt, a span over 8,000 characters anchors
no figure). Only the NLI contradiction check is left out.

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
    #   task_row | task_unchecked
    # check_claim_rules: claim_wrong_entity | claim_out_of_range | claim_malformed |
    #   claim_unanchored | claim_tokens_unchecked | unit_refused
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

#: Published caps. Over any of them, the unit's reasoning is not judged and scores 0.
MAX_REASONS = 3
MAX_CITATION_CHARS = 8_000
#: Per-unit caps in UTF-8 bytes of the compact JSON the judge reads (the grader's measure).
#: They sum to MAX_JUDGE_BYTES, so an answer within them never reaches that limit.
MAX_ANSWER_BYTES = 3_000
MAX_REASON_BYTES = 6_500
MAX_EVIDENCE_BYTES = 46_500
MAX_JUDGE_BYTES = MAX_ANSWER_BYTES + MAX_REASON_BYTES + MAX_EVIDENCE_BYTES  # 56,000

#: Published deny list (case-insensitive substrings of participant-written text).
DENY_LIST = (
    "leaderboard",
    "canary",
    "://",
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


#: A URI as the grader masks it: a scheme, "://", and everything up to whitespace or closing
#: punctuation.
_URI = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s,;()\[\]{}<>\"']*")
#: The grader's premise filler: one "#" per masked character.
PREMISE_URI_FILLER = "#"
#: A premise is a quote only with at least this many words once its URLs are masked.
MIN_PREMISE_QUOTE_WORDS = 3


def _mask_premise_uris(text: str) -> str:
    return _URI.sub(lambda m: PREMISE_URI_FILLER * (m.end() - m.start()), text)


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
    - ``cap_citation_chars`` (one citation > 8,000 characters), and three caps counted in
      UTF-8 bytes of the compact JSON the judge reads: ``cap_answer_bytes`` (your per-entity
      answer > 3,000), ``cap_reason_bytes`` (your reasons: id, premise, mechanism,
      answer_implication > 6,500), ``cap_evidence_bytes`` (the resolved cited passages with
      their doc id and offsets > 46,500) -- the unit's reasoning is not judged and scores 0.
      Nothing is clipped. The three sum to the grader's 56,000-byte limit.
    - ``deny_list`` -- a deny-list phrase in ``mechanism`` or ``answer_implication``, or in a
      ``premise`` that is not a verbatim corpus quote (`premise_is_corpus_quote`: at least 3
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
        return []
    findings: list[RailFinding] = []

    def flag(index: int, code: str, message: str) -> None:
        findings.append(RailFinding(_REASONS_ENTITY, index, code, message))

    reasons = answer[_REASONS_ENTITY]
    if not isinstance(reasons, list):
        flag(-1, "reasons_shape", f"submitted_reasons must be a list (got {type(reasons).__name__})")
        return findings
    if not 1 <= len(reasons) <= MAX_REASONS:
        flag(
            -1,
            "reasons_shape",
            f"submitted_reasons must hold 1 to {MAX_REASONS} reasons (got {len(reasons)}); "
            "omit the field to submit none",
        )

    quotable = [_mask_premise_uris(doc.text) for doc in corpus.values()]
    projected_reasons: list[dict] = []
    trusted: list[dict] = []
    seen_content: dict[str, int] = {}

    for i, reason in enumerate(reasons):
        if not isinstance(reason, dict):
            flag(i, "reasons_shape", "a reason must be an object")
            continue
        missing = [k for k in _REASON_REQUIRED if k not in reason]
        if missing:
            flag(i, "reasons_shape", f"reason is missing field(s): {', '.join(missing)}")
        wrong = [k for k in _REASON_REQUIRED if k in reason and not isinstance(reason[k], str)]
        if wrong:
            flag(i, "reasons_shape", f"field(s) must be strings: {', '.join(wrong)}")
        if all(isinstance(reason.get(k), str) for k in _REASON_TEXT_FIELDS):
            key = "\x1f".join(_duplicate_key(reason[k]) for k in _REASON_TEXT_FIELDS)
            if key in seen_content:
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
        # the grader renumbers reasons r1, r2, r3 before the judge reads them
        projected_reasons.append({**projected, "reason_id": f"r{i + 1}"})

        # Deny list: mechanism and answer_implication always; the premise unless it is a
        # whole verbatim quote of a corpus document.
        premise = projected.get("premise", "")
        for field in _REASON_TEXT_FIELDS:
            value = projected.get(field)
            if not value:
                continue
            if field == "premise" and premise_is_corpus_quote(premise, quotable):
                continue
            hits = [p for p in DENY_LIST if p in value.lower()]
            if hits:
                flag(
                    i,
                    "deny_list",
                    f"{field} contains deny-list phrase(s) {hits}; the grader refuses the unit",
                )

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
                flag(
                    i,
                    "cap_citation_chars",
                    f"{where} spans {length:,} characters (cap {MAX_CITATION_CHARS:,}); "
                    "cite the passage, not the document",
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
            trusted.append(
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
    parts = (
        ("cap_answer_bytes", "your per-entity answer", entity_answer, MAX_ANSWER_BYTES),
        ("cap_reason_bytes", "your reasons (id, premise, mechanism, answer_implication)",
         projected_reasons, MAX_REASON_BYTES),
        ("cap_evidence_bytes", "the cited passages the judge reads (with doc id and offsets)",
         trusted, MAX_EVIDENCE_BYTES),
    )
    for code, what, items, cap in parts:
        try:
            size = _compact_bytes(items)
        except (TypeError, ValueError):
            continue  # not JSON-serialisable: the schema findings above already say why
        if size > cap:
            flag(
                -1,
                code,
                f"{what} come to {size:,} UTF-8 bytes as compact JSON (cap {cap:,} per unit); "
                "escaped characters, multi-byte characters and each citation's JSON count",
            )
    return findings


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
    """The analysis scorer's deterministic 5.2.0 claim verdicts for ``answer``, by the scorer's
    own code (`qfbench2_track_analysis.scoring.evaluate_claims` over the unit as `hydrate` reads
    it), so this check and the scorer cannot disagree. Only the NLI contradiction check is left
    out. Findings, one per false reason of a claim:

    - ``claim_wrong_entity`` -- a citation of a document the unit manifest does not label for
      the claim's entity (nor mark shared), or a ``"task"`` span outside the entity's own row;
    - ``claim_out_of_range`` -- offsets that are not a slice of the cited document;
    - ``claim_malformed`` -- empty, over 4,000 characters, or over 400 judge tokens;
    - ``claim_unanchored`` -- a figure no cited span carries (the every-figure rule: whole
      span; a word-for-word quote of a cited span passes at any span length; the unit's own
      entity names and tickers and your scored values are exempt; a span over 8,000 characters
      anchors no figure).

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
    for verdict in claims.verdicts:
        index = position.get(verdict.entity_id, 0)
        position[verdict.entity_id] = index + 1
        for reason in verdict.reasons:
            if reason == "contradicted":
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
}
