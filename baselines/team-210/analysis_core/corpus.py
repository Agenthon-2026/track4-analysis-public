"""Load the frozen evidence corpus, embargoed.

The embargo is the whole trust proposition of Track 4. A claim cited to a
document dated after the task's cutoff is not weak evidence, it is a look at the
answer, and the faithfulness gate treats it as such. So documents past the
cutoff are dropped at load time rather than filtered later - code that never
sees them cannot cite them.

Spans are byte-ish offsets into `text`. A citation whose span does not resolve
to the text it claims is unfaithful even when the underlying belief is right, so
the loader keeps the raw text and every span is checked against it before the
answer is written.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    doc_date: dt.date | None
    ticker: str | None
    title: str
    text: str

    def span(self, start: int, end: int) -> str:
        return self.text[start:end]


def _parse_date(value) -> dt.date | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def load(corpus_dir: pathlib.Path, cutoff: str) -> tuple[list[Document], list[str]]:
    """Returns (admissible documents, embargoed doc_ids)."""
    limit = _parse_date(cutoff)
    documents: list[Document] = []
    embargoed: list[str] = []

    for path in sorted(corpus_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except ValueError:
            continue
        if not isinstance(raw, dict) or "text" not in raw:
            continue
        doc_id = str(raw.get("doc_id") or path.stem)
        date = _parse_date(raw.get("doc_date") or raw.get("filed_date"))
        if limit is not None and date is not None and date > limit:
            embargoed.append(doc_id)
            continue
        documents.append(Document(
            doc_id=doc_id,
            doc_date=date,
            ticker=raw.get("ticker"),
            title=str(raw.get("title") or ""),
            text=str(raw["text"]),
        ))
    return documents, embargoed


SENTENCE = re.compile(r"[^.!?]+[.!?]")


def sentences(document: Document, min_chars: int = 60) -> list[tuple[int, int, str]]:
    """(start, end, text) for each sentence, with offsets into the raw text.

    Offsets come from the match, never from re-finding the string afterwards:
    a sentence that occurs twice would otherwise be cited at the wrong place and
    fail span verification for a reason that looks like a retrieval bug.
    """
    out = []
    for match in SENTENCE.finditer(document.text):
        body = match.group().strip()
        if len(body) >= min_chars:
            out.append((match.start(), match.end(), body))
    if not out and document.text.strip():
        out.append((0, min(len(document.text), 600), document.text[:600].strip()))
    return out
