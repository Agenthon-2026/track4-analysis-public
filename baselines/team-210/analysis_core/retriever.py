"""Find the passages that actually bear on one entity.

Stdlib BM25 over sentences, with two Track-4-specific twists:

  * entity-aware. The roster gives a ticker and a name; a passage from another
    company's filing is not evidence about this one, however well it matches.
  * target-aware. The task's own prompt supplies the query terms, so a task
    about margin guidance retrieves margin language rather than whatever is
    most frequent.

No model is involved. That is deliberate for the floor: the endpoint may be
slow, metered or refused, and a retrieval layer that only works when the model
answers turns one bad minute into a whole unit lost.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .corpus import Document, sentences

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9%.-]{1,}")
STOP = {"the", "and", "for", "that", "this", "with", "from", "was", "were",
        "has", "have", "had", "its", "our", "their", "will", "are", "been",
        "not", "but", "which", "than", "into", "per", "over", "under"}

K1, B = 1.5, 0.75


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text)
            if t.lower() not in STOP and len(t) > 2]


class Index:
    def __init__(self, documents: list[Document]) -> None:
        self.passages: list[tuple[Document, int, int, str, list[str]]] = []
        for document in documents:
            for start, end, body in sentences(document):
                self.passages.append((document, start, end, body, _tokens(body)))

        self.df = Counter()
        for _, _, _, _, tokens in self.passages:
            self.df.update(set(tokens))
        self.n = max(1, len(self.passages))
        self.avg_len = (sum(len(t) for *_, t in self.passages) / self.n) or 1.0

    def _idf(self, term: str) -> float:
        n_t = self.df.get(term, 0)
        return math.log(1.0 + (self.n - n_t + 0.5) / (n_t + 0.5))

    def search(self, query: str, *, entity_terms: tuple[str, ...] = (),
               top_k: int = 6) -> list[tuple[Document, int, int, str, float]]:
        q = _tokens(query)
        if not q:
            return []
        entity_lower = tuple(t.lower() for t in entity_terms if t)

        scored = []
        for document, start, end, body, tokens in self.passages:
            if not tokens:
                continue
            counts = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for term in q:
                f = counts.get(term, 0)
                if not f:
                    continue
                denominator = f + K1 * (1 - B + B * length / self.avg_len)
                score += self._idf(term) * f * (K1 + 1) / denominator

            # Entity gate. A passage that mentions neither the ticker nor the
            # company is about somebody else. It used to survive with a 0.15
            # penalty - which meant that for an entity with no matching
            # documents, ANOTHER company's filing was cited as evidence about
            # it: the exact faithfulness failure this track punishes with
            # whole-submission ineligibility. Now: matching passages are a
            # HARD filter; the penalised pool is only a last resort when the
            # matching pool is empty.
            haystack = (body + " " + (document.ticker or "") + " " +
                        document.title).lower()
            matches_entity = (not entity_lower
                              or any(t in haystack for t in entity_lower))
            if not matches_entity:
                score *= 0.15

            if score > 0:
                scored.append((document, start, end, body, score,
                               matches_entity))

        matching = [row for row in scored if row[5]]
        if matching:
            scored = matching
        scored = [row[:5] for row in scored]
        scored.sort(key=lambda row: row[4], reverse=True)

        # Diversity: at most two passages per document, so a single verbose
        # filing cannot fill the whole citation list.
        chosen, per_document = [], Counter()
        for row in scored:
            doc_id = row[0].doc_id
            if per_document[doc_id] >= 2:
                continue
            per_document[doc_id] += 1
            chosen.append(row)
            if len(chosen) >= top_k:
                break
        return chosen
