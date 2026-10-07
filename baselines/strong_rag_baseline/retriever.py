"""Embargo-aware BM25 retriever over span-level chunks.

BM25 (Okapi, k1=1.5, b=0.75) implemented in pure Python for determinism and
zero dependencies. Dense retrieval is deliberately omitted: the eval sandbox's
restricted network cannot fetch embedding weights at run time, so a lexical index
keeps the baseline reproducible everywhere. Under docs/ARTIFACT-POLICY.md the only
embedding weights a dense index may use are NeMo Retriever embedding models baked
into the image at build time; no other embedding model is permitted. Ties
break by (doc_id, span_start) so ranking is stable across runs and platforms.

Documents whose ``doc_date`` is missing or after the task ``cutoff_date`` are
dropped BEFORE scoring — post-cutoff evidence never reaches the reasoning step
(the stale-filing traps target exactly this mistake).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .indexer import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
_K1 = 1.5
_B = 0.75


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class BM25Index:
    """Okapi BM25 over a fixed chunk list (embargo applied at build time)."""

    def __init__(self, chunks: list[Chunk], cutoff_date: str) -> None:
        self.chunks = [
            c
            for c in chunks
            if c.doc_date is not None and c.doc_date <= cutoff_date
        ]
        self._chunk_tokens = [_tokens(c.text) for c in self.chunks]
        self._doc_freq: dict[str, int] = {}
        for toks in self._chunk_tokens:
            for t in set(toks):
                self._doc_freq[t] = self._doc_freq.get(t, 0) + 1
        self._avg_len = (
            sum(len(t) for t in self._chunk_tokens) / len(self._chunk_tokens)
            if self._chunk_tokens
            else 0.0
        )

    def _idf(self, term: str) -> float:
        n, df = len(self.chunks), self._doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        q_terms = _tokens(query)
        if not q_terms or not self.chunks:
            return []
        scored: list[ScoredChunk] = []
        for chunk, toks in zip(self.chunks, self._chunk_tokens):
            if not toks:
                continue
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            norm = _K1 * (1 - _B + _B * len(toks) / self._avg_len)
            score = sum(
                self._idf(t) * tf[t] * (_K1 + 1) / (tf[t] + norm)
                for t in set(q_terms)
                if t in tf
            )
            if score > 0:
                scored.append(ScoredChunk(chunk=chunk, score=score))
        scored.sort(
            key=lambda s: (-s.score, s.chunk.doc_id, s.chunk.span_start)
        )
        return scored[:top_k]
