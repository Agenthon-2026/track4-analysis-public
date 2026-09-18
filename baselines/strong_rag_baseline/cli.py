"""``analyze`` CLI for the strong RAG baseline.

Usage — exactly the argv the scoring harness issues (see SUBMISSION_CLI.md)::

    analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json

The leading ``analyze`` is the container command and is accepted here; it is
optional when running the module by hand::

    python -m baselines.strong_rag_baseline.cli \
        --task   units/t4-EXAMPLE-eps-beat/task.json \
        --corpus units/t4-EXAMPLE-eps-beat/corpus \
        --out    /tmp/answer.json

Requires an OpenAI-compatible model server at ``$MODEL_ENDPOINT`` (injected by
the harness at scoring time; locally use ollama/llama.cpp or ``--mock`` for a
network-free smoke run with a canned model reply).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .agent import run_entity
from .client import HTTPModelClient, MockModelClient, ModelClient
from .config import Config
from .formatter import build_answer
from .indexer import build_index
from .retriever import BM25Index

# --------------------------------------------------------------------------- #
# --mock                                                                      #
# --------------------------------------------------------------------------- #
# The mock used to return a fixed reply carrying `"evidence": []`. The pipeline then
# worked exactly as designed -- it grounded the zero quotes it was given -- and wrote an
# answer whose every row had an empty `claims` array. That is not a weak answer: an empty
# or absent `claims` on ANY row fails `g1_schema` for the WHOLE submission, scored
# `t4.schema_invalid` at `W = -0.27`. So the documented smoke command produced the
# worst-scoring artifact the benchmark can express, and reported "0 grounded claims" as
# though that were a neutral fact.
#
# The mock now answers from the prompt it is handed, which is its only channel: it quotes
# a verbatim slice of the first retrieved excerpt, so the quote grounds to a real span
# through the same `find_span` path a real model's quote takes. `--mock` therefore
# exercises retrieval -> prompt -> parse -> ground -> assemble end to end, which is what
# "runnable end to end" was always meant to claim.
#
# It is still a stub and does not predict: `point_forecast` is 0.0, because a number a stub
# invented is not a forecast and pretending otherwise reads as one. The single exception is
# RANKING units, where the scorer reads `point_forecast` and nothing else, so an identical
# vector is the degenerate answer whose score depends on the sealed roster order rather than
# on the prediction (public #47). There the stub echoes one of the entity's own features --
# not because the number means anything, but so that no shipped code here models that shape.
# Echoing a feature on the other target types would be worse than 0.0, not better: on the
# EPS exemplar the first numeric feature is `mktcap_bn`, and emitting 2650.0 as a forecast
# of a ~1.50 EPS looks like a real prediction that is badly wrong, rather than an obvious
# placeholder.

#: `[1] doc_id=... (doc_date=...)` followed by the excerpt in triple quotes, as
#: `prompts.build_user_prompt` emits it.
_EXCERPT_RE = re.compile(r'\[\d+\] doc_id=(\S+) \(doc_date=[^)]*\)\n"""(.*?)"""', re.DOTALL)
_ALLOWED_LABELS_RE = re.compile(r"^ALLOWED LABELS: (.+)$", re.MULTILINE)
_TARGET_TYPE_RE = re.compile(r"^TARGET: .*\((\w+)\)$", re.MULTILINE)
_ENTITY_NUMERIC_RE = re.compile(r"^  \w+: (-?\d+(?:\.\d+)?)$", re.MULTILINE)

#: Quote budget. Long enough that the NLI judge gets a real premise rather than a
#: fragment, short enough to stay inside one excerpt.
_QUOTE_CHARS = 200


def _verbatim_quote(excerpt: str) -> str:
    """A prefix of ``excerpt``, trimmed at a word boundary so it stays an exact substring.

    Exactness is the whole contract: `span_finder.find_span` locates the quote by
    `doc_text.find(quote)`, and an excerpt is itself a slice of the document, so any slice
    of the excerpt resolves. Trimming mid-word would still resolve; trimming at whitespace
    just makes the emitted claim readable.
    """
    text = excerpt.strip()
    if len(text) <= _QUOTE_CHARS:
        return text
    cut = text[:_QUOTE_CHARS]
    spaced = cut.rsplit(" ", 1)[0]
    return spaced if spaced else cut


def _mock_reply(system: str, user: str) -> str:
    """Answer the prompt from the prompt: quote the first excerpt it offers.

    Returns the same JSON shape `prompts.build_user_prompt` asks a real model for, so it
    travels the identical parse-and-ground path. With no excerpt to quote -- a unit where
    retrieval returned nothing -- it emits no evidence, which is the honest answer and
    still the one that fails `g1_schema`. That is a property of the unit, not of the mock.
    """
    del system  # the stub does not read its instructions
    match = _EXCERPT_RE.search(user)
    evidence = []
    if match is not None:
        doc_id, excerpt = match.group(1), match.group(2)
        quote = _verbatim_quote(excerpt)
        if quote:
            evidence = [
                {
                    "doc_id": doc_id,
                    "quote": quote,
                    "claim": (
                        f"Mock baseline: the cited passage from {doc_id} is the "
                        "top-ranked pre-cutoff excerpt retrieved for this entity."
                    ),
                }
            ]

    labels_match = _ALLOWED_LABELS_RE.search(user)
    label = labels_match.group(1).split(",")[0].strip() if labels_match else None

    target_match = _TARGET_TYPE_RE.search(user)
    is_ranking = target_match is not None and target_match.group(1) == "ranking"
    numbers = _ENTITY_NUMERIC_RE.findall(user) if is_ranking else []
    point = float(numbers[0]) if numbers else 0.0

    half = max(abs(point) * 0.5, 1.0)
    return json.dumps(
        {
            "label": label,
            "point_forecast": point,
            "interval": {"level": 0.90, "lo": point - half, "hi": point + half},
            "evidence": evidence,
        }
    )


def run(
    task_path: Path,
    corpus_dir: Path,
    out_path: Path,
    client: ModelClient,
    top_k: int,
) -> dict:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    corpus = build_index(corpus_dir)
    index = BM25Index(corpus.chunks, task["cutoff_date"])
    results = [
        run_entity(task, entity, index, corpus, client, top_k)
        for entity in task.get("entities", [])
    ]
    answer = build_answer(task, results, corpus)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(answer, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # The harness passes the verb as the container command, so it arrives as argv[0].
    # Optional, so hand-invocation without it keeps working. See baseline_agent/cli.py.
    parser.add_argument("verb", nargs="?", default="analyze", choices=["analyze"])
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a canned model reply (no network) — wiring smoke runs only.",
    )
    args = parser.parse_args(argv)

    config = Config.from_env()
    client: ModelClient = (
        MockModelClient(reply=_mock_reply) if args.mock else HTTPModelClient(config)
    )
    answer = run(args.task, args.corpus, args.out, client, config.top_k)
    n_claims = sum(len(e["claims"]) for e in answer["entity_predictions"])
    print(
        f"wrote {args.out} — {len(answer['entity_predictions'])} entities, "
        f"{n_claims} grounded claims"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
