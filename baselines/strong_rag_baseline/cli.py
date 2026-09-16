"""``analyze`` CLI for the strong RAG baseline.

Usage — exactly the argv the scoring harness issues (see SUBMISSION_CLI.md)::

    analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json

The leading ``analyze`` is the container command and is accepted here; it is
optional when running the module by hand::

    python -m baselines.strong_rag_baseline.cli \
        --task   units/t4-EXAMPLE-eps-beat/task.json \
        --corpus units/t4-EXAMPLE-eps-beat/corpus \
        --out    /tmp/answer.json

Official ``analyze`` uses ``MODEL_ENDPOINT`` and ``MODEL_NAME`` when supplied
by the harness. Missing endpoints and invalid model replies use the grounded
reasoner. ``--local-llama`` is an explicit developer-machine experiment;
``--mock`` forces the network-free reasoner.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from .agent import run_entity, run_entity_grounded
from .client import HTTPModelClient, ModelClient
from .config import Config
from .formatter import build_answer
from .indexer import build_index
from .local_server import LocalLlamaServer, try_start_local_server
from .retriever import BM25Index


def _validate_diagnostics_path(
    diagnostics_path: Path, task_path: Path, corpus_dir: Path, out_path: Path
) -> None:
    resolved = diagnostics_path.resolve()
    corpus_root = corpus_dir.resolve()
    if resolved == corpus_root or corpus_root in resolved.parents:
        raise ValueError("diagnostics path must be outside the input corpus")
    if diagnostics_path.is_dir():
        raise ValueError("diagnostics path must be a file")
    protected = [task_path, out_path]
    protected.extend(path for path in corpus_dir.rglob("*") if path.is_file())
    for path in protected:
        if resolved == path.resolve() or (
            diagnostics_path.exists()
            and path.exists()
            and diagnostics_path.samefile(path)
        ):
            raise ValueError(
                "diagnostics path must differ from all inputs and the answer"
            )


def _write_diagnostics(path: Path, diagnostics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(
                diagnostics, output, indent=2, ensure_ascii=False, allow_nan=False
            )
            output.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(
    task_path: Path,
    corpus_dir: Path,
    out_path: Path,
    client: ModelClient | None,
    top_k: int,
    *,
    grounded: bool = False,
    diagnostics_path: Path | None = None,
) -> dict:
    if diagnostics_path is not None:
        _validate_diagnostics_path(diagnostics_path, task_path, corpus_dir, out_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    corpus = build_index(corpus_dir)
    index = BM25Index(corpus.chunks, task["cutoff_date"])
    results = []
    diagnostics: dict = {
        "schema_version": "1",
        "task_id": task.get("task_id", ""),
        "mode": "grounded" if grounded or client is None else "model",
        "entities": [],
    }
    for entity in task.get("entities", []):
        started = time.monotonic()
        if grounded or client is None:
            result = run_entity_grounded(task, entity, index, corpus, top_k)
            result.fallback_reason = "forced_grounded" if grounded else "no_endpoint"
        else:
            result = run_entity(task, entity, index, corpus, client, top_k)
        elapsed_s = max(0.0, time.monotonic() - started)
        results.append(result)
        diagnostics["entities"].append(
            {
                "entity_id": entity.get("entity_id", ""),
                "source": result.source,
                "fallback_reason": result.fallback_reason,
                "elapsed_s": elapsed_s,
                "dropped_claims": result.dropped_claims,
            }
        )
    answer = build_answer(task, results, corpus)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(answer, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if diagnostics_path is not None:
        ledger = getattr(client, "request_ledger", None)
        diagnostics["request_ledger"] = (
            ledger()
            if callable(ledger) and not grounded
            else {"version": 1, "source": "offline", "attempts_used": 0, "attempts": []}
            if grounded or client is None
            else None
        )
        _write_diagnostics(diagnostics_path, diagnostics)
    return answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # The harness passes the verb as the container command, so it arrives as argv[0].
    # Optional, so hand-invocation without it keeps working. See baseline_agent/cli.py.
    parser.add_argument("verb", nargs="?", default="analyze", choices=["analyze"])
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run the grounded reasoner without a model call.",
    )
    parser.add_argument(
        "--local-llama",
        action="store_true",
        help=(
            "Developer-machine only: start llama.cpp on 127.0.0.1 against a "
            "local GGUF. Default OFF."
        ),
    )
    args = parser.parse_args(argv)
    if args.diagnostics is not None:
        try:
            _validate_diagnostics_path(
                args.diagnostics, args.task, args.corpus, args.out
            )
        except ValueError as exc:
            parser.error(str(exc))

    config = Config.from_env()
    deadline = time.monotonic() + config.unit_timeout_s
    server: LocalLlamaServer | None = None
    client: ModelClient | None = None
    want_local = (not args.mock) and (bool(args.local_llama) or config.local_llama)
    use_grounded = bool(args.mock) or not (config.model_endpoint or want_local)
    if want_local:
        server = try_start_local_server(
            host=config.local_host,
            port=config.local_port,
            ctx=config.local_ctx,
            startup_s=config.local_startup_s,
        )
        if server is None:
            use_grounded = True
        else:
            client = HTTPModelClient(
                config, base_url=server.base_url, deadline=deadline
            )
            use_grounded = False
    elif not use_grounded:
        client = HTTPModelClient(config, deadline=deadline)
    try:
        answer = run(
            args.task,
            args.corpus,
            args.out,
            client,
            config.top_k,
            grounded=bool(args.mock),
            diagnostics_path=args.diagnostics,
        )
    finally:
        if server is not None:
            server.stop()
    n_claims = sum(len(e["claims"]) for e in answer["entity_predictions"])
    mode = (
        "extract-then-predict"
        if use_grounded
        else "local-llama"
        if want_local
        else "house-model"
    )
    print(
        f"wrote {args.out} — {len(answer['entity_predictions'])} entities, "
        f"{n_claims} grounded claims ({mode})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
