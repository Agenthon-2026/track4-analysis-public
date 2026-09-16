"""Run the participant baseline, then assess its output in a separate process context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .artifacts import digest as artifact_digest
from .dataset import REPO, load_cases, public_roots, require_external, stage_inputs
from .engineering import (
    LOCAL_CPUS,
    LOCAL_MEMORY_BYTES,
    contract,
    image_identity,
    make_run_plan,
    verify_run_plan,
)
from .report import markdown, summarize


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def provenance() -> dict:
    import qfbench2_common as common
    from qfbench2_track_analysis.scoring import SCORER_VERSION

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    source = hashlib.sha256()
    for folder in ("baselines", "qfbench2_track_analysis", "faithfulness"):
        for path in sorted((REPO / folder).rglob("*.py")):
            if "__pycache__" not in path.parts:
                source.update(
                    path.relative_to(REPO).as_posix().encode()
                    + b"\0"
                    + path.read_bytes()
                )
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_digest": source.hexdigest(),
        "python": platform.python_version(),
        "scorer_version": SCORER_VERSION,
        "toolkit_build": common.build_identity(),
        "toolkit_build_verified": common.verify_build_identity(),
        "toolkit_source_digest": common.package_tree_digest(),
    }


def run_local(
    stage: Path, output: Path, *, mode: str, seed: int, timeout: float
) -> dict:
    argv = [
        sys.executable,
        str(REPO / "baselines/analyze.py"),
        "analyze",
        "--task",
        str(stage / "task.json"),
        "--corpus",
        str(stage / "corpus"),
        "--out",
        str(output / "answer.json"),
        "--diagnostics",
        str(output / "diagnostics.json"),
    ]
    if mode == "grounded":
        argv.append("--mock")
    env = dict(os.environ, QFBENCH_SEED=str(seed), T4_LOCAL_LLAMA="0")
    if mode == "grounded":
        for key in ("MODEL_ENDPOINT", "MODEL_TOKEN", "MODEL_NAME"):
            env.pop(key, None)
    try:
        proc = subprocess.run(
            argv,
            cwd=stage,
            env=env,
            timeout=timeout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {
            "returncode": proc.returncode,
            "timed_out": False,
            "isolation": "local-process",
        }
    except subprocess.TimeoutExpired:
        return {"returncode": 124, "timed_out": True, "isolation": "local-process"}


def run_container(
    stage: Path,
    output: Path,
    *,
    image: str,
    seed: int,
    timeout: float,
    exercise: tuple[Path, str] | None = None,
) -> dict:
    def docker(*args, limit=60, check=True, quiet=False):
        return subprocess.run(
            ["docker", *args],
            stdout=subprocess.DEVNULL if quiet else subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=check,
            timeout=limit,
        )

    command = [
        "analyze",
        "--task",
        "/input/task.json",
        "--corpus",
        "/input/corpus",
        "--out",
        "/output/answer.json",
        "--diagnostics",
        "/output/diagnostics.json",
        "--mock",
    ]
    entrypoint = []
    if exercise is not None:
        entrypoint = ["--entrypoint", "python"]
        command = ["/exercise.py", exercise[1]]
    container = docker(
        "create",
        "--platform",
        "linux/amd64",
        "--network=none",
        f"--cpus={LOCAL_CPUS}",
        f"--memory={LOCAL_MEMORY_BYTES}",
        "-e",
        f"QFBENCH_SEED={seed}",
        *entrypoint,
        image,
        *command,
    ).stdout.strip()
    try:
        # Precreate the output directory so a missing answer is scored as a failed case.
        with tempfile.TemporaryDirectory(prefix="t4-empty-output-") as empty:
            docker("cp", empty, f"{container}:/output")
        docker("cp", str(stage), f"{container}:/input")
        if exercise is not None:
            docker("cp", str(exercise[0]), f"{container}:/exercise.py")
        started = time.monotonic()
        try:
            docker(
                "start", "--attach", container, limit=timeout, check=False, quiet=True
            )
        except subprocess.TimeoutExpired:
            return {
                "returncode": 124,
                "timed_out": True,
                "isolation": "docker-network-none",
                "limits": {"cpus": LOCAL_CPUS, "memory_bytes": LOCAL_MEMORY_BYTES},
            }
        process_elapsed = time.monotonic() - started
        info = json.loads(docker("inspect", container).stdout)[0]
        state = info["State"]
        host = info["HostConfig"]
        observation = {
            "image_id": info["Image"],
            "cpus": host["NanoCpus"] / 10**9,
            "memory_bytes": host["Memory"],
            "network": host["NetworkMode"],
            "oom_killed": state["OOMKilled"],
        }
        if state["Status"] != "exited" or state.get("Error"):
            raise RuntimeError(
                "evaluation container did not complete: Docker runtime failure"
            )
        code = int(state["ExitCode"])
        artifact_errors = []
        if code == 0:
            # Participant files cannot replace evaluator records or link to host data.
            with tempfile.TemporaryDirectory(prefix="t4-container-output-") as temp:
                docker("cp", f"{container}:/output/.", temp)
                names = ("answer.json", "diagnostics.json")
                if exercise is not None:
                    names += ("exercise.json",)
                for name in names:
                    source = Path(temp) / name
                    if source.is_symlink() or (
                        source.exists() and not source.is_file()
                    ):
                        artifact_errors.append(f"{name}: expected a regular file")
                    elif source.is_file():
                        shutil.copyfile(source, output / name)
        return {
            "returncode": code,
            "timed_out": False,
            "isolation": "docker-network-none",
            "limits": {"cpus": LOCAL_CPUS, "memory_bytes": LOCAL_MEMORY_BYTES},
            "artifact_errors": artifact_errors,
            "process_elapsed_s": process_elapsed,
            "resource_observation": observation,
        }
    finally:
        docker("rm", "--force", container)


def evidence_ledger(stage: Path, *, model: bool = False) -> dict:
    from baselines.strong_rag_baseline.config import Config
    from baselines.strong_rag_baseline.evidence import prepare_evidence
    from baselines.strong_rag_baseline.indexer import build_index
    from baselines.strong_rag_baseline.retriever import BM25Index

    task = json.loads((stage / "task.json").read_text(encoding="utf-8"))
    corpus = build_index(stage / "corpus")
    index = BM25Index(corpus.chunks, task["cutoff_date"])
    top_k = Config.from_env().top_k if model else 8
    return {
        "task_id": task["task_id"],
        "used_for_prediction": model,
        "entities": [
            prepare_evidence(
                task,
                entity,
                index,
                corpus,
                top_k=top_k,
            ).ledger
            for entity in task["entities"]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--units", type=Path, help="Directory of public practice units")
    source.add_argument(
        "--manifest", type=Path, help="External private development roster"
    )
    ap.add_argument(
        "--out", type=Path, required=True, help="New directory outside public worktrees"
    )
    ap.add_argument("--mode", choices=("grounded", "model"), default="grounded")
    ap.add_argument("--profile", choices=("smoke", "production"), default="smoke")
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260731])
    ap.add_argument("--timeout", type=float, default=480)
    ap.add_argument(
        "--image", help="Run the grounded baseline in this local Docker image"
    )
    args = ap.parse_args(argv)
    if not 0 < args.timeout <= 600:
        ap.error("timeout must be in (0, 600] seconds")
    if len(set(args.seeds)) != len(args.seeds) or any(
        not 0 <= seed < 2**32 for seed in args.seeds
    ):
        ap.error("seeds must be distinct unsigned 32-bit integers")
    if args.mode == "model" and (args.image or not os.environ.get("MODEL_ENDPOINT")):
        ap.error(
            "model evaluation requires a configured local MODEL_ENDPOINT; container mode is offline only"
        )
    try:
        require_external(args.out, public_roots())
        cases = load_cases(units=args.units, manifest=args.manifest)
        if args.profile == "production" and any(
            case.truth_path is None for case in cases
        ):
            raise ValueError(
                "production evaluation requires external truth for every case"
            )
        from qfbench2_common import manifest, taskcard
        from .assess import assess_unit

        for case in cases:
            _, errors = taskcard.load_and_validate(case.unit_dir)
            errors += manifest.verify_manifest(case.unit_dir)
            if errors:
                raise ValueError(f"invalid input card or manifest for {case.case_id}")
        if args.profile == "production":
            from qfbench2_track_analysis.scoring import build_verifier

            build_verifier(
                {}
            )  # Refuse unavailable production scoring before running an agent.
        identity = provenance()
        from baselines.strong_rag_baseline.config import Config

        identity["prediction_settings"] = {
            "mode": args.mode,
            "top_k": Config.from_env().top_k,
        }
        image = None
        if args.image:
            identity.update(image_identity(args.image))
            image = identity["image_id"]
        plan = make_run_plan(
            cases,
            args.seeds,
            identity=identity,
            timeout=args.timeout,
            mode=args.mode,
            profile=args.profile,
        )
        planned = {(row["case_id"], row["seed"]): row for row in plan["runs"]}
        args.out = args.out.resolve()
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / "run-plan.json", plan)
        rows = []
        for case in cases:
            for seed in args.seeds:
                frozen = planned[(case.case_id, seed)]
                output = args.out / case.case_id / f"seed-{seed}"
                output.mkdir(parents=True)
                started = time.monotonic()
                with tempfile.TemporaryDirectory(prefix="t4-evaluation-input-") as temp:
                    stage = Path(temp) / "input"
                    digest = stage_inputs(case.unit_dir, stage)
                    if (
                        digest != frozen["input_digest"]
                        or contract(case.unit_dir) != frozen["resource_contract"]
                    ):
                        raise ValueError(
                            "prediction inputs or resource contract changed after run plan was frozen"
                        )
                    evidence = evidence_ledger(stage, model=args.mode == "model")
                    write_json(output / "evidence.json", evidence)
                    preparation_elapsed = time.monotonic() - started
                    started = time.monotonic()
                    if image:
                        execution = run_container(
                            stage,
                            output,
                            image=image,
                            seed=seed,
                            timeout=frozen["timeout_s"],
                        )
                    else:
                        execution = run_local(
                            stage,
                            output,
                            mode=args.mode,
                            seed=seed,
                            timeout=frozen["timeout_s"],
                        )
                elapsed = time.monotonic() - started
                with tempfile.TemporaryDirectory(prefix="t4-verify-input-") as temp:
                    if stage_inputs(case.unit_dir, Path(temp) / "input") != digest:
                        raise ValueError("prediction inputs changed during execution")
                # Truth paths are never passed to the predictor and opened only after it exits.
                truth_text = (
                    case.truth_path.read_text(encoding="utf-8")
                    if case.truth_path
                    else None
                )
                realized = json.loads(truth_text) if truth_text is not None else None
                assessment_output = output
                if execution["returncode"] != 0:
                    assessment_output = output / "failed-execution"
                    assessment_output.mkdir()
                assessment = assess_unit(
                    case.unit_dir,
                    assessment_output,
                    realized=realized,
                    profile=args.profile,
                )
                if contract(case.unit_dir) != frozen["resource_contract"]:
                    raise ValueError("resource contract changed during prediction")
                row = {
                    **frozen,
                    "case_id": case.case_id,
                    "split": case.split,
                    "group": case.group,
                    "domain": case.domain,
                    "seed": seed,
                    "profile": args.profile,
                    "input_digest": digest,
                    "truth_digest": hashlib.sha256(truth_text.encode()).hexdigest()
                    if truth_text is not None
                    else None,
                    "answer_sha256": hashlib.sha256(
                        (output / "answer.json").read_bytes()
                    ).hexdigest()
                    if (output / "answer.json").is_file()
                    else None,
                    "elapsed_s": elapsed,
                    "preparation_elapsed_s": preparation_elapsed,
                    "evidence_stats": {
                        "entities": len(evidence["entities"]),
                        "without_candidates": sum(
                            not entity["records"] for entity in evidence["entities"]
                        ),
                        "candidate_spans": sum(
                            len(entity["records"]) for entity in evidence["entities"]
                        ),
                    },
                    "execution": execution,
                    "assessment": assessment,
                    "status": "completed",
                }
                diagnostics_path = output / "diagnostics.json"
                if diagnostics_path.is_file():
                    row["diagnostics_sha256"] = hashlib.sha256(
                        diagnostics_path.read_bytes()
                    ).hexdigest()
                    row["diagnostics"] = json.loads(
                        diagnostics_path.read_text(encoding="utf-8")
                    )
                write_json(output / "result.json", row)
                rows.append(row)
                print(
                    f"{case.case_id} seed={seed}: exit={execution['returncode']} "
                    f"{args.profile}_admissible={assessment['admissible']}",
                    flush=True,
                )
        report = {
            "schema_version": 1,
            "mode": args.mode,
            "profile": args.profile,
            "provenance": identity,
            "run_plan": plan,
            "run_plan_digest": artifact_digest(plan),
            "runs": rows,
            "summary": summarize(rows),
            "by_split": {
                split: summarize([row for row in rows if row["split"] == split])
                for split in sorted({row["split"] for row in rows})
            },
        }
        verify_run_plan(report)
        write_json(args.out / "report.json", report)
        (args.out / "report.md").write_text(markdown(report), encoding="utf-8")
        print(f"Report: {args.out / 'report.md'}")
        return (
            0
            if all(
                row["execution"]["returncode"] == 0 and row["assessment"]["admissible"]
                for row in rows
            )
            else 1
        )
    except Exception as exc:
        print(f"Evaluation aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
