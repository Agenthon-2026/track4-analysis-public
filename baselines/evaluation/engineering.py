"""Freeze engineering run plans and bind local Docker observations to source bytes.

These records cover the local CPU/memory/network envelope. They do not certify
organiser hardware equivalence, remote model availability or proxy billing.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path

from .artifacts import digest, read_json_inside
from .dataset import REPO, stage_inputs

LOCAL_CPUS = 1
LOCAL_MEMORY_BYTES = 1024**3


def memory_bytes(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([KMGT]?)B?", value.upper())
    if not match:
        raise ValueError("unsupported card memory format")
    return int(match[1]) * 1024 ** ("KMGT".index(match[2]) + 1 if match[2] else 0)


def contract(unit: Path) -> dict:
    payload = (unit / "card.toml").read_bytes()
    card = tomllib.loads(payload.decode())
    result = {
        "card_sha256": hashlib.sha256(payload).hexdigest(),
        "timeout_s": card.get("agent", {}).get("timeout_sec"),
        "cpus": card["environment"]["cpus"],
        "memory_bytes": memory_bytes(card["environment"]["memory"]),
        "network": card["environment"]["network"],
    }
    if any(
        result[k] is not None
        and (
            isinstance(result[k], bool)
            or not isinstance(result[k], (int, float))
            or not math.isfinite(result[k])
            or result[k] <= 0
        )
        for k in ("timeout_s", "cpus", "memory_bytes")
    ):
        raise ValueError("invalid card resource contract")
    return result


def runtime_manifest(root: Path) -> dict:
    names = ["analyze.py", "requirements.txt"]
    names += [
        f"strong_rag_baseline/{p.name}"
        for p in sorted((root / "strong_rag_baseline").glob("*.py"))
    ]
    if len(names) < 3:
        raise ValueError("runtime contains no agent package")
    result = {}
    for name in names:
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError(
                "runtime source must be regular files within the image app"
            )
        if path.stat().st_size > 1024**2:
            raise ValueError("runtime source file exceeds evidence size bound")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def image_identity(image: str) -> dict:
    def docker(*args):
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, check=True, timeout=60
        ).stdout.strip()

    info = json.loads(docker("image", "inspect", image))[0]
    if info["Architecture"] != "amd64":
        raise ValueError("evaluation image must be linux/amd64")
    config = info["Config"]
    if (
        config.get("Entrypoint") != ["python", "analyze.py"]
        or config.get("WorkingDir") != "/app"
    ):
        raise ValueError("image must expose the baseline analyze entrypoint")
    if config.get("Labels", {}).get("qfbench2.interface_version") != "2.0":
        raise ValueError("image lacks the interface label")
    container = docker(
        "create", "--platform", "linux/amd64", "--network=none", info["Id"]
    )
    try:
        with tempfile.TemporaryDirectory(prefix="t4-image-evidence-") as temp:
            root = Path(temp)
            docker("cp", f"{container}:/app/.", str(root))
            actual = runtime_manifest(root)
    finally:
        docker("rm", "--force", container)
    expected = runtime_manifest(REPO / "baselines")
    if actual != expected:
        raise ValueError(
            "image runtime source differs from the evaluator checkout; rebuild the image"
        )
    return {
        "image_id": info["Id"],
        "runtime_manifest": actual,
        "runtime_digest": digest(actual),
        "architecture": "amd64",
        "entrypoint": config["Entrypoint"],
        "working_dir": config["WorkingDir"],
        "interface_version": config["Labels"]["qfbench2.interface_version"],
    }


def make_run_plan(
    cases: list,
    seeds: list[int],
    *,
    identity: dict,
    timeout: float,
    mode: str,
    profile: str,
) -> dict:
    rows = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="t4-plan-input-") as temp:
            input_digest = stage_inputs(case.unit_dir, Path(temp) / "input")
        task = json.loads((case.unit_dir / "task.json").read_text())
        resources = contract(case.unit_dir)
        for seed in seeds:
            rows.append(
                {
                    "case_id": case.case_id,
                    "seed": seed,
                    "input_digest": input_digest,
                    "group": case.group,
                    "domain": case.domain,
                    "split": case.split,
                    "target_type": task["target"]["type"],
                    "resource_contract": resources,
                    "timeout_s": min(timeout, resources["timeout_s"] or timeout),
                }
            )
    return {
        "version": 1,
        "provenance": identity,
        "mode": mode,
        "profile": profile,
        "runs": rows,
    }


def verify_run_plan(report: dict) -> None:
    plan = report["run_plan"]
    if plan.get("version") != 1:
        raise ValueError("unsupported run plan version")
    if digest(plan) != report["run_plan_digest"]:
        raise ValueError("run plan digest differs")
    if any(plan[key] != report[key] for key in ("provenance", "mode", "profile")):
        raise ValueError("run plan version or execution mode differs from report")
    expected = {(r["case_id"], r["seed"]): r for r in plan["runs"]}
    rows = {(r["case_id"], r["seed"]): r for r in report["runs"]}
    if (
        not expected
        or len(expected) != len(plan["runs"])
        or len(rows) != len(report["runs"])
        or set(expected) != set(rows)
    ):
        raise ValueError("report does not cover its entire frozen run plan")
    for key, frozen in expected.items():
        if any(rows[key].get(name) != value for name, value in frozen.items()):
            raise ValueError("run input or resource contract differs from frozen plan")


def verify_plan_file(path: Path, report: dict) -> None:
    if "run_plan" in report:
        if read_json_inside(path.parent, "run-plan.json") != report["run_plan"]:
            raise ValueError("persisted run plan differs from report")
        verify_run_plan(report)


def resource_observations(
    rows: list[dict], p95_fraction: float, image_id: str | None = None
) -> tuple[bool | None, dict]:
    ratios = []
    missing = 0
    failures = []
    for row in rows:
        cap = row.get("resource_contract")
        observed = row.get("execution", {}).get("resource_observation")
        if cap is None or cap.get("timeout_s") is None or observed is None:
            missing += 1
            continue
        elapsed = row.get("execution", {}).get("process_elapsed_s")
        values = (
            elapsed,
            observed.get("cpus"),
            observed.get("memory_bytes"),
            row.get("timeout_s"),
        )
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in values
        ):
            failures.append(row["case_id"])
            continue
        if (
            not 0 <= elapsed <= row["timeout_s"] <= cap["timeout_s"]
            or not 0 < observed["cpus"] <= cap["cpus"]
            or not 0 < observed["memory_bytes"] <= cap["memory_bytes"]
            or observed.get("network") != "none"
            or observed.get("oom_killed") is not False
            or cap["network"] not in ("none", "restricted")
            or not image_id
            or observed.get("image_id") != image_id
        ):
            failures.append(row["case_id"])
        ratios.append(elapsed / cap["timeout_s"])
    ratio = (
        sorted(ratios)[max(0, math.ceil(len(ratios) * 0.95) - 1)] if ratios else None
    )
    return (
        False
        if failures or (ratio is not None and ratio > p95_fraction)
        else None
        if missing or not rows
        else True,
        {
            "missing_runs": missing,
            "failed_cases": sorted(set(failures)),
            "p95_timeout_fraction": ratio,
            "scope": "Docker-observed caps and process wall time; no organiser hardware-equivalence claim",
        },
    )
