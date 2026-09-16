"""Run and verify frozen-image recovery exercises using invented protocol fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .artifacts import digest, read_json_inside
from .dataset import public_roots, require_external
from .engineering import image_identity, resource_observations
from .fault_driver import SCENARIOS

DRIVER = Path(__file__).with_name("fault_driver.py")
KINDS = ("classification", "regression", "ranking")


def fixture(kind: str, count: int) -> dict[str, dict]:
    """Invented forecasts, not resolved outcomes or variants of competition units."""
    entities = [
        {"entity_id": f"SYN{i:02}", "name": f"Synthetic Issuer SYN{i:02}"}
        for i in range(count)
    ]
    task = {
        "task_id": f"synthetic-recovery-{kind}",
        "schema_version": "3",
        "cutoff_date": "2024-01-31",
        "resolution_date": "2024-04-30",
        "target": {"type": kind, "name": "revenue_growth_pct", "unit": "percent"},
        "interval_level": 0.9,
        "entities": entities,
    }
    if kind == "classification":
        task["target"]["labels"] = ["up", "down"]
    result = {"task.json": task}
    for entity in entities:
        result[f"corpus/{entity['entity_id']}.json"] = {
            "doc_date": "2024-01-10",
            "text": f"{entity['name']} expects revenue growth of 5 percent next quarter, in a range of 4 to 6 percent.",
        }
    return result


def protocol_plan(identity: dict) -> dict:
    from baselines.strong_rag_baseline.config import HOUSE_MAX_REQUESTS

    cases = []
    for kind in KINDS:
        for scenario in SCENARIOS:
            count = HOUSE_MAX_REQUESTS + 1 if scenario == "budget_exhaustion" else 3
            cases.append(
                {
                    "case_id": f"{kind}-{scenario}",
                    "kind": kind,
                    "scenario": scenario,
                    "entities": count,
                    "fixture_digest": digest(fixture(kind, count)),
                }
            )
    return {
        "version": 1,
        "identity": identity,
        "seed": 20260915,
        "timeout_s": 120,
        "driver_sha256": hashlib.sha256(DRIVER.read_bytes()).hexdigest(),
        "cases": cases,
    }


def verify_case(case: dict, plan: dict, root: Path, execution: dict) -> dict:
    from jsonschema import Draft202012Validator, ValidationError
    from qfbench2_common.taskcard import schema_path
    from baselines.strong_rag_baseline.config import HOUSE_MAX_REQUESTS
    from baselines.strong_rag_baseline.indexer import build_index
    from baselines.strong_rag_baseline.validation import validate_answer

    inputs = fixture(case["kind"], case["entities"])
    if digest(inputs) != case["fixture_digest"]:
        raise ValueError("fault fixture differs from frozen protocol")
    for relative, payload in inputs.items():
        if read_json_inside(root / "input", relative) != payload:
            raise ValueError("persisted fault input differs from frozen fixture")
    answer = read_json_inside(root, "answer.json")
    diagnostics = read_json_inside(root, "diagnostics.json")
    exercise = read_json_inside(root, "exercise.json")
    try:
        Draft202012Validator(
            json.loads(schema_path("analysis.schema.json").read_text())
        ).validate(answer)
    except ValidationError as exc:
        raise ValueError("fault output violates the submission schema") from exc
    validate_answer(inputs["task.json"], answer, build_index(root / "input/corpus"))
    if exercise.get("scenario") != case["scenario"] or exercise.get("returncode") != 0:
        raise ValueError("fault driver did not complete its declared scenario")
    if (
        execution.get("returncode") != 0
        or execution.get("timed_out") is not False
        or execution.get("artifact_errors")
    ):
        raise ValueError("fault container did not complete cleanly")
    resource_row = {
        "case_id": case["case_id"],
        "timeout_s": plan["timeout_s"],
        "execution": execution,
        "resource_contract": {
            "timeout_s": plan["timeout_s"],
            "cpus": 1,
            "memory_bytes": 1024**3,
            "network": "none",
        },
    }
    resources, _ = resource_observations(
        [resource_row], 1.0, plan["identity"]["image_id"]
    )
    if resources is not True:
        raise ValueError("fault container resource envelope failed")
    ledger = diagnostics.get("request_ledger", {})
    attempts = ledger.get("attempts", [])
    received = exercise.get("received_requests", [])
    scenario = case["scenario"]
    count = (
        case["entities"]
        if scenario == "cold_start"
        else HOUSE_MAX_REQUESTS
        if scenario == "budget_exhaustion"
        else 2 * case["entities"]
    )
    if (
        ledger.get("source") != "http-client"
        or ledger.get("attempts_used") != count
        or ledger.get("attempt_limit") != HOUSE_MAX_REQUESTS
        or len(attempts) != count
        or [r.get("sequence") for r in attempts] != list(range(1, count + 1))
    ):
        raise ValueError("fault request attempts differ from the protocol budget")
    if any(
        row.get("status") != ("success" if scenario == "cold_start" else "error")
        for row in attempts
    ):
        raise ValueError("fault scenario did not exercise the intended HTTP result")
    if scenario == "connection_refused":
        if received or any(row.get("response_bytes") != 0 for row in attempts):
            raise ValueError("refused connection unexpectedly received a reply")
    elif (
        [r.get("request_sha256") for r in attempts]
        != [r.get("request_sha256") for r in received]
        or [r.get("sequence") for r in received] != list(range(1, count + 1))
        or any(
            r.get("action") != scenario or r.get("path") != "/v1/chat/completions"
            for r in received
        )
    ):
        raise ValueError("server observations disagree with client request ledger")
    entities = diagnostics.get("entities", [])
    if [r.get("entity_id") for r in entities] != [
        e["entity_id"] for e in inputs["task.json"]["entities"]
    ]:
        raise ValueError("fault diagnostics do not cover the entire entity roster")
    for index, row in enumerate(entities):
        source = "model" if scenario == "cold_start" else "grounded"
        reason = (
            None
            if scenario == "cold_start"
            else "model_budget"
            if scenario == "budget_exhaustion" and index >= HOUSE_MAX_REQUESTS
            else "model_request"
        )
        if row.get("source") != source or row.get("fallback_reason") != reason:
            raise ValueError("fault did not produce the required whole-entity fallback")
    return {
        "case_id": case["case_id"],
        "attempts": count,
        "received_requests": len(received),
        "recovered_entities": sum(row["source"] == "grounded" for row in entities),
        "status": "PASS",
    }


def load_suite(path: Path, identity: dict) -> dict:
    """Recompute recovery from persisted artifacts; no caller-supplied pass flags."""
    require_external(path, public_roots())
    report = read_json_inside(path.parent, path.name)
    plan = read_json_inside(path.parent, "fault-plan.json")
    if plan != report.get("plan") or digest(plan) != report.get("plan_digest"):
        raise ValueError("fault plan differs from report")
    for key in ("image_id", "runtime_digest", "runtime_manifest"):
        if not identity.get(key) or plan["identity"].get(key) != identity[key]:
            raise ValueError("fault suite belongs to a different candidate runtime")
    if plan != protocol_plan(plan["identity"]):
        raise ValueError("fault suite uses a different driver or protocol")
    rows = report.get("runs", [])
    if [row.get("case_id") for row in rows] != [
        case["case_id"] for case in plan["cases"]
    ]:
        raise ValueError("fault suite does not cover its full scenario roster")
    checks = []
    for case, row in zip(plan["cases"], rows):
        root = path.parent / case["case_id"]
        if read_json_inside(root, "result.json") != row:
            raise ValueError("fault result differs from persisted result")
        for name, expected_hash in row["artifact_hashes"].items():
            read_json_inside(root, name)
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected_hash:
                raise ValueError("fault artifact hash differs")
        if set(row["artifact_hashes"]) != {
            "answer.json",
            "diagnostics.json",
            "exercise.json",
        }:
            raise ValueError("fault evidence bindings are incomplete")
        checks.append(verify_case(case, plan, root, row["execution"]))
    return {
        "cases": checks,
        "image_id": identity["image_id"],
        "plan_digest": digest(plan),
        "scope": "synthetic HTTP recovery on frozen runtime; not production model quality",
    }


def main(argv: list[str] | None = None) -> int:
    from .__main__ import run_container, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require_external(args.out, public_roots())
        identity = image_identity(args.image)
        plan = protocol_plan(identity)
        args.out.mkdir(parents=True, exist_ok=False)
        write_json(args.out / "fault-plan.json", plan)
        rows = []
        for case in plan["cases"]:
            root = args.out / case["case_id"]
            for relative, payload in fixture(case["kind"], case["entities"]).items():
                path = root / "input" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, payload)
            execution = run_container(
                root / "input",
                root,
                image=identity["image_id"],
                seed=plan["seed"],
                timeout=plan["timeout_s"],
                exercise=(DRIVER, case["scenario"]),
            )
            row = {
                "case_id": case["case_id"],
                "execution": execution,
                "artifact_hashes": {
                    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in ("answer.json", "diagnostics.json", "exercise.json")
                },
            }
            write_json(root / "result.json", row)
            verify_case(case, plan, root, execution)
            rows.append(row)
            print(f"{case['case_id']}: recovery PASS", flush=True)
        report = {"version": 1, "plan": plan, "plan_digest": digest(plan), "runs": rows}
        write_json(args.out / "report.json", report)
        load_suite(args.out / "report.json", identity)
        return 0
    except Exception as exc:
        print(f"Fault suite aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
