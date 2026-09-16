"""Recovery evidence is checked against the protocol, not report pass flags."""

import hashlib
import json

import pytest

pytest.importorskip("qfbench2_common")

from baselines.evaluation.__main__ import write_json  # noqa: E402
from baselines.evaluation import faults  # noqa: E402
from baselines.strong_rag_baseline.cli import run  # noqa: E402


def traces(root, scenario="malformed_reply", kind="regression"):
    identity = {
        "image_id": "sha256:" + "1" * 64,
        "runtime_digest": "synthetic",
        "runtime_manifest": {"analyze.py": "synthetic"},
    }
    plan = faults.protocol_plan(identity)
    case = next(
        c for c in plan["cases"] if c["kind"] == kind and c["scenario"] == scenario
    )
    for name, content in faults.fixture(case["kind"], case["entities"]).items():
        path = root / "input" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, content)
    run(
        root / "input/task.json",
        root / "input/corpus",
        root / "answer.json",
        None,
        3,
        grounded=True,
        diagnostics_path=root / "diagnostics.json",
    )
    diagnostic = json.loads((root / "diagnostics.json").read_text())
    count = (
        case["entities"]
        if scenario == "cold_start"
        else 25
        if scenario == "budget_exhaustion"
        else 2 * case["entities"]
    )
    attempts = [
        {
            "sequence": i + 1,
            "request_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
            "status": "success" if scenario == "cold_start" else "error",
            "response_bytes": 0,
        }
        for i in range(count)
    ]
    diagnostic["mode"] = "model"
    diagnostic["request_ledger"] = {
        "version": 1,
        "source": "http-client",
        "attempts_used": count,
        "attempt_limit": 25,
        "attempts": attempts,
    }
    for i, entity in enumerate(diagnostic["entities"]):
        entity["source"] = "model" if scenario == "cold_start" else "grounded"
        entity["fallback_reason"] = (
            None
            if scenario == "cold_start"
            else "model_budget"
            if scenario == "budget_exhaustion" and i == 25
            else "model_request"
        )
    write_json(root / "diagnostics.json", diagnostic)
    write_json(
        root / "exercise.json",
        {
            "scenario": scenario,
            "returncode": 0,
            "received_requests": []
            if scenario == "connection_refused"
            else [
                {
                    "sequence": r["sequence"],
                    "request_sha256": r["request_sha256"],
                    "action": scenario,
                    "path": "/v1/chat/completions",
                }
                for r in attempts
            ],
        },
    )
    execution = {
        "returncode": 0,
        "timed_out": False,
        "process_elapsed_s": 2,
        "resource_observation": {
            "image_id": identity["image_id"],
            "cpus": 1,
            "memory_bytes": 1024**3,
            "network": "none",
            "oom_killed": False,
        },
    }
    return case, plan, execution


@pytest.mark.parametrize("scenario", faults.SCENARIOS)
def test_protocol_verifier_accepts_complete_invented_traces(tmp_path, scenario):
    case, plan, execution = traces(tmp_path, scenario)
    result = faults.verify_case(case, plan, tmp_path, execution)
    assert result["attempts"] > 0
    assert result["status"] == "PASS"
    assert result["recovered_entities"] == (
        0 if scenario == "cold_start" else case["entities"]
    )


@pytest.mark.parametrize(
    "change",
    ["lost_request", "mixed_fallback", "missing_entity", "changed_input", "oom"],
)
def test_broken_recovery_cannot_be_certified(tmp_path, change):
    case, plan, execution = traces(tmp_path)
    if change == "lost_request":
        path = tmp_path / "exercise.json"
        data = json.loads(path.read_text())
        data["received_requests"].pop()
    elif change == "mixed_fallback":
        path = tmp_path / "diagnostics.json"
        data = json.loads(path.read_text())
        data["entities"][0]["source"] = "model"
    elif change == "missing_entity":
        path = tmp_path / "answer.json"
        data = json.loads(path.read_text())
        data["entity_predictions"].pop()
    elif change == "changed_input":
        path = tmp_path / "input/task.json"
        data = json.loads(path.read_text())
        data["cutoff_date"] = "2025-01-31"
    else:
        execution["resource_observation"]["oom_killed"] = True
        path = None
    if path:
        write_json(path, data)
    with pytest.raises(ValueError):
        faults.verify_case(case, plan, tmp_path, execution)


def test_suite_rejects_different_image_and_incomplete_roster(tmp_path):
    _, plan, _ = traces(tmp_path / "trace")
    report = {"plan": plan, "plan_digest": faults.digest(plan), "runs": []}
    write_json(tmp_path / "fault-plan.json", plan)
    path = tmp_path / "report.json"
    write_json(path, report)
    with pytest.raises(ValueError, match="different candidate runtime"):
        faults.load_suite(path, {**plan["identity"], "image_id": "other"})
    with pytest.raises(ValueError, match="full scenario roster"):
        faults.load_suite(path, plan["identity"])


def synthetic_suite(tmp_path):
    identity = {
        "image_id": "sha256:" + "1" * 64,
        "runtime_digest": "synthetic",
        "runtime_manifest": {"analyze.py": "synthetic"},
    }
    plan = faults.protocol_plan(identity)
    rows = []
    for case in plan["cases"]:
        root = tmp_path / case["case_id"]
        _, _, execution = traces(root, case["scenario"], case["kind"])
        row = {
            "case_id": case["case_id"],
            "execution": execution,
            "artifact_hashes": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ("answer.json", "diagnostics.json", "exercise.json")
            },
        }
        write_json(root / "result.json", row)
        rows.append(row)
    write_json(tmp_path / "fault-plan.json", plan)
    path = tmp_path / "report.json"
    write_json(path, {"plan": plan, "plan_digest": faults.digest(plan), "runs": rows})
    return path, identity, plan


def test_complete_suite_is_recomputed_and_artifact_edits_are_rejected(tmp_path):
    from baselines.evaluation.acceptance import engineering_checks
    from baselines.evaluation.tests.test_acceptance import reports, checks_by_name

    path, identity, plan = synthetic_suite(tmp_path)
    loaded = faults.load_suite(path, identity)
    assert len(loaded["cases"]) == len(faults.SCENARIOS) * len(faults.KINDS)
    assert sum(row["recovered_entities"] for row in loaded["cases"]) > 0
    before, _ = reports(groups=1, repeats=1)
    before["provenance"].update(identity)
    assert (
        checks_by_name(engineering_checks(before, faults=path))["fault_recovery"][
            "status"
        ]
        == "PASS"
    )
    write_json(tmp_path / plan["cases"][0]["case_id"] / "diagnostics.json", {})
    with pytest.raises(ValueError, match="artifact hash differs"):
        faults.load_suite(path, identity)
