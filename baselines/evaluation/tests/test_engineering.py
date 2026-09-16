"""Engineering evidence must bind the frozen roster, actual caps and runtime."""

import copy
import json
import subprocess
from pathlib import Path

import pytest

from baselines.evaluation import engineering
from baselines.evaluation.artifacts import digest
from baselines.evaluation.acceptance import load_report
from baselines.evaluation.dataset import Case
from baselines.evaluation.tests.test_runner_integration import build_unit


def planned_report(tmp_path):
    unit = build_unit(tmp_path / "inputs")
    with (unit / "card.toml").open("a") as card:
        card.write("\n[agent]\ntimeout_sec = 60\n")
    plan = engineering.make_run_plan(
        [Case("synthetic", unit, "public-dev", "event")],
        [1, 2],
        identity={"image_id": "sha256:" + "1" * 64},
        timeout=50,
        mode="grounded",
        profile="smoke",
    )
    rows = copy.deepcopy(plan["runs"])
    for row in rows:
        row.update(
            assessment={},
            execution={
                "process_elapsed_s": 10,
                "resource_observation": {
                    "image_id": plan["provenance"]["image_id"],
                    "cpus": 1,
                    "memory_bytes": 1024**3,
                    "network": "none",
                    "oom_killed": False,
                },
            },
        )
    return {
        **{key: plan[key] for key in ("provenance", "mode", "profile")},
        "run_plan": plan,
        "run_plan_digest": digest(plan),
        "runs": rows,
    }


def test_run_plan_freezes_caps_and_detects_dropped_run(tmp_path):
    report = planned_report(tmp_path)
    engineering.verify_run_plan(report)
    assert {row["timeout_s"] for row in report["runs"]} == {50}
    report["runs"].pop()
    with pytest.raises(ValueError, match="entire frozen run plan"):
        engineering.verify_run_plan(report)


@pytest.mark.parametrize(
    "field,value", [("input_digest", "other"), ("seed", 3), ("timeout_s", 500)]
)
def test_run_plan_rejects_changed_execution_inputs(tmp_path, field, value):
    report = planned_report(tmp_path)
    report["runs"][0][field] = value
    with pytest.raises(ValueError):
        engineering.verify_run_plan(report)


def test_resources_use_actual_limits_and_time_not_requested_flags(tmp_path):
    report = planned_report(tmp_path)
    image = report["provenance"]["image_id"]
    rows = report["runs"]
    assert engineering.resource_observations(rows, 0.8, image)[0] is True
    rows[0]["execution"]["resource_observation"]["memory_bytes"] = 0
    assert engineering.resource_observations(rows, 0.8, image)[0] is False
    rows[0]["execution"].pop("resource_observation")
    assert engineering.resource_observations(rows, 0.8, image)[0] is None


@pytest.mark.parametrize("change", ["oom", "image", "network", "slow", "nan"])
def test_bad_observations_never_pass(tmp_path, change):
    report = planned_report(tmp_path)
    execution = report["runs"][0]["execution"]
    observation = execution["resource_observation"]
    if change == "oom":
        observation["oom_killed"] = True
    elif change == "image":
        observation["image_id"] = "other"
    elif change == "network":
        observation["network"] = "bridge"
    else:
        execution["process_elapsed_s"] = float("nan") if change == "nan" else 49
    assert (
        engineering.resource_observations(
            report["runs"], 0.8, report["provenance"]["image_id"]
        )[0]
        is False
    )


def test_artifact_reader_detects_changed_plan_and_diagnostics(tmp_path):
    report = planned_report(tmp_path)
    path = tmp_path / "report.json"
    plan_path = tmp_path / "run-plan.json"
    plan_path.write_text(json.dumps(report["run_plan"]))
    for row in report["runs"]:
        folder = tmp_path / row["case_id"] / f"seed-{row['seed']}"
        folder.mkdir(parents=True)
        row["diagnostics"] = {"request_ledger": {"attempts": []}}
        payload = json.dumps(row["diagnostics"]).encode()
        import hashlib

        row["diagnostics_sha256"] = hashlib.sha256(payload).hexdigest()
        (folder / "diagnostics.json").write_bytes(payload)
        (folder / "result.json").write_text(json.dumps(row))
    path.write_text(json.dumps(report))
    assert load_report(path) == report
    plan_path.write_text("{}")
    with pytest.raises(ValueError, match="persisted run plan"):
        load_report(path)
    plan_path.write_text(json.dumps(report["run_plan"]))
    (folder / "diagnostics.json").write_text("{}")
    with pytest.raises(ValueError, match="diagnostics differ"):
        load_report(path)


@pytest.mark.parametrize("changed", [False, True])
def test_image_identity_binds_actual_runtime_and_always_cleans_up(
    tmp_path, monkeypatch, changed
):
    baseline = tmp_path / "baselines"
    package = baseline / "strong_rag_baseline"
    package.mkdir(parents=True)
    for name in ("analyze.py", "requirements.txt", "strong_rag_baseline/agent.py"):
        (baseline / name).write_text("synthetic runtime")
    calls = []

    def docker(argv, **kwargs):
        calls.append(argv)
        command = argv[1]
        output = ""
        if command == "image":
            output = json.dumps(
                [
                    {
                        "Id": "sha256:" + "1" * 64,
                        "Architecture": "amd64",
                        "Config": {
                            "Entrypoint": ["python", "analyze.py"],
                            "WorkingDir": "/app",
                            "Labels": {"qfbench2.interface_version": "2.0"},
                        },
                    }
                ]
            )
        elif command == "create":
            output = "synthetic-container"
        elif command == "cp":
            import shutil

            shutil.copytree(baseline, Path(argv[3]), dirs_exist_ok=True)
            if changed:
                (Path(argv[3]) / "analyze.py").write_text("different runtime")
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(engineering, "REPO", tmp_path)
    monkeypatch.setattr(engineering.subprocess, "run", docker)
    if changed:
        with pytest.raises(ValueError, match="runtime source differs"):
            engineering.image_identity("tag")
    else:
        identity = engineering.image_identity("tag")
        assert identity["runtime_digest"] == digest(
            engineering.runtime_manifest(baseline)
        )
    assert calls[-1][1:] == ["rm", "--force", "synthetic-container"]
