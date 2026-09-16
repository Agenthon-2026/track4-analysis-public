"""Sealed batches bind the full pair and cannot reuse observed events."""

import copy
import hashlib
import json

import pytest

from baselines.evaluation import batch
from baselines.evaluation.acceptance import digest
from baselines.evaluation.tests.test_runner_integration import build_unit
from scoring.tests.synthetic import outcome_for


@pytest.fixture
def setup(tmp_path, monkeypatch):
    unit = build_unit(tmp_path / "inputs")
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps(outcome_for()))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "synthetic-case",
                        "unit_dir": str(unit),
                        "truth_path": str(truth),
                        "group": "synthetic-event",
                        "domain": "synthetic-domain",
                        "split": "test",
                    }
                ],
            }
        )
    )
    identity = {
        "source_digest": "source",
        "git_commit": "commit",
        "toolkit_source_digest": "toolkit",
    }
    monkeypatch.setattr(batch, "snapshot", lambda spec: copy.deepcopy(identity))
    versions = {
        role: {"repo": "synthetic", "python": "synthetic", "image": None}
        for role in ("before", "after")
    }
    return tmp_path / "registry", manifest, versions, identity


def completed_report(directory, role, record):
    rows = [
        {
            **r,
            "status": "completed",
            "profile": "smoke",
            "execution": {"returncode": 0, "timed_out": False},
            "assessment": {"development_score": 0.1, "admissible": True},
            "answer_sha256": None,
        }
        for r in record["expected_runs"]
    ]
    report = {
        "runs": rows,
        "provenance": record["versions"][role]["identity"],
        "mode": "grounded",
        "profile": "smoke",
    }
    for row in rows:
        folder = directory / role / row["case_id"] / f"seed-{row['seed']}"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps(row))
    path = directory / role / "report.json"
    path.write_text(json.dumps(report))
    batch.write_new(
        directory / f"{role}.started.json", {"registration_digest": directory.name}
    )
    batch.write_new(
        directory / f"{role}.receipt.json",
        {
            "registration_digest": directory.name,
            "report_digest": digest(report),
            "report_path": str(path.resolve()),
            "returncode": 0,
        },
    )
    return report


def test_register_freezes_policy_inputs_truth_and_pair(setup):
    root, manifest, versions, _ = setup
    directory = batch.register(root, manifest, versions, [1, 2, 3])
    record = batch.read_registration(directory)
    assert record["budget"]["model_request_attempts"] == 0
    assert len(record["expected_runs"]) == 3
    assert (
        record["expected_runs"][0]["truth_digest"]
        == hashlib.sha256((manifest.parent / "truth.json").read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="reserved/consumed"):
        batch.register(root, manifest, versions, [4, 5, 6])


def test_rename_case_and_group_does_not_reuse_same_input(setup):
    root, manifest, versions, _ = setup
    batch.register(root, manifest, versions, [1])
    data = json.loads(manifest.read_text())
    data["cases"][0].update(id="renamed", group="renamed-event")
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="reserved/consumed"):
        batch.register(root, manifest, versions, [2])


def test_registration_edit_is_detected(setup):
    root, manifest, versions, _ = setup
    directory = batch.register(root, manifest, versions, [1])
    path = directory / "registration.json"
    data = json.loads(path.read_text())
    data["seeds"] = [2]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="modified"):
        batch.read_registration(directory)


def test_one_time_decision_never_promotes_missing_production_evidence(setup):
    root, manifest, versions, _ = setup
    directory = batch.register(root, manifest, versions, [1, 2, 3])
    record = batch.read_registration(directory)
    for role in ("before", "after"):
        completed_report(directory, role, record)
    result = batch.decide(directory)
    assert result["decision"] == "KEEP_INCUMBENT"
    assert result["goals"]["G3"]["status"] == "UNMEASURED"
    g1 = {c["name"]: c["status"] for c in result["goals"]["G1"]["checks"]}
    assert g1["planned_roster"] == "PASS"
    with pytest.raises(ValueError, match="already consumed"):
        batch.decide(directory)


def test_receipt_rejects_report_replacement_even_with_matching_result_files(setup):
    root, manifest, versions, _ = setup
    directory = batch.register(root, manifest, versions, [1])
    record = batch.read_registration(directory)
    for role in ("before", "after"):
        completed_report(directory, role, record)
    path = directory / "after/report.json"
    data = json.loads(path.read_text())
    data["runs"][0]["assessment"]["development_score"] = 0.9
    path.write_text(json.dumps(data))
    row = data["runs"][0]
    (directory / "after" / row["case_id"] / "seed-1/result.json").write_text(
        json.dumps(row)
    )
    with pytest.raises(ValueError, match="changed after"):
        batch.verified_reports(directory)


def test_complete_roster_binding_detects_omitted_run(setup):
    root, manifest, versions, _ = setup
    directory = batch.register(root, manifest, versions, [1, 2])
    record = batch.read_registration(directory)
    report = completed_report(directory, "before", record)
    report["runs"].pop()
    with pytest.raises(ValueError, match="complete registered roster"):
        batch.validate_report(report, record, "before")


def test_started_run_cannot_be_retried(setup):
    root, manifest, versions, _ = setup
    directory = batch.register(root, manifest, versions, [1])
    batch.write_new(
        directory / "before.started.json", {"registration_digest": directory.name}
    )
    with pytest.raises(FileExistsError):
        batch.run(directory, "before")


def test_source_mutation_before_run_does_not_start(setup):
    root, manifest, versions, identity = setup
    directory = batch.register(root, manifest, versions, [1])
    identity["source_digest"] = "changed"
    with pytest.raises(ValueError, match="version changed"):
        batch.run(directory, "before")
    assert not (directory / "before.started.json").exists()


def test_calibration_data_cannot_be_registered_as_blind_acceptance(setup):
    root, manifest, versions, _ = setup
    data = json.loads(manifest.read_text())
    data["cases"][0]["split"] = "calibration"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="test-only"):
        batch.register(root, manifest, versions, [1])


def test_registered_pair_runs_real_predictor_and_produces_bound_decision(tmp_path):
    import sys
    from baselines.evaluation.dataset import REPO

    unit = build_unit(tmp_path / "inputs")
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps(outcome_for()))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "synthetic-case",
                        "unit_dir": str(unit),
                        "truth_path": str(truth),
                        "group": "synthetic-event",
                        "domain": "synthetic-domain",
                        "split": "test",
                    }
                ],
            }
        )
    )
    versions = {
        role: {"repo": str(REPO), "python": sys.executable, "image": None}
        for role in ("before", "after")
    }
    directory = batch.register(tmp_path / "registry", manifest, versions, [1])
    for role in ("before", "after"):
        path = batch.run(directory, role)
        assert json.loads(path.read_text())["runs"][0]["execution"]["returncode"] == 0
    result = batch.decide(directory)
    assert result["comparison"]["overall"]["event_mean_delta"] == 0
    assert result["goals"]["G2"]["status"] == "UNMEASURED"


def test_missing_reservation_prevents_run_and_decision(setup):
    root, manifest, versions, _ = setup
    directory = batch.register(root, manifest, versions, [1])
    marker = root / "events" / digest({"group": "synthetic-event"})
    marker.unlink()
    with pytest.raises(FileNotFoundError):
        batch.read_registration(directory)


def test_registered_faults_are_reverified_in_both_engineering_audits(setup):
    from baselines.evaluation.tests.test_faults import synthetic_suite
    from baselines.evaluation.tests.test_acceptance import checks_by_name

    root, manifest, versions, identity = setup
    path, runtime, _ = synthetic_suite(root.parent / "recovery")
    identity.update(runtime)
    directory = batch.register(
        root, manifest, versions, [1], faults={"before": path, "after": path}
    )
    record = batch.read_registration(directory)
    for role in ("before", "after"):
        completed_report(directory, role, record)
    result = batch.decide(directory)
    assert (
        checks_by_name(result["goals"]["G1"]["checks"])["fault_recovery"]["status"]
        == "PASS"
    )
    baseline = checks_by_name(result["goals"]["G2"]["checks"])["baseline_engineering"][
        "detail"
    ]
    assert checks_by_name(baseline["checks"])["fault_recovery"]["status"] == "PASS"
    assert result["decision"] == "KEEP_INCUMBENT"


@pytest.mark.parametrize("change", ["report", "artifact", "runtime"])
def test_registered_fault_changes_are_rejected_before_execution(setup, change):
    from baselines.evaluation.tests.test_faults import synthetic_suite

    root, manifest, versions, identity = setup
    path, runtime, plan = synthetic_suite(root.parent / "recovery")
    identity.update(runtime)
    directory = batch.register(
        root, manifest, versions, [1], faults={"before": path, "after": path}
    )
    if change == "report":
        path.write_text(path.read_text() + "\n")
    elif change == "artifact":
        (path.parent / plan["cases"][0]["case_id"] / "exercise.json").write_text("{}")
    else:
        identity["runtime_digest"] = "different"
    with pytest.raises(ValueError, match="changed since|artifact hash|version changed"):
        batch.run(directory, "before")
    assert not (directory / "before.started.json").exists()


def test_fault_registration_requires_both_roles_and_matching_runtime(setup):
    from baselines.evaluation.tests.test_faults import synthetic_suite

    root, manifest, versions, identity = setup
    path, runtime, _ = synthetic_suite(root.parent / "recovery")
    identity.update(runtime)
    with pytest.raises(ValueError, match="both versions"):
        batch.register(root, manifest, versions, [1], faults={"before": path})
    identity["image_id"] = "different"
    with pytest.raises(ValueError, match="different candidate runtime"):
        batch.register(
            root, manifest, versions, [1], faults={"before": path, "after": path}
        )
    assert not root.exists()
