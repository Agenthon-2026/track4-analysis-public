"""Development search consumes budget and exposes data before fresh acceptance."""

import json
import sys
from types import SimpleNamespace

import pytest

from baselines.evaluation import batch, lifecycle, search
from baselines.evaluation.tests import test_batch
from baselines.evaluation.tests.test_runner_integration import build_unit
from scoring.tests.synthetic import outcome_for


def manifest_at(path, *, split="calibration", tag="development", later=False):
    unit = build_unit(path / "inputs")
    task_path = unit / "task.json"
    task = json.loads(task_path.read_text())
    if later:
        task["cutoff_date"] = "2026-08-01"
        task["resolution_date"] = "2026-08-20"
    # Distinct synthetic bytes; not a claim of independent real financial events.
    task["task_id"] = tag
    task_path.write_text(json.dumps(task))
    truth = path / "truth.json"
    batch.write_new(truth, outcome_for())
    manifest = path / "manifest.json"
    batch.write_new(
        manifest,
        {
            "version": 1,
            "cases": [
                {
                    "id": tag,
                    "unit_dir": str(unit),
                    "truth_path": str(truth),
                    "group": tag,
                    "domain": "synthetic-domain",
                    "split": split,
                }
            ],
        },
    )
    return manifest


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "registry"
    manifest = manifest_at(tmp_path / "development")

    def snapshot(spec):
        return {
            "git_commit": spec["repo"],
            "git_dirty": False,
            "source_digest": spec["repo"],
            "toolkit_source_digest": "test-toolkit",
        }

    monkeypatch.setattr(batch, "snapshot", snapshot)
    baseline = {"repo": "baseline", "python": sys.executable, "image": None}
    lifecycle.initialize(root, baseline)
    lifecycle.start_round(
        root,
        "synthetic correction improves development scores",
        {"max_runs": 8, "max_reserved_seconds": 4800},
        development_manifest=manifest,
        seeds=[1],
    )
    return root, manifest, baseline


def register(workspace, name="candidate", *, seeds=None, manifest=None):
    root, original, baseline = workspace
    return batch.register(
        root,
        manifest or original,
        {"before": baseline, "after": {**baseline, "repo": name}},
        [1] if seeds is None else seeds,
        purpose="development",
    )


def complete(directory, *, gain=0.02, failure=False):
    record = batch.read_registration(directory)
    for role in ("before", "after"):
        with batch.locked(directory.parent.parent):
            reservation = search.reserve_run(
                directory.parent.parent, directory, role, record
            )
        report = test_batch.completed_report(directory, role, record)
        report["runs"][0]["assessment"]["development_score"] = 0.1 + (
            gain if role == "after" else 0
        )
        report["runs"][0]["assessment"]["admissible"] = not (
            failure and role == "after"
        )
        (directory / role / "report.json").write_text(json.dumps(report))
        row = report["runs"][0]
        (directory / role / row["case_id"] / "seed-1/result.json").write_text(
            json.dumps(row)
        )
        receipt_path = directory / f"{role}.receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["report_digest"] = batch.digest(report)
        receipt_path.write_text(json.dumps(receipt))
        (directory / f"{role}.started.json").write_text(
            json.dumps(
                {
                    "registration_digest": directory.name,
                    "round_reservation": reservation,
                }
            )
        )


def fresh(workspace, *, name="candidate", later=True):
    root, manifest, baseline = workspace
    path = manifest_at(
        manifest.parent.parent / ("fresh-" + name),
        split="test",
        tag="fresh-" + name,
        later=later,
    )
    return batch.register(
        root, path, {"before": baseline, "after": {**baseline, "repo": name}}, [1]
    )


def test_search_selects_best_measured_candidate_and_attaches_fresh_pair(workspace):
    first = register(workspace, "first")
    complete(first, gain=0.01)
    second = register(workspace, "second")
    complete(second, gain=0.03)
    result = search.select(workspace[0])
    assert result["selected_batch"] == second.name
    assert (
        result["decision"] == "SELECT_FOR_FRESH_ACCEPTANCE" and not result["rankable"]
    )
    directory = fresh(workspace, name="second")
    state = lifecycle.attach(workspace[0], directory)
    assert state["pending"]["development_selection"][
        "selection_digest"
    ] == batch.digest(result)
    assert sum(r["budget"]["runs"] for r in state["round"]["reservations"]) == 4
    with pytest.raises(ValueError, match="already closed"):
        register(workspace, "third")
    with pytest.raises(ValueError, match="already closed"):
        search.select(workspace[0])


def test_candidate_cap_and_duplicate_cannot_be_reset_inside_round(workspace):
    first = register(workspace, "first")
    with pytest.raises(ValueError, match="already registered"):
        register(workspace, "first")
    search.abandon(workspace[0], first, "synthetic interrupted candidate")
    register(workspace, "second")
    register(workspace, "third")
    with pytest.raises(ValueError, match="candidate limit"):
        register(workspace, "fourth")
    assert (
        len(lifecycle.status(workspace[0])["round"]["development"]["candidates"]) == 3
    )


def test_development_data_cannot_be_relabeled_as_fresh_acceptance(workspace):
    directory = register(workspace)
    root, manifest, baseline = workspace
    raw = json.loads(manifest.read_text())
    raw["cases"][0].update(split="test", id="renamed", group="renamed")
    changed = manifest.with_name("relabeled.json")
    batch.write_new(changed, raw)
    with pytest.raises(ValueError, match="reserved/consumed"):
        batch.register(
            root,
            changed,
            {"before": baseline, "after": {**baseline, "repo": "other"}},
            [1],
        )
    with pytest.raises(ValueError, match="cannot be attached"):
        lifecycle.attach(root, directory)
    complete(directory)
    with pytest.raises(ValueError, match="cannot receive an acceptance"):
        batch.decide(directory)


@pytest.mark.parametrize("change", ["seed", "manifest", "baseline"])
def test_search_roster_seed_and_baseline_are_frozen(workspace, change):
    root, manifest, baseline = workspace
    versions = {"before": baseline, "after": {**baseline, "repo": "candidate"}}
    seeds = [1]
    if change == "seed":
        seeds = [2]
    if change == "manifest":
        other = manifest.with_name("other.json")
        other.write_bytes(manifest.read_bytes())
        manifest = other
    if change == "baseline":
        versions["before"] = {**baseline, "repo": "other"}
    with pytest.raises(ValueError, match="roster or seeds|retain the round"):
        batch.register(root, manifest, versions, seeds, purpose="development")
    assert not lifecycle.status(root)["round"]["development"]["candidates"]


@pytest.mark.parametrize("gain,failure", [(0, False), (-0.02, False), (0.05, True)])
def test_no_qualifying_development_candidate_keeps_incumbent(workspace, gain, failure):
    directory = register(workspace)
    complete(directory, gain=gain, failure=failure)
    result = search.select(workspace[0])
    assert result["decision"] == "KEEP_INCUMBENT"
    assert result["selected_batch"] is None
    with pytest.raises(ValueError, match="differs from the selected"):
        lifecycle.attach(workspace[0], fresh(workspace))


def test_unfinished_candidate_cannot_disappear_from_search(workspace):
    first = register(workspace, "first")
    complete(first)
    second = register(workspace, "second")
    with pytest.raises((OSError, ValueError)):
        search.select(workspace[0])
    search.abandon(workspace[0], second, "no complete evaluator output")
    with pytest.raises(ValueError, match="must participate"):
        search.abandon(workspace[0], first, "hide completed score")
    result = search.select(workspace[0])
    assert [m["status"] for m in result["measurements"]] == ["MEASURED", "ABANDONED"]
    assert result["selected_batch"] == first.name


@pytest.mark.parametrize("change", ["candidate", "dates", "report"])
def test_fresh_acceptance_requires_selected_version_and_unchanged_prior_evidence(
    workspace, change
):
    directory = register(workspace)
    complete(directory)
    search.select(workspace[0])
    if change == "report":
        path = directory / "after/report.json"
        data = json.loads(path.read_text())
        data["runs"][0]["assessment"]["development_score"] = 0.99
        path.write_text(json.dumps(data))
    future = fresh(
        workspace,
        name="other" if change == "candidate" else "candidate",
        later=change != "dates",
    )
    with pytest.raises(
        ValueError, match="selected development|precede fresh|persisted result"
    ):
        lifecycle.attach(workspace[0], future)


def test_combined_development_and_acceptance_budget_cannot_overspend(workspace):
    root, manifest, _ = workspace
    lifecycle.close_round(root, "start exact budget control")
    lifecycle.start_round(
        root,
        "budget control",
        {"max_runs": 2, "max_reserved_seconds": 1200},
        development_manifest=manifest,
        seeds=[1],
    )
    directory = register(workspace)
    complete(directory)
    search.select(root)
    with pytest.raises(ValueError, match="budget exhausted"):
        lifecycle.attach(root, fresh(workspace))
    assert (
        sum(
            r["budget"]["runs"] for r in lifecycle.status(root)["round"]["reservations"]
        )
        == 2
    )


def test_crash_burns_candidate_role_and_abandon_does_not_refund(workspace, monkeypatch):
    root, _, _ = workspace
    directory = register(workspace)
    real = batch.subprocess.run

    def crash(argv, **kwargs):
        if argv[0] == "git":
            return real(argv, **kwargs)
        raise batch.subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(batch.subprocess, "run", crash)
    with pytest.raises(batch.subprocess.TimeoutExpired):
        batch.run(directory, "before")
    search.abandon(root, directory, "synthetic process timed out")
    with pytest.raises(FileExistsError, match="do not retry"):
        batch.run(directory, "before")
    with pytest.raises(ValueError, match="abandoned"):
        batch.run(directory, "after")
    state = lifecycle.status(root)
    assert sum(r["budget"]["runs"] for r in state["round"]["reservations"]) == 1
    result = search.select(root)
    assert result["decision"] == "KEEP_INCUMBENT"


def test_started_reservation_is_verified_after_round_is_closed(workspace):
    directory = register(workspace)
    complete(directory)
    root = workspace[0]
    search.select(root)
    lifecycle.close_round(root, "retain measured development evidence")
    assert batch.verified_reports(directory)[2]["purpose"] == "development"
    started = directory / "after.started.json"
    raw = json.loads(started.read_text())
    raw["round_reservation"]["budget"]["runs"] = 0
    started.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="round journal"):
        batch.verified_reports(directory)


def test_batch_runner_executes_both_development_roles_with_receipts(
    workspace, monkeypatch
):
    directory = register(workspace)
    record = batch.read_registration(directory)
    real = batch.subprocess.run
    calls = []

    def evaluator(argv, **kwargs):
        if argv[0] == "git":
            return real(argv, **kwargs)
        role = "before" if not calls else "after"
        calls.append(kwargs)
        report = {
            "runs": [
                {
                    **r,
                    "status": "completed",
                    "profile": "smoke",
                    "execution": {"returncode": 0, "timed_out": False},
                    "assessment": {
                        "development_score": 0.1 if role == "before" else 0.12,
                        "admissible": True,
                    },
                    "answer_sha256": None,
                }
                for r in record["expected_runs"]
            ],
            "mode": "grounded",
            "profile": "smoke",
            "provenance": record["versions"][role]["identity"],
        }
        out = directory / role
        for row in report["runs"]:
            path = out / row["case_id"] / f"seed-{row['seed']}"
            path.mkdir(parents=True)
            batch.write_new(path / "result.json", row)
        batch.write_new(out / "report.json", report)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(batch.subprocess, "run", evaluator)
    batch.run(directory, "before")
    batch.run(directory, "after")
    assert all(c["timeout"] == 600 for c in calls)
    assert search.select(workspace[0])["selected_batch"] == directory.name


@pytest.mark.parametrize("phase", ["before-run", "after-close"])
def test_acceptance_rechecks_development_evidence_after_attach(workspace, phase):
    development = register(workspace)
    complete(development)
    root = workspace[0]
    search.select(root)
    directory = fresh(workspace)
    lifecycle.attach(root, directory)
    if phase == "after-close":
        from baselines.evaluation.tests.test_lifecycle import (
            complete as complete_acceptance,
        )

        complete_acceptance(directory)
        lifecycle.resolve(root)
        lifecycle.close_round(root, "synthetic acceptance finished")
        assert (
            batch.verified_decision(directory)["registration_digest"] == directory.name
        )
    path = development / "after/report.json"
    data = json.loads(path.read_text())
    data["runs"][0]["assessment"]["development_score"] = 0.5
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="persisted result"):
        if phase == "before-run":
            batch.run(directory, "before")
        else:
            batch.verified_decision(directory)


def test_development_reservation_survives_crash_before_started_marker(workspace):
    directory = register(workspace)
    root = workspace[0]
    with batch.locked(root):
        search.reserve_run(
            root, directory, "before", batch.read_registration(directory)
        )
    with pytest.raises(ValueError, match="already consumed"):
        batch.run(directory, "before")
    assert len(lifecycle.status(root)["round"]["reservations"]) == 1


def test_search_requires_both_manifest_and_seeds(workspace):
    root, manifest, _ = workspace
    lifecycle.close_round(root, "configuration control")
    with pytest.raises(ValueError, match="frozen together"):
        lifecycle.start_round(
            root,
            "bad config",
            {"max_runs": 2, "max_reserved_seconds": 1200},
            development_manifest=manifest,
        )
    with pytest.raises(ValueError, match="frozen together"):
        lifecycle.start_round(
            root, "bad config", {"max_runs": 2, "max_reserved_seconds": 1200}, seeds=[1]
        )


def test_search_cannot_register_against_acceptance_only_round(workspace):
    root = workspace[0]
    lifecycle.close_round(root, "preselected candidate round")
    lifecycle.start_round(
        root, "external selection", {"max_runs": 2, "max_reserved_seconds": 1200}
    )
    with pytest.raises(ValueError, match="frozen development"):
        register(workspace)
