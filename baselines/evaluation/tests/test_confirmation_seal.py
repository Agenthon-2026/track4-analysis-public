"""A production confirmation keeps its original comparison and one-use budget."""

import copy
import json
from types import SimpleNamespace

import pytest

from baselines.evaluation import batch, confirmations, lifecycle
from baselines.evaluation.tests import test_production
from baselines.evaluation.tests.test_runner_integration import build_unit
from scoring.tests.synthetic import outcome_for

judge_spec = test_production.judge_spec


@pytest.fixture
def registered(tmp_path, monkeypatch, judge_spec):
    root = tmp_path / "registry"
    versions = {
        role: {"repo": role, "python": "synthetic", "image": None}
        for role in ("before", "after")
    }
    monkeypatch.setattr(
        batch,
        "snapshot",
        lambda spec: {
            "git_dirty": False,
            "git_commit": spec["repo"],
            "source_digest": spec["repo"],
            "toolkit_source_digest": "synthetic-toolkit",
        },
    )
    paths = []
    for index in range(4):
        base = tmp_path / str(index)
        unit = build_unit(base / "inputs")
        # Distinct synthetic task bytes, never a claim of independent real events.
        task = unit / "task.json"
        task.write_text(task.read_text() + "\n" * index)
        truth = base / "truth.json"
        batch.write_new(truth, outcome_for())
        manifest = base / "manifest.json"
        batch.write_new(
            manifest,
            {
                "version": 1,
                "cases": [
                    {
                        "id": f"synthetic-{index}",
                        "unit_dir": str(unit),
                        "truth_path": str(truth),
                        "group": f"synthetic-event-{index}",
                        "domain": "synthetic-domain",
                        "split": "test",
                    }
                ],
            },
        )
        paths.append(
            batch.register(
                root,
                manifest,
                versions,
                [1],
                **(
                    {"profile": "production", "judge_spec": judge_spec} if index else {}
                ),
            )
        )
    selection = paths[0]
    selected = batch.read_registration(selection)
    decision = {"goals": {goal: {"status": "PASS"} for goal in ("G1", "G2")}}
    original_verify = batch.verified_decision
    monkeypatch.setattr(
        batch,
        "verified_decision",
        lambda path: copy.deepcopy(decision)
        if path == selection
        else original_verify(path),
    )
    # Drive the real journal across the development promotion. Only the G1/G2
    # measurement boundary above is stubbed; these are not measured acceptance data.
    lifecycle.initialize(root, versions["before"])
    lifecycle.start_round(
        root, "synthetic hypothesis", {"max_runs": 2, "max_reserved_seconds": 1200}
    )
    lifecycle.attach(root, selection)
    lifecycle.resolve(root)
    assert (
        lifecycle.status(root)["incumbent"]["identity"]
        == selected["versions"]["after"]["identity"]
    )
    return paths, decision


def seal(registered, **kwargs):
    paths = registered[0]
    return confirmations.seal(
        paths[0],
        paths[1:3],
        kwargs.get(
            "budget",
            {
                "max_runs": 4,
                "max_reserved_seconds": 2400,
            },
        ),
    )


def test_seal_keeps_original_baseline_after_development_promotion(registered):
    paths, _ = registered
    plan = seal(registered)
    for path in paths[1:3]:
        reservation = confirmations.reserve_run(path, "before")
        assert reservation["budget"] == {"runs": 1, "timeout_s": 600}
        checked, _, records = confirmations.verify_plan(path)
        assert checked == plan
        assert records[1]["versions"]["before"]["identity"]["git_commit"] == "before"
        assert records[1]["versions"]["after"]["identity"]["git_commit"] == "after"
    with pytest.raises(ValueError, match="seal both"):
        confirmations.seal(paths[0], [paths[3], paths[2]], plan["budget"])


@pytest.mark.parametrize(
    "field,value", [("max_runs", 3), ("max_reserved_seconds", 2399), ("max_runs", True)]
)
def test_insufficient_confirmation_budget_does_not_seal(registered, field, value):
    budget = {"max_runs": 4, "max_reserved_seconds": 2400, field: value}
    with pytest.raises(ValueError, match="budget"):
        seal(registered, budget=budget)
    assert not (registered[0][0] / "confirmation-plan.json").exists()


@pytest.mark.parametrize("role", ["before", "after"])
def test_cannot_seal_after_either_confirmation_has_started(registered, role):
    paths, _ = registered
    batch.write_new(paths[2] / f"{role}.started.json", {})
    with pytest.raises(ValueError, match="before any run"):
        seal(registered)


@pytest.mark.parametrize("damage", ["missing-peer", "plan", "selection", "reservation"])
def test_broken_confirmation_chain_rejected(registered, damage):
    paths, decision = registered
    seal(registered)
    started = {
        "confirmation_reservation": confirmations.reserve_run(paths[1], "before")
    }
    if damage == "missing-peer":
        (paths[2] / "confirmation.json").unlink()
    elif damage == "plan":
        path = paths[0] / "confirmation-plan.json"
        value = json.loads(path.read_text())
        value["budget"]["max_runs"] += 1
        path.write_text(json.dumps(value))
    elif damage == "selection":
        decision["goals"]["G2"]["status"] = "FAIL"
    else:
        started["confirmation_reservation"]["role"] = "after"
    with pytest.raises((ValueError, OSError)):
        confirmations.verify_reservation(paths[1], "before", started)


def test_third_confirmation_and_replacement_pair_refused(registered):
    paths, _ = registered
    with pytest.raises(ValueError, match="exactly two"):
        confirmations.seal(
            paths[0], paths[1:], {"max_runs": 6, "max_reserved_seconds": 3600}
        )
    seal(registered)
    with pytest.raises((ValueError, FileExistsError)):
        confirmations.seal(
            paths[0],
            [paths[1], paths[3]],
            {"max_runs": 4, "max_reserved_seconds": 2400},
        )
    with pytest.raises(ValueError, match="regular file"):
        confirmations.reserve_run(paths[3], "before")


def test_rollback_blocks_confirmation_execution(registered):
    paths, _ = registered
    seal(registered)
    lifecycle.rollback(paths[0].parent.parent, "synthetic rollback")
    with pytest.raises(ValueError, match="selected G2 incumbent"):
        batch.run(paths[1], "before")
    assert not (paths[1] / "before.started.json").exists()


def test_production_cannot_consume_a_development_selection(registered):
    paths, _ = registered
    root = paths[0].parent.parent
    before = lifecycle.status(root)
    with pytest.raises(ValueError, match="seal production confirmations"):
        lifecycle.attach(root, paths[1])
    assert lifecycle.status(root) == before


@pytest.mark.parametrize("crash", [False, True])
def test_batch_runner_consumes_sealed_roles_and_verifies_receipts(
    registered, monkeypatch, crash
):
    paths, _ = registered
    seal(registered)
    directory = paths[1]
    record = batch.read_registration(directory)
    actual_subprocess = batch.subprocess.run
    calls = []

    def evaluator(argv, **kwargs):
        if argv[0] == "git":
            return actual_subprocess(argv, **kwargs)
        calls.append(kwargs)
        if crash:
            raise batch.subprocess.TimeoutExpired(argv, kwargs["timeout"])
        out = directory / ("before" if len(calls) == 1 else "after")
        rows = [
            {
                **r,
                "status": "completed",
                "profile": "production",
                "execution": {"returncode": 0, "timed_out": False},
                "assessment": {
                    "profile": "production",
                    "development_score": 0.1,
                    "judge": record["production_judge"]["provenance"],
                },
                "answer_sha256": None,
            }
            for r in record["expected_runs"]
        ]
        for row in rows:
            folder = out / row["case_id"] / f"seed-{row['seed']}"
            folder.mkdir(parents=True)
            batch.write_new(folder / "result.json", row)
        batch.write_new(
            out / "report.json",
            {
                "runs": rows,
                "profile": "production",
                "mode": "grounded",
                "provenance": record["versions"][out.name]["identity"],
            },
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(batch.subprocess, "run", evaluator)
    if crash:
        with pytest.raises(batch.subprocess.TimeoutExpired):
            batch.run(directory, "before")
        assert not (directory / "before.receipt.json").exists()
    else:
        batch.run(directory, "before")
        batch.run(directory, "after")
        assert batch.verified_reports(directory)[2] == record
        path = directory / "before.started.json"
        started = json.loads(path.read_text())
        started["confirmation_reservation"]["budget"]["runs"] = 0
        path.write_text(json.dumps(started))
        with pytest.raises(ValueError, match="budget reservation"):
            batch.verified_reports(directory)
    assert calls[0]["timeout"] == 600
    with pytest.raises(FileExistsError, match="do not retry"):
        batch.run(directory, "before")
