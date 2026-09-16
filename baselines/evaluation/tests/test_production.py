"""Pinned production-profile batches never inherit a different ambient judge."""

import copy
import json
from types import SimpleNamespace

import pytest

from baselines.evaluation import batch, production
from baselines.evaluation.tests import test_batch
from qfbench2_track_analysis.judge_factory import compute_cache_tree_digest

setup = test_batch.setup


@pytest.fixture
def judge_spec(tmp_path):
    cache = tmp_path / "judge-cache"
    cache.mkdir()
    (cache / "synthetic-weights").write_text("test-only; not a production model")
    path = tmp_path / "judge.json"
    path.write_text(
        json.dumps(
            {
                "model_ids": ["synthetic/nli"],
                "model_revisions": {"synthetic/nli": "a" * 40},
                "tokenizer_digest": "sha256:" + "b" * 64,
                "cache_tree_digest": compute_cache_tree_digest(cache),
                "cache_dir": str(cache),
            }
        )
    )
    return path


def registered(setup, judge_spec):
    root, manifest, versions, _ = setup
    return batch.register(
        root, manifest, versions, [1], profile="production", judge_spec=judge_spec
    )


def test_production_requires_explicit_configuration_before_event_reservation(
    setup, monkeypatch
):
    root, manifest, versions, _ = setup
    monkeypatch.setenv("QFBENCH2_T4_JUDGE_SPEC", "ambient-is-not-explicit")
    with pytest.raises(ValueError, match="explicit judge"):
        batch.register(root, manifest, versions, [1], profile="production")
    assert not root.exists()


def test_smoke_cannot_carry_judge_and_invalid_pin_does_not_reserve(setup, judge_spec):
    root, manifest, versions, _ = setup
    with pytest.raises(ValueError, match="only production"):
        batch.register(root, manifest, versions, [1], judge_spec=judge_spec)
    raw = json.loads(judge_spec.read_text())
    raw["model_revisions"]["synthetic/nli"] = "main"
    judge_spec.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="40-hex"):
        registered(setup, judge_spec)
    assert not root.exists()


@pytest.mark.parametrize("change", ["spec", "cache", "symlink"])
def test_production_artifact_changes_refuse_before_burning_run(
    setup, judge_spec, change
):
    directory = registered(setup, judge_spec)
    if change == "spec":
        judge_spec.write_text(judge_spec.read_text() + "\n")
    elif change == "cache":
        (judge_spec.parent / "judge-cache/synthetic-weights").write_text("changed")
    else:
        target = judge_spec.with_name("moved.json")
        judge_spec.rename(target)
        judge_spec.symlink_to(target)
    with pytest.raises(ValueError):
        batch.run(directory, "before")
    assert not (directory / "before.started.json").exists()


def test_each_report_row_must_bind_the_registered_judge(setup, judge_spec):
    directory = registered(setup, judge_spec)
    record = batch.read_registration(directory)
    row = {
        **record["expected_runs"][0],
        "profile": "production",
        "assessment": {
            "profile": "production",
            "judge": record["production_judge"]["provenance"],
        },
    }
    report = {
        "runs": [row],
        "provenance": record["versions"]["before"]["identity"],
        "mode": "grounded",
        "profile": "production",
    }
    batch.validate_report(report, record, "before")
    changed = copy.deepcopy(report)
    changed["runs"][0]["assessment"]["judge"]["model_revisions"]["synthetic/nli"] = (
        "c" * 40
    )
    with pytest.raises(ValueError, match="report judge"):
        batch.validate_report(changed, record, "before")


@pytest.mark.parametrize("mutate_cache", [False, True])
def test_run_overrides_ambient_judge_and_rechecks_cache_after_execution(
    setup, judge_spec, monkeypatch, mutate_cache
):
    directory = registered(setup, judge_spec)
    record = batch.read_registration(directory)
    monkeypatch.setenv("QFBENCH2_T4_JUDGE_SPEC", "wrong")
    monkeypatch.setenv("QFBENCH2_T4_MODEL_CACHE", "wrong")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    observed = []
    real_run = batch.subprocess.run

    def evaluator(argv, **kwargs):
        if argv[0] == "git":
            return real_run(argv, **kwargs)
        observed.append(kwargs["env"])
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
        report = {
            "runs": rows,
            "provenance": record["versions"]["before"]["identity"],
            "mode": "grounded",
            "profile": "production",
        }
        for row in rows:
            folder = directory / "before" / row["case_id"] / f"seed-{row['seed']}"
            folder.mkdir(parents=True)
            (folder / "result.json").write_text(json.dumps(row))
        (directory / "before/report.json").write_text(json.dumps(report))
        if mutate_cache:
            (judge_spec.parent / "judge-cache/synthetic-weights").write_text(
                "changed during run"
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(batch.subprocess, "run", evaluator)
    if mutate_cache:
        with pytest.raises(ValueError, match="cache differs"):
            batch.run(directory, "before")
    else:
        batch.run(directory, "before")
    assert observed[0]["QFBENCH2_T4_JUDGE_SPEC"] == str(judge_spec)
    assert observed[0]["QFBENCH2_T4_MODEL_CACHE"] == str(
        judge_spec.parent / "judge-cache"
    )
    assert observed[0]["HF_HUB_OFFLINE"] == observed[0]["TRANSFORMERS_OFFLINE"] == "1"
    assert (directory / "before.started.json").exists()
    assert (directory / "before.receipt.json").exists() is not mutate_cache


def test_smoke_registration_remains_backward_compatible(setup):
    root, manifest, versions, _ = setup
    record = batch.read_registration(batch.register(root, manifest, versions, [1]))
    assert "production_judge" not in record
    assert production.evaluation_environment(record) is None
