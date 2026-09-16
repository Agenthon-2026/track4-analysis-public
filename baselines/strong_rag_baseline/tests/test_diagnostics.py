"""Optional diagnostics explain fallbacks without changing submitted answers."""

from __future__ import annotations

import json
import os
from copy import deepcopy

import pytest

from baselines.strong_rag_baseline import cli
from baselines.strong_rag_baseline.agent import EntityResult
from baselines.strong_rag_baseline.client import MockModelClient


SECRET = "synthetic-private-token-do-not-log"


@pytest.fixture
def unit(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = "Acme expects revenue growth of 5 percent next quarter, in a range of 4 to 6 percent."
    (corpus / "release.json").write_text(
        json.dumps({"doc_date": "2024-01-10", "text": text}), encoding="utf-8"
    )
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "task_id": "synthetic-diagnostics",
                "cutoff_date": "2024-01-31",
                "target": {
                    "type": "regression",
                    "name": "revenue_growth_pct",
                    "unit": "percent",
                },
                "entities": [{"entity_id": "ACME", "name": "Acme"}],
                "interval_level": 0.9,
            }
        ),
        encoding="utf-8",
    )
    reply = {
        "point_forecast": 5.0,
        "interval": {"level": 0.9, "lo": 4.0, "hi": 6.0},
        "evidence": [
            {
                "doc_id": "release",
                "quote": text,
                "claim": "Acme expects 5 percent growth.",
            }
        ],
    }
    return task, corpus, reply


def test_entity_result_positional_construction_is_compatible():
    result = EntityResult({}, 0, "", "unchanged rationale")
    assert result.rationale == "unchanged rationale"
    assert result.source == "grounded"
    assert result.fallback_reason is None


@pytest.mark.parametrize(
    "stage", ["model_request", "model_json", "model_evidence", "model_prediction"]
)
def test_fallback_stage_is_exact_and_never_contains_raw_errors(unit, tmp_path, stage):
    task, corpus, valid_reply = unit
    reply = deepcopy(valid_reply)
    reply["private_debug"] = SECRET

    def request_failure(*args):
        raise RuntimeError(SECRET)

    if stage == "model_request":
        client = MockModelClient(request_failure)
    elif stage == "model_json":
        client = MockModelClient(SECRET)
    else:
        if stage == "model_evidence":
            reply["evidence"][0]["quote"] = SECRET
        else:
            reply["point_forecast"] = SECRET
        client = MockModelClient(json.dumps(reply))
    diagnostics_path = tmp_path / "diagnostics.json"
    actual = cli.run(
        task,
        corpus,
        tmp_path / "answer.json",
        client,
        5,
        diagnostics_path=diagnostics_path,
    )
    expected = cli.run(task, corpus, tmp_path / "grounded.json", None, 5)
    assert actual == expected
    serialized = diagnostics_path.read_text(encoding="utf-8")
    assert SECRET not in serialized
    diagnostics = json.loads(serialized)
    assert diagnostics["mode"] == "model"
    [entity] = diagnostics["entities"]
    assert entity["source"] == "grounded"
    assert entity["fallback_reason"] == stage
    assert entity["elapsed_s"] >= 0


def test_model_success_records_only_bounded_diagnostics(unit, tmp_path, monkeypatch):
    task, corpus, reply = unit
    reply["private_debug"] = SECRET
    clock = iter([10.0, 12.5, 30.0, 35.0])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(clock))
    client = MockModelClient(json.dumps(reply))
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    diagnostics_path = tmp_path / "reports" / "diagnostics.json"
    without = cli.run(task, corpus, first, client, 5)
    with_diagnostics = cli.run(
        task, corpus, second, client, 5, diagnostics_path=diagnostics_path
    )
    assert without == with_diagnostics
    assert first.read_bytes() == second.read_bytes()
    assert "elapsed_s" not in second.read_text(encoding="utf-8")
    assert json.loads(diagnostics_path.read_text(encoding="utf-8")) == {
        "request_ledger": None,
        "schema_version": "1",
        "task_id": "synthetic-diagnostics",
        "mode": "model",
        "entities": [
            {
                "entity_id": "ACME",
                "source": "model",
                "fallback_reason": None,
                "elapsed_s": 5.0,
                "dropped_claims": 0,
            }
        ],
    }
    assert SECRET not in diagnostics_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("forced", [False, True])
def test_cli_distinguishes_forced_grounded_and_no_endpoint(
    unit, tmp_path, monkeypatch, forced
):
    task, corpus, _ = unit
    monkeypatch.delenv("MODEL_ENDPOINT", raising=False)
    monkeypatch.delenv("T4_LOCAL_LLAMA", raising=False)
    diagnostics_path = tmp_path / "diagnostics.json"
    argv = [
        "--task",
        str(task),
        "--corpus",
        str(corpus),
        "--out",
        str(tmp_path / "answer.json"),
        "--diagnostics",
        str(diagnostics_path),
    ]
    if forced:
        monkeypatch.setenv("MODEL_ENDPOINT", "https://unused.example/v1")
        argv.append("--mock")
    assert cli.main(argv) == 0
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["mode"] == "grounded"
    [entity] = diagnostics["entities"]
    assert entity["source"] == "grounded"
    assert entity["fallback_reason"] == ("forced_grounded" if forced else "no_endpoint")


@pytest.mark.parametrize(
    "destination",
    ["answer", "task", "corpus_file", "new_corpus_file", "symlink", "hardlink"],
)
def test_diagnostics_refuses_input_and_answer_aliases(unit, tmp_path, destination):
    task, corpus, _ = unit
    out = tmp_path / "answer.json"
    out.write_text("existing answer", encoding="utf-8")
    if destination == "answer":
        diagnostics_path = out
    elif destination == "task":
        diagnostics_path = task
    elif destination == "corpus_file":
        diagnostics_path = corpus / "release.json"
    elif destination == "new_corpus_file":
        diagnostics_path = corpus / "new.json"
    else:
        diagnostics_path = tmp_path / "alias.json"
        if destination == "symlink":
            diagnostics_path.symlink_to(task)
        else:
            os.link(corpus / "release.json", diagnostics_path)
    before = {path: path.read_bytes() for path in (task, corpus / "release.json", out)}
    with pytest.raises(ValueError, match="diagnostics path"):
        cli.run(task, corpus, out, None, 5, diagnostics_path=diagnostics_path)
    assert {path: path.read_bytes() for path in before} == before


def test_cli_rejects_unsafe_diagnostics_before_running(unit, tmp_path, monkeypatch):
    task, corpus, _ = unit

    def must_not_run(*args, **kwargs):
        pytest.fail("unsafe diagnostics must be rejected before inference")

    monkeypatch.setattr(cli, "run", must_not_run)
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--task",
                str(task),
                "--corpus",
                str(corpus),
                "--out",
                str(tmp_path / "answer.json"),
                "--diagnostics",
                str(task),
            ]
        )
    assert error.value.code == 2


def test_failed_answer_does_not_write_diagnostics(unit, tmp_path, monkeypatch):
    task, corpus, _ = unit
    diagnostics_path = tmp_path / "diagnostics.json"

    def invalid_answer(*args):
        raise ValueError("synthetic invalid answer")

    monkeypatch.setattr(cli, "build_answer", invalid_answer)
    with pytest.raises(ValueError, match="invalid answer"):
        cli.run(
            task,
            corpus,
            tmp_path / "answer.json",
            None,
            5,
            diagnostics_path=diagnostics_path,
        )
    assert not diagnostics_path.exists()


def test_offline_diagnostics_record_zero_requests_without_inventing_model_usage(
    unit, tmp_path
):
    task, corpus, _ = unit
    path = tmp_path / "diagnostics.json"
    cli.run(task, corpus, tmp_path / "answer.json", None, 5, diagnostics_path=path)
    ledger = json.loads(path.read_text())["request_ledger"]
    assert ledger == {
        "version": 1,
        "source": "offline",
        "attempts_used": 0,
        "attempts": [],
    }
