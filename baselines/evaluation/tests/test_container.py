"""Container failures and participant artifacts must not corrupt assessment."""

import json
import subprocess
from pathlib import Path

import pytest

from baselines.evaluation import __main__ as runner


def fake_docker(monkeypatch, *, state=None, failure=None, artifacts=None):
    state = state or {"Status": "exited", "ExitCode": 0, "Error": ""}
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        command = argv[1]
        code = 0
        stdout = ""
        if command == failure:
            raise subprocess.CalledProcessError(1, argv, stderr="daemon unavailable")
        if command == "create":
            stdout = "synthetic-container\n"
        elif command == "start":
            code = state["ExitCode"]
        elif command == "inspect":
            stdout = json.dumps(
                [
                    {
                        "State": {**state, "OOMKilled": False},
                        "Image": "test",
                        "HostConfig": {
                            "NanoCpus": 1000000000,
                            "Memory": 1024**3,
                            "NetworkMode": "none",
                        },
                    }
                ]
            )
        elif command == "cp" and argv[2].startswith("synthetic-container:"):
            if artifacts:
                artifacts(Path(argv[3]))
        if code and kwargs.get("check"):
            raise subprocess.CalledProcessError(code, argv)
        return subprocess.CompletedProcess(argv, code, stdout, "")

    monkeypatch.setattr(runner.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("failure", ["start", "inspect"])
def test_daemon_failures_abort_instead_of_becoming_participant_scores(
    tmp_path, monkeypatch, failure
):
    calls = fake_docker(monkeypatch, failure=failure)
    with pytest.raises(subprocess.CalledProcessError):
        runner.run_container(tmp_path, tmp_path, image="test", seed=1, timeout=1)
    assert calls[-1][1:3] == ["rm", "--force"]


@pytest.mark.parametrize(
    "state",
    [
        {"Status": "created", "ExitCode": 0, "Error": ""},
        {"Status": "exited", "ExitCode": 128, "Error": "runtime could not start"},
    ],
)
def test_container_runtime_failure_is_not_a_submission_failure(
    tmp_path, monkeypatch, state
):
    fake_docker(monkeypatch, state=state)
    with pytest.raises(RuntimeError, match="container did not complete"):
        runner.run_container(tmp_path, tmp_path, image="test", seed=1, timeout=1)


def test_nonzero_submission_exit_uses_container_state(tmp_path, monkeypatch):
    calls = fake_docker(
        monkeypatch, state={"Status": "exited", "ExitCode": 7, "Error": ""}
    )
    result = runner.run_container(tmp_path, tmp_path, image="test", seed=1, timeout=1)
    assert result["returncode"] == 7
    assert result["timed_out"] is False
    assert not any(
        c[1] == "cp" and c[2].startswith("synthetic-container:") for c in calls
    )


def test_artifacts_cannot_replace_evaluator_evidence(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    (output / "evidence.json").write_text('{"owner":"evaluator"}')

    def artifacts(destination):
        (destination / "evidence.json").write_text('{"owner":"participant"}')
        (destination / "answer.json").write_text('{"answer":"preserved"}')
        (destination / "diagnostics.json").write_text('{"entities":[]}')

    fake_docker(monkeypatch, artifacts=artifacts)
    result = runner.run_container(tmp_path, output, image="test", seed=1, timeout=1)
    assert result["returncode"] == 0
    assert json.loads((output / "evidence.json").read_text()) == {"owner": "evaluator"}
    assert json.loads((output / "answer.json").read_text()) == {"answer": "preserved"}
    assert json.loads((output / "diagnostics.json").read_text()) == {"entities": []}


def test_output_symlinks_are_not_read_as_host_files(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    host_file = tmp_path / "private.json"
    host_file.write_text('{"host":"private"}')

    def artifacts(destination):
        (destination / "answer.json").symlink_to(host_file)

    fake_docker(monkeypatch, artifacts=artifacts)
    result = runner.run_container(tmp_path, output, image="test", seed=1, timeout=1)
    assert not (output / "answer.json").is_symlink()
    assert not (output / "answer.json").exists()
    assert result["artifact_errors"] == ["answer.json: expected a regular file"]
    assert json.loads(host_file.read_text()) == {"host": "private"}
