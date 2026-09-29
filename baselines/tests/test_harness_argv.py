"""The baseline must run under the EXACT argv the scoring harness issues.

`test_cli_exemplar.py` calls `run()` directly, which is why the shipped baseline could be
unrunnable without any test noticing: `run()` never sees argv, so it cannot see a CLI contract
mismatch. These tests go through `main()` — and through the real subprocess entry point the
Dockerfile's ENTRYPOINT uses — with the argv from SUBMISSION_CLI.md:

    docker run <image> analyze --task /input/task.json --corpus /input/corpus/ \
                               --out /output/answer.json

Two defects motivated them, both of which made the baseline exit 2 on every single unit:

1. the harness passed `--question /input/q.json`, a flag this CLI does not define naming a file
   no Track 4 unit contains (units ship `task.json`; `question.json` was a redirect stub and
   has since been removed entirely);
2. the CLI did not accept the leading `analyze` verb, which the harness passes as the container
   command — so even with the flag corrected, argparse rejected it as an unrecognized positional.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_BASELINES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BASELINES))

from baseline_agent.cli import VERB, main  # noqa: E402

_REPO = _BASELINES.parent
_UNIT = _REPO / "units" / "t4-EXAMPLE-eps-beat"
_ENTRYPOINT = _BASELINES / "baseline_agent.py"


def _harness_argv(unit: Path, out: Path) -> list[str]:
    """The argv the harness issues, with /input and /output resolved to local paths."""
    return [VERB, "--task", str(unit / "task.json"),
            "--corpus", str(unit / "corpus"), "--out", str(out)]


def test_verb_is_analyze() -> None:
    """The verb is published in SUBMISSION_CLI.md; renaming it breaks every submission."""
    assert VERB == "analyze"


def test_main_accepts_the_harness_argv(tmp_path: Path, monkeypatch) -> None:
    """The whole point: `analyze --task ... --corpus ... --out ...` must run, not exit 2."""
    out = tmp_path / "answer.json"
    monkeypatch.setattr(sys, "argv", ["baseline_agent.py", *_harness_argv(_UNIT, out)])
    main()
    answer = json.loads(out.read_text())
    assert answer["task_id"] == "t4-EXAMPLE-eps-beat"
    assert answer["entity_predictions"]


def test_entrypoint_subprocess_runs_under_the_harness_argv(tmp_path: Path) -> None:
    """Same, through the real process boundary the Dockerfile ENTRYPOINT crosses.

    The ENTRYPOINT is `python baseline_agent.py`, so the harness's verb and flags land on this
    script's argv exactly as constructed here. Asserting the exit code is the assertion that
    matters: the shipped baseline returned 2.
    """
    out = tmp_path / "answer.json"
    proc = subprocess.run(
        [sys.executable, str(_ENTRYPOINT), *_harness_argv(_UNIT, out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"baseline exited {proc.returncode} under the harness argv.\n"
        f"stderr:\n{proc.stderr}"
    )
    assert out.exists()


def test_verb_is_optional_for_hand_invocation(tmp_path: Path, monkeypatch) -> None:
    """Dropping the verb must keep working — the README and both baselines document that form."""
    out = tmp_path / "answer.json"
    argv = _harness_argv(_UNIT, out)[1:]  # strip the verb
    monkeypatch.setattr(sys, "argv", ["baseline_agent.py", *argv])
    main()
    assert out.exists()


def test_a_wrong_verb_is_rejected_loudly(tmp_path: Path, monkeypatch) -> None:
    """Fail closed. A verb we do not implement must not be silently swallowed as `analyze`."""
    out = tmp_path / "answer.json"
    monkeypatch.setattr(
        sys, "argv",
        ["baseline_agent.py", "simulate", "--task", str(_UNIT / "task.json"),
         "--corpus", str(_UNIT / "corpus"), "--out", str(out)],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert not out.exists()


def test_the_input_files_the_harness_names_actually_exist() -> None:
    """The T4 defect in one assertion: every path in the argv must be real.

    `--question /input/q.json` failed exactly here — no unit has ever contained `q.json`.
    """
    for unit in sorted((_REPO / "units").iterdir()):
        if not unit.is_dir():
            continue
        for arg in _harness_argv(unit, Path("/output/answer.json")):
            if arg.startswith("-") or arg == VERB or arg.startswith("/output"):
                continue
            assert Path(arg).exists(), f"{unit.name}: harness argv names missing path {arg}"


def test_question_json_stays_removed() -> None:
    """Pins WHY `--task`/`task.json` is the canonical side, so nobody 'restores' `--question`.

    `question.json` was a redirect stub kept for backward compatibility with tools that never
    shipped. It is gone from every unit and from `templates/`. The previous version of this guard
    skipped itself when the file was absent, which is the decoration this repo's fail-closed rule exists to
    forbid: the guard would have gone quietly green on the very change it was written to notice.
    Restoring the file has to be a deliberate act that turns this red, not a silent one.
    """
    strays = sorted(
        str(path.relative_to(_REPO))
        for path in _REPO.rglob("question.json")
        if ".git" not in path.parts
    )
    assert not strays, (
        "question.json is a removed format; task.json is the only canonical task input. "
        f"Found: {strays}. Reviving it means revisiting the CLI contract on purpose."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
