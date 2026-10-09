"""Test-wide isolation for the scorer's process-level state (scorer 5.3.0).

Imports nothing from this repository, so the stdlib-only firewall job (`pytest baselines`) still
collects without the toolkit.

* The production judge's device defaults to the CPU in tests. `auto` would pick a GPU on a
  machine that has one and move the stand-in pipelines the tests inject; a test that wants
  another device sets `QFBENCH2_T4_JUDGE_DEVICE` itself.
* Each test starts with an empty process cache of built production judges, so a judge built by
  one test is never reused by another.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _isolated_judge_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QFBENCH2_T4_JUDGE_DEVICE", "cpu")
    factory = sys.modules.get("qfbench2_track_analysis.judge_factory")
    if factory is not None:
        monkeypatch.setattr(factory, "_PROCESS_JUDGES", {})
