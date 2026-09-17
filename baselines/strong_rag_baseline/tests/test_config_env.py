"""The reference agent must read the env var the harness actually sets (finding H6,
prose-drift sweep 2026-08-27).

SUBMISSION_CLI.md's container-environment contract injects ``MODEL_NAME`` — the pinned
house-model id served at ``MODEL_ENDPOINT``. Before the fix, config.py read only
``MODEL_ID`` (a local-dev spelling no contract defines), so under the real harness the
agent posted ``"model": ""`` in every request. ``MODEL_ID`` survives as a local-dev
fallback only; ``MODEL_NAME`` wins when both are set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strong_rag_baseline.config import Config  # noqa: E402

_ENV_VARS = ("MODEL_NAME", "MODEL_ID", "MODEL_ENDPOINT", "MODEL_TOKEN")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_harness_injected_model_name_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "house-model-pin-2026-07")
    assert Config.from_env().model_id == "house-model-pin-2026-07"


def test_model_name_beats_local_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "house-model-pin-2026-07")
    monkeypatch.setenv("MODEL_ID", "qwen2.5:7b")
    assert Config.from_env().model_id == "house-model-pin-2026-07"


def test_model_id_survives_as_local_dev_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_ID", "qwen2.5:7b")
    assert Config.from_env().model_id == "qwen2.5:7b"


def test_chat_completions_url_adds_v1_for_the_injected_origin() -> None:
    """The harness injects the route origin; the API lives under /v1 (hub HOUSE-MODEL.md)."""
    from strong_rag_baseline.client import chat_completions_url

    assert (
        chat_completions_url("http://house-rehearsal.agenthon.internal:8443")
        == "http://house-rehearsal.agenthon.internal:8443/v1/chat/completions"
    )
    assert (
        chat_completions_url("http://house-rehearsal.agenthon.internal:8443/")
        == "http://house-rehearsal.agenthon.internal:8443/v1/chat/completions"
    )


def test_chat_completions_url_keeps_a_local_v1_base() -> None:
    from strong_rag_baseline.client import chat_completions_url

    assert chat_completions_url("http://localhost:11434/v1") == "http://localhost:11434/v1/chat/completions"
    assert chat_completions_url("http://localhost:11434/v1/") == "http://localhost:11434/v1/chat/completions"
