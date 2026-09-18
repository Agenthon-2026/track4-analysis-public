"""Harness model and seed settings take precedence over developer defaults."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strong_rag_baseline.config import Config  # noqa: E402

_ENV_VARS = (
    "MODEL_NAME",
    "MODEL_ID",
    "MODEL_ENDPOINT",
    "MODEL_TOKEN",
    "QFBENCH_SEED",
    "T4_SEED",
    "T4_LOCAL_LLAMA",
    "T4_UNIT_TIMEOUT_S",
    "T4_MODEL_TIMEOUT_S",
    "T4_MODEL_RETRIES",
    "T4_TOP_K",
    "T4_MAX_TOKENS",
)


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


def test_harness_endpoint_and_seed_override_developer_settings(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "https://house.example")
    monkeypatch.setenv("QFBENCH_SEED", "1234")
    monkeypatch.setenv("T4_SEED", "77")
    config = Config.from_env()
    assert config.model_endpoint == "https://house.example"
    assert config.seed == 1234
    assert config.local_llama is False


def test_invalid_budget_settings_cannot_disable_deadlines(monkeypatch):
    monkeypatch.setenv("T4_UNIT_TIMEOUT_S", "inf")
    monkeypatch.setenv("T4_MODEL_TIMEOUT_S", "nan")
    monkeypatch.setenv("T4_MODEL_RETRIES", "99999")
    monkeypatch.setenv("T4_TOP_K", "nonsense")
    config = Config.from_env()
    assert 0 < config.unit_timeout_s <= 480
    assert 0 < config.timeout_s <= 60
    assert config.max_retries == 3
    assert config.top_k == 10


def test_chat_completions_url_adds_v1_for_the_injected_origin() -> None:
    """The harness injects the route origin; the API lives under /v1."""
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
