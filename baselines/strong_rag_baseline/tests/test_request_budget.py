"""Every attempted HTTP exchange shares a hard unit budget and an auditable ledger."""

import io
import json
from dataclasses import replace

import pytest

from baselines.strong_rag_baseline.client import HTTPModelClient, ModelBudgetExceeded
from baselines.strong_rag_baseline.config import (
    Config,
    HOUSE_MAX_REQUESTS,
    HOUSE_MAX_OUTPUT_TOKENS,
)


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "https://house.example/v1")
    monkeypatch.setenv("MODEL_TOKEN", "synthetic-secret")
    monkeypatch.setattr(
        "baselines.strong_rag_baseline.client.time.sleep", lambda _: None
    )
    return replace(Config.from_env(), max_retries=1)


def response(usage=None):
    return io.BytesIO(
        json.dumps(
            {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": usage,
            }
        ).encode()
    )


def test_hard_cap_is_shared_across_entities(config, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: calls.append(1) or response()
    )
    client = HTTPModelClient(config)
    for _ in range(HOUSE_MAX_REQUESTS):
        assert client.complete("system", "entity") == "{}"
    with pytest.raises(ModelBudgetExceeded):
        client.complete("system", "one more entity")
    assert len(calls) == HOUSE_MAX_REQUESTS
    ledger = client.request_ledger()
    assert ledger["attempts_used"] == len(calls)
    assert ledger["budget_blocks"] == 1
    assert all(row["status"] == "success" for row in ledger["attempts"])


def test_retries_and_failures_consume_same_budget(config, monkeypatch):
    calls = []

    def request(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("synthetic-secret")

    monkeypatch.setattr("urllib.request.urlopen", request)
    client = HTTPModelClient(replace(config, max_requests=2, max_retries=3))
    with pytest.raises(ModelBudgetExceeded):
        client.complete("s", "u")
    with pytest.raises(ModelBudgetExceeded):
        client.complete("s", "another entity")
    ledger = client.request_ledger()
    assert len(calls) == 2
    assert [row["status"] for row in ledger["attempts"]] == ["error", "error"]
    assert "synthetic-secret" not in json.dumps(ledger)
    assert "house.example" not in json.dumps(ledger)


def test_provider_usage_is_optional_never_invented(config, monkeypatch):
    replies = [response(), response({"prompt_tokens": 10, "completion_tokens": 2})]
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: replies.pop(0)
    )
    client = HTTPModelClient(config)
    client.complete("s", "u")
    client.complete("s", "u")
    rows = client.request_ledger()["attempts"]
    assert rows[0]["prompt_tokens"] is None
    assert rows[0]["completion_tokens"] is None
    assert rows[1]["prompt_tokens"] == 10
    assert rows[1]["completion_tokens"] == 2
    rows[0]["status"] = "tampered"
    assert client.request_ledger()["attempts"][0]["status"] == "success"


def test_truncated_reply_is_counted(config, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b'{"choices":')
    )
    client = HTTPModelClient(config)
    with pytest.raises(RuntimeError):
        client.complete("s", "u")
    record = client.request_ledger()["attempts"][0]
    assert record["status"] == "error"
    assert record["response_bytes"] > 0


def test_output_cap_cannot_be_raised_by_environment(config, monkeypatch):
    monkeypatch.setenv("T4_MAX_TOKENS", "99999")
    config = Config.from_env()
    assert config.max_tokens == HOUSE_MAX_OUTPUT_TOKENS
    calls = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, **kwargs: calls.append(json.loads(request.data)) or response(),
    )
    HTTPModelClient(config).complete("s", "u")
    assert calls[0]["max_tokens"] == HOUSE_MAX_OUTPUT_TOKENS


@pytest.mark.parametrize(
    "changes",
    [
        {"max_requests": 26},
        {"max_requests": True},
        {"max_tokens": 4096},
        {"max_tokens": float("inf")},
    ],
)
def test_direct_config_cannot_bypass_caps(config, changes):
    with pytest.raises(ValueError, match="ceiling"):
        HTTPModelClient(replace(config, **changes))


def test_reported_output_overrun_is_not_a_success(config, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: response({"completion_tokens": config.max_tokens + 1}),
    )
    client = HTTPModelClient(config)
    with pytest.raises(RuntimeError):
        client.complete("s", "u")
    assert client.request_ledger()["attempts"][0]["status"] == "error"
