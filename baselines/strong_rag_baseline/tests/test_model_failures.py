"""Synthetic model-contract failures must fall back as a complete entity."""

from __future__ import annotations

import io
import json
from http.client import BadStatusLine, IncompleteRead
from copy import deepcopy
from dataclasses import replace

import pytest

from baselines.strong_rag_baseline import agent, cli
from baselines.strong_rag_baseline.client import HTTPModelClient, MockModelClient
from baselines.strong_rag_baseline.config import Config
from baselines.strong_rag_baseline.indexer import Chunk, IndexedCorpus
from baselines.strong_rag_baseline.prompts import build_user_prompt
from baselines.strong_rag_baseline.retriever import BM25Index


TEXT = "Acme forecasts revenue growth of 5 percent, with a range of 4 to 6 percent."


@pytest.fixture
def model_case(monkeypatch):
    task = {
        "task_id": "synthetic-growth",
        "cutoff_date": "2024-01-31",
        "target": {"type": "regression", "name": "revenue_growth", "unit": "percent"},
        "interval_level": 0.9,
    }
    entity = {"entity_id": "ACME", "name": "Acme"}
    chunks = [
        Chunk("release", "2024-01-01", 0, len(TEXT), TEXT),
        Chunk("future", "2025-01-01", 0, len(TEXT), TEXT),
    ]
    corpus = IndexedCorpus(
        chunks,
        {c.doc_id: c.text for c in chunks},
        {c.doc_id: c.doc_date for c in chunks},
    )
    index = BM25Index(chunks, task["cutoff_date"])
    fallback = {
        "entity_id": "ACME",
        "point_forecast": 2.0,
        "interval": {"level": 0.9, "lo": 1.0, "hi": 3.0},
        "claims": [
            {"doc_id": "release", "span_start": 0, "span_end": 4, "claim": "fallback"}
        ],
    }
    monkeypatch.setattr(
        agent,
        "run_entity_grounded",
        lambda *args, **kwargs: agent.EntityResult(deepcopy(fallback), 0, ""),
    )
    reply = {
        "point_forecast": 5.0,
        "interval": {"level": 0.9, "lo": 4.0, "hi": 6.0},
        "evidence": [
            {
                "doc_id": "release",
                "quote": TEXT,
                "claim": "Acme expects 5 percent growth.",
            }
        ],
    }
    return task, entity, index, corpus, fallback, reply


def _run(case, reply):
    task, entity, index, corpus, _, _ = case
    return agent.run_entity(
        task, entity, index, corpus, MockModelClient(reply), 5
    ).prediction


@pytest.mark.parametrize(
    "reply",
    ["", "null", "[]", "[{}]", "{} trailing", "{", "42", "[" * 1100 + "]" * 1100],
)
def test_malformed_model_reply_falls_back_atomically(model_case, reply):
    assert _run(model_case, reply) == model_case[4]


@pytest.mark.parametrize(
    "field,value",
    [
        ("point_forecast", float("nan")),
        ("point_forecast", float("inf")),
        ("point_forecast", True),
        ("point_forecast", None),
        ("interval", None),
        ("interval", []),
        ("interval", {"level": 0.9, "lo": 6.0, "hi": 4.0}),
        ("interval", {"level": 0.9, "lo": 4.0, "hi": float("inf")}),
        ("interval", {"level": 0.9, "lo": True, "hi": 6.0}),
        ("interval", {"level": 0.8, "lo": 4.0, "hi": 6.0}),
        ("interval", {"level": 0.9, "lo": 0.0, "hi": 1.0}),
        ("evidence", []),
        ("evidence", "invalid"),
    ],
)
def test_invalid_prediction_never_keeps_model_numbers_with_fallback_evidence(
    model_case, field, value
):
    reply = deepcopy(model_case[5])
    reply[field] = value
    assert _run(model_case, json.dumps(reply)) == model_case[4]


def test_off_vocabulary_label_falls_back_as_a_whole(model_case):
    model_case[0]["target"].update(type="classification", labels=["up", "down"])
    reply = deepcopy(model_case[5])
    reply["label"] = "maybe"
    assert _run(model_case, json.dumps(reply)) == model_case[4]


@pytest.mark.parametrize(
    "target",
    [
        {"type": "regression", "name": "credit_event_probability"},
        {"type": "regression", "name": "revenue_growth", "maximum": 3.0},
    ],
)
def test_target_domain_violations_use_the_shared_fallback(model_case, target):
    model_case[0]["target"] = target
    assert _run(model_case, json.dumps(model_case[5])) == model_case[4]


@pytest.mark.parametrize(
    "evidence",
    [
        {"doc_id": "future", "quote": TEXT, "claim": "future evidence"},
        {"doc_id": "missing", "quote": TEXT, "claim": "missing document"},
        {
            "doc_id": "release",
            "quote": "Acme forecasts enormous growth",
            "claim": "overlapping words",
        },
        {
            "doc_id": "release",
            "quote": TEXT.replace(" ", "  "),
            "claim": "not verbatim",
        },
        {"doc_id": "release", "quote": "", "claim": "empty quote"},
        {"doc_id": ["release"], "quote": TEXT, "claim": "wrong id type"},
    ],
)
def test_any_unresolved_evidence_rejects_the_complete_prediction(model_case, evidence):
    reply = deepcopy(model_case[5])
    reply["evidence"].append(evidence)
    assert _run(model_case, json.dumps(reply)) == model_case[4]


def test_valid_model_reply_keeps_its_own_prediction_and_exact_offsets(model_case):
    pred = _run(model_case, json.dumps(model_case[5]))
    assert pred["point_forecast"] == 5.0
    assert pred["interval"] == model_case[5]["interval"]
    [claim] = pred["claims"]
    assert claim["span_start"] == 0 and claim["span_end"] == len(TEXT)
    assert claim["claim"] == model_case[5]["evidence"][0]["claim"]


def test_quotes_in_unretrieved_chunks_of_a_known_doc_are_rejected():
    text = "first excerpt. second excerpt."
    corpus = IndexedCorpus([], {"doc": text}, {"doc": "2024-01-01"})
    retrieved = [Chunk("doc", "2024-01-01", 0, 14, text[:14])]
    claims, dropped = agent._ground_claims(
        [{"doc_id": "doc", "quote": "second excerpt", "claim": "hidden text"}],
        corpus,
        retrieved,
    )
    assert claims == [] and dropped == 1


def test_http_failures_use_the_same_atomic_fallback(model_case):
    task, entity, index, corpus, expected, _ = model_case

    def timeout(*args):
        raise TimeoutError("simulated model timeout")

    assert (
        agent.run_entity(
            task, entity, index, corpus, MockModelClient(timeout), 5
        ).prediction
        == expected
    )


@pytest.mark.parametrize(
    "error", [BadStatusLine("broken status"), IncompleteRead(b"partial")]
)
def test_http_protocol_errors_fall_back_instead_of_aborting(
    model_case, monkeypatch, error
):
    monkeypatch.setenv("MODEL_ENDPOINT", "https://house.example/v1")
    calls = []

    class InterruptedResponse(io.BytesIO):
        def read1(self, size):
            raise error

    def failed_request(req, timeout):
        calls.append(req.full_url)
        if isinstance(error, IncompleteRead):
            return InterruptedResponse()
        raise error

    monkeypatch.setattr("urllib.request.urlopen", failed_request)
    client = HTTPModelClient(replace(Config.from_env(), max_retries=1))
    task, entity, index, corpus, expected, _ = model_case
    result = agent.run_entity(task, entity, index, corpus, client, 5)
    assert result.prediction == expected
    assert len(calls) == 1


def test_prompt_uses_shared_target_shape_and_omits_rank():
    prompt = build_user_prompt(
        {
            "target_type": "ranking",
            "target": {"name": "return"},
            "cutoff_date": "2024-01-01",
        },
        {"entity_id": "ACME"},
        [],
    )
    assert "return (ranking)" in prompt
    assert 'Do not emit "rank"' in prompt
    assert '"rank":' not in prompt


def test_prompt_includes_target_units_domain_and_resolution():
    prompt = build_user_prompt(
        {
            "target": {"type": "regression", "name": "credit_event_probability"},
            "cutoff_date": "2024-01-01",
            "resolution_date": "2024-12-31",
        },
        {"entity_id": "ACME"},
        [],
    )
    assert "TARGET UNIT: probability" in prompt
    assert "RESOLUTION DATE: 2024-12-31" in prompt
    assert "TARGET DOMAIN: minimum=0.0, maximum=1.0" in prompt


def test_http_house_endpoint_honors_model_seed_and_auth(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "https://house.example/v1")
    monkeypatch.setenv("MODEL_NAME", "pinned-house-model")
    monkeypatch.setenv("MODEL_TOKEN", "synthetic-test-token")
    monkeypatch.setenv("QFBENCH_SEED", "42")
    calls = []

    def request(req, timeout):
        calls.append((req, timeout))
        return io.BytesIO(
            json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", request)
    client = HTTPModelClient(Config.from_env())
    assert client.complete("system", "user") == "{}"
    req, timeout = calls[0]
    assert req.full_url == "https://house.example/v1/chat/completions"
    assert json.loads(req.data)["model"] == "pinned-house-model"
    assert json.loads(req.data)["seed"] == 42
    assert req.headers["Authorization"] == "Bearer synthetic-test-token"
    assert 0 < timeout <= 60


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"[]",
        b'{"choices": []}',
        b'{"choices": [{"message": {"content": null}}]}',
        b"x" * 1_048_577,
    ],
)
def test_malformed_http_response_has_bounded_retries(monkeypatch, body):
    monkeypatch.setenv("MODEL_ENDPOINT", "https://house.example/v1")
    monkeypatch.setattr(
        "baselines.strong_rag_baseline.client.time.sleep", lambda _: None
    )
    calls = []

    def request(req, timeout):
        calls.append(req.full_url)
        return io.BytesIO(body)

    monkeypatch.setattr("urllib.request.urlopen", request)
    client = HTTPModelClient(replace(Config.from_env(), max_retries=2))
    with pytest.raises(RuntimeError):
        client.complete("s", "u")
    assert len(calls) == 2


def test_continuously_streaming_response_cannot_keep_the_unit_alive(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "https://house.example/v1")
    now = [0.0]
    monkeypatch.setattr(
        "baselines.strong_rag_baseline.client.time.monotonic", lambda: now[0]
    )
    reads = []

    class SlowResponse(io.BytesIO):
        def read1(self, size):
            reads.append(size)
            now[0] += 3
            return b" "

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: SlowResponse()
    )
    client = HTTPModelClient(replace(Config.from_env(), unit_timeout_s=5))
    with pytest.raises(RuntimeError, match="deadline"):
        client.complete("s", "u")
    assert len(reads) == 2
    assert now[0] == 6


def test_unit_deadline_is_shared_across_entities_and_retries(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "https://house.example/v1")
    now = [0.0]
    calls = []
    monkeypatch.setattr(
        "baselines.strong_rag_baseline.client.time.monotonic", lambda: now[0]
    )
    monkeypatch.setattr(
        "baselines.strong_rag_baseline.client.time.sleep",
        lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    def timeout(req, timeout):
        calls.append(timeout)
        now[0] += timeout
        raise TimeoutError("simulated")

    monkeypatch.setattr("urllib.request.urlopen", timeout)
    config = replace(Config.from_env(), unit_timeout_s=5, timeout_s=3, max_retries=3)
    client = HTTPModelClient(config)
    with pytest.raises(RuntimeError):
        client.complete("s", "first entity")
    with pytest.raises(RuntimeError, match="deadline"):
        client.complete("s", "second entity")
    assert calls == [3, 1]
    assert now[0] == 5


def test_cli_selects_house_client_without_starting_local_server(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_ENDPOINT", "https://house.example/v1")
    monkeypatch.delenv("T4_LOCAL_LLAMA", raising=False)
    seen = []

    def no_local(**kwargs):
        pytest.fail("house endpoint must not start a local model server")

    def run(*args, **kwargs):
        seen.append((args[3], kwargs["grounded"]))
        return {"entity_predictions": []}

    monkeypatch.setattr(cli, "try_start_local_server", no_local)
    monkeypatch.setattr(cli, "run", run)
    assert (
        cli.main(
            [
                "analyze",
                "--task",
                str(tmp_path / "task.json"),
                "--corpus",
                str(tmp_path),
                "--out",
                str(tmp_path / "answer.json"),
            ]
        )
        == 0
    )
    assert isinstance(seen[0][0], HTTPModelClient)
    assert seen[0][1] is False


def test_budget_exhaustion_falls_back_without_a_second_http_request(
    model_case, monkeypatch
):
    monkeypatch.setenv("MODEL_ENDPOINT", "https://house.example/v1")
    calls = []
    reply = json.dumps(
        {
            "choices": [
                {
                    "message": {"content": json.dumps(model_case[5])},
                    "finish_reason": "stop",
                }
            ]
        }
    ).encode()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: calls.append(1) or io.BytesIO(reply),
    )
    client = HTTPModelClient(replace(Config.from_env(), max_requests=1, max_retries=1))
    task, entity, index, corpus, fallback, _ = model_case
    first = agent.run_entity(task, entity, index, corpus, client, 5)
    second = agent.run_entity(task, entity, index, corpus, client, 5)
    assert first.source == "model"
    assert second.prediction == fallback
    assert second.fallback_reason == "model_budget"
    assert calls == [1]
    assert client.request_ledger()["attempts_used"] == 1
