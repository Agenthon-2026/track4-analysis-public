"""One unusable model reply must cost one entity, not the whole unit.

`run_entity` used to parse the reply with no handler, and `cli.run` runs every entity in one
list comprehension, so a single reply with no JSON (or bad JSON) ended the run: no
answer.json at all. The entity whose reply cannot be used now gets a fallback row, listed in
`notes.fallback_entities`, and every other row is exactly what it would have been.

Every row must also pass the published analysis schema, because one invalid row fails
`g1_schema` for the WHOLE submission (`t4.schema_invalid`). `_assert_schema_valid` checks the
schema rules a row from this agent can break; it is written out here because these tests run
in the stdlib-only CI job, where `jsonschema` and the toolkit are not installed.

Standard library only, like the pipeline under test.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from baselines.strong_rag_baseline import cli
from baselines.strong_rag_baseline.client import (
    MockModelClient,
    ModelBudgetExhausted,
    ModelCallError,
)
from baselines.strong_rag_baseline.indexer import build_index

_REPO = Path(__file__).resolve().parents[3]


def _first_unit_of_type(target_type: str) -> Path:
    """The first public unit (sorted by name) of this target type with more than one entity."""
    for unit in sorted((_REPO / "units").iterdir()):
        task_path = unit / "task.json"
        if not task_path.is_file():
            continue
        task = json.loads(task_path.read_text(encoding="utf-8"))
        if task["target"]["type"] == target_type and len(task["entities"]) > 1:
            return unit
    raise AssertionError(f"no public unit of type {target_type} with more than one entity")


CLASSIFICATION_UNIT = _first_unit_of_type("classification")
REGRESSION_UNIT = _first_unit_of_type("regression")
RANKING_UNIT = _first_unit_of_type("ranking")

_ENTITY_ID_RE = re.compile(r"^  entity_id: (.+)$", re.MULTILINE)

#: Replies a model can send that the agent cannot use.
BAD_REPLIES = {
    "no_json": "I cannot answer that.",
    "broken_json": '{"label": "x", "point_forecast": }',
    "interval_not_an_object": json.dumps(
        {"point_forecast": 1.0, "interval": "wide", "evidence": []}
    ),
    "evidence_not_a_list": json.dumps({"point_forecast": 1.0, "evidence": 5}),
}


def _first_entity(unit: Path) -> str:
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    return str(task["entities"][0]["entity_id"])


def _run(unit: Path, out: Path, monkeypatch: pytest.MonkeyPatch, bad: dict[str, str]) -> dict:
    """Run the CLI end to end; entities named in `bad` get that reply, the rest the mock's."""
    mock_reply = cli._mock_reply

    def reply(system: str, user: str) -> str:
        match = _ENTITY_ID_RE.search(user)
        entity_id = match.group(1) if match else ""
        if entity_id in bad:
            return bad[entity_id]
        return mock_reply(system, user)

    monkeypatch.setattr(cli, "_mock_reply", reply)
    exit_code = cli.main(
        ["--task", str(unit / "task.json"), "--corpus", str(unit / "corpus"),
         "--out", str(out), "--mock"]
    )
    assert exit_code == 0
    return json.loads(out.read_text(encoding="utf-8"))


def _is_number(value: object) -> bool:
    """A finite JSON number: the scorer refuses NaN and infinities (`t4.nonfinite_value`)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _assert_schema_valid(answer: dict, unit: Path) -> None:
    """The published schema's row rules, plus: every span resolves in the corpus."""
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    corpus = build_index(unit / "corpus")
    assert isinstance(answer["task_id"], str)
    rows = answer["entity_predictions"]
    assert [r["entity_id"] for r in rows] == [e["entity_id"] for e in task["entities"]]
    for row in rows:
        where = row["entity_id"]
        if "label" in row:
            assert isinstance(row["label"], str), f"{where}: label must be a string, not null"
        if "point_forecast" in row:
            assert _is_number(row["point_forecast"]), f"{where}: point_forecast must be a number"
        if "rank" in row:
            assert isinstance(row["rank"], int) and not isinstance(row["rank"], bool), where
            assert "point_forecast" in row, where
        interval = row["interval"]
        assert interval["level"] == 0.9, where
        assert _is_number(interval["lo"]) and _is_number(interval["hi"]), where
        assert row["claims"], f"{where}: claims must not be empty"
        for claim in row["claims"]:
            assert isinstance(claim["claim"], str) and claim["claim"], where
            start, end = claim["span_start"], claim["span_end"]
            assert isinstance(start, int) and isinstance(end, int), where
            doc_text = corpus.doc_texts[claim["doc_id"]]
            assert 0 <= start < end <= len(doc_text), where
        if task["target"]["type"] in ("regression", "ranking"):
            assert "point_forecast" in row, f"{where}: the scorer needs a point_forecast"
        if task["target"].get("labels"):
            assert row.get("label") in task["target"]["labels"], where


@pytest.mark.parametrize("kind", sorted(BAD_REPLIES))
@pytest.mark.parametrize("unit", [CLASSIFICATION_UNIT, REGRESSION_UNIT, RANKING_UNIT],
                         ids=lambda u: u.name)
def test_one_bad_reply_costs_only_that_entity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unit: Path, kind: str
) -> None:
    good = _run(unit, tmp_path / "good.json", monkeypatch, bad={})
    victim = _first_entity(unit)
    answer = _run(unit, tmp_path / "bad.json", monkeypatch, bad={victim: BAD_REPLIES[kind]})

    good_rows = {r["entity_id"]: r for r in good["entity_predictions"]}
    rows = {r["entity_id"]: r for r in answer["entity_predictions"]}
    assert rows.keys() == good_rows.keys()
    for entity_id, row in rows.items():
        if entity_id != victim:
            assert row == good_rows[entity_id], f"{entity_id}: row changed"
    assert answer["notes"]["fallback_entities"] == [victim]
    assert good["notes"]["fallback_entities"] == []
    # A whole-row fallback is listed once, as a fallback row; good rows keep their interval.
    assert answer["notes"].get("interval_fallback_entities") == []
    assert good["notes"].get("interval_fallback_entities") == []
    _assert_schema_valid(answer, unit)


def test_fallback_row_is_a_placeholder_not_a_forecast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0.0 and the first allowed label, as `--mock` does; the claim quotes the corpus verbatim."""
    victim = _first_entity(CLASSIFICATION_UNIT)
    answer = _run(CLASSIFICATION_UNIT, tmp_path / "a.json", monkeypatch,
                  bad={victim: BAD_REPLIES["no_json"]})
    task = json.loads((CLASSIFICATION_UNIT / "task.json").read_text(encoding="utf-8"))
    [row] = [r for r in answer["entity_predictions"] if r["entity_id"] == victim]
    assert row["point_forecast"] == 0.0
    assert row["label"] == task["target"]["labels"][0]
    corpus = build_index(CLASSIFICATION_UNIT / "corpus")
    [claim] = row["claims"]
    cited = corpus.doc_texts[claim["doc_id"]][claim["span_start"]:claim["span_end"]]
    assert cited == claim["claim"], "the fallback claim must be the cited corpus text itself"


def test_regression_reply_without_a_number_gets_the_fallback_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regression row with no point_forecast is refused by the scorer, so it is unusable."""
    victim = _first_entity(REGRESSION_UNIT)
    no_number = json.dumps({"label": None, "point_forecast": None, "evidence": []})
    answer = _run(REGRESSION_UNIT, tmp_path / "a.json", monkeypatch, bad={victim: no_number})
    _assert_schema_valid(answer, REGRESSION_UNIT)
    assert answer["notes"]["fallback_entities"] == [victim]


@pytest.mark.parametrize("rank", [True, 1, 7], ids=["bool", "one", "seven"])
def test_a_rank_in_the_reply_is_never_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rank: object
) -> None:
    """`rank` is not scored, and a partial or non-permutation rank gets the whole unit refused,
    so a rank in the reply is ignored: the row is still the model's, without `rank`."""
    victim = _first_entity(RANKING_UNIT)
    with_rank = json.dumps(
        {"point_forecast": 1.0, "rank": rank,
         "interval": {"level": 0.9, "lo": 0.0, "hi": 2.0}, "evidence": []}
    )
    answer = _run(RANKING_UNIT, tmp_path / "a.json", monkeypatch, bad={victim: with_rank})
    _assert_schema_valid(answer, RANKING_UNIT)
    assert answer["notes"]["fallback_entities"] == []
    [row] = [r for r in answer["entity_predictions"] if r["entity_id"] == victim]
    assert row["point_forecast"] == 1.0
    assert "rank" not in row


def test_a_ranking_unit_with_mixed_replies_writes_no_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reviewer's case: some replies carry a rank, one cannot be used (fallback row), one
    carries a boolean rank. No row carries `rank`, so the ranks cannot be partial."""
    task = json.loads((RANKING_UNIT / "task.json").read_text(encoding="utf-8"))
    ids = [str(e["entity_id"]) for e in task["entities"]]
    assert len(ids) >= 3

    def reply(rank: object, point: float) -> str:
        return json.dumps({"point_forecast": point, "rank": rank,
                           "interval": {"level": 0.9, "lo": point - 1, "hi": point + 1},
                           "evidence": []})

    bad = {ids[0]: reply(1, 3.0), ids[1]: BAD_REPLIES["no_json"], ids[2]: reply(True, 2.0)}
    answer = _run(RANKING_UNIT, tmp_path / "a.json", monkeypatch, bad=bad)
    _assert_schema_valid(answer, RANKING_UNIT)
    assert answer["notes"]["fallback_entities"] == [ids[1]]
    assert all("rank" not in row for row in answer["entity_predictions"])


def test_the_ranking_prompt_does_not_ask_for_a_rank() -> None:
    """The prompt asks only for `point_forecast` on a ranking unit."""
    from baselines.strong_rag_baseline.prompts import build_user_prompt

    task = json.loads((RANKING_UNIT / "task.json").read_text(encoding="utf-8"))
    prompt = build_user_prompt(task, task["entities"][0], [])
    assert '"rank"' not in prompt
    assert '"point_forecast"' in prompt


@pytest.mark.parametrize("point", [None, "n/a"], ids=["null", "string"])
def test_a_classification_reply_without_a_number_leaves_point_forecast_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, point: object
) -> None:
    """`point_forecast` is optional on a classification unit, but a null one fails the schema
    for the whole submission: the key is left out and the model's label and interval are kept."""
    task = json.loads((CLASSIFICATION_UNIT / "task.json").read_text(encoding="utf-8"))
    label = task["target"]["labels"][-1]
    victim = _first_entity(CLASSIFICATION_UNIT)
    reply = json.dumps(
        {"label": label, "point_forecast": point,
         "interval": {"level": 0.9, "lo": 0.0, "hi": 2.0}, "evidence": []}
    )
    answer = _run(CLASSIFICATION_UNIT, tmp_path / "a.json", monkeypatch, bad={victim: reply})
    _assert_schema_valid(answer, CLASSIFICATION_UNIT)
    assert answer["notes"]["fallback_entities"] == []
    [row] = [r for r in answer["entity_predictions"] if r["entity_id"] == victim]
    assert "point_forecast" not in row
    assert row["label"] == label
    assert row["interval"] == {"level": 0.9, "lo": 0.0, "hi": 2.0}


@pytest.mark.parametrize("unit", [CLASSIFICATION_UNIT, REGRESSION_UNIT, RANKING_UNIT],
                         ids=lambda u: u.name)
def test_good_replies_write_no_null_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unit: Path
) -> None:
    """The prompt allows `"label": null` on regression and ranking units; it is never written."""
    answer = _run(unit, tmp_path / "a.json", monkeypatch, bad={})
    _assert_schema_valid(answer, unit)


# --------------------------------------------------------------------------- #
# Rows with no grounded claim                                                 #
# --------------------------------------------------------------------------- #
def _reply_with_evidence(evidence: list) -> str:
    return json.dumps(
        {"label": None, "point_forecast": 1.0,
         "interval": {"level": 0.9, "lo": 0.0, "hi": 2.0}, "evidence": evidence}
    )


@pytest.mark.parametrize(
    "evidence",
    [[], [{"doc_id": "NOT_A_DOC", "quote": "anything at all", "claim": "Cited from nowhere."}]],
    ids=["empty", "ungroundable"],
)
def test_a_row_whose_evidence_does_not_ground_still_cites_the_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evidence: list
) -> None:
    """An empty `claims` array fails `g1_schema` for the whole submission, so it is never written."""
    victim = _first_entity(REGRESSION_UNIT)
    answer = _run(REGRESSION_UNIT, tmp_path / "a.json", monkeypatch,
                  bad={victim: _reply_with_evidence(evidence)})
    _assert_schema_valid(answer, REGRESSION_UNIT)
    [row] = [r for r in answer["entity_predictions"] if r["entity_id"] == victim]
    assert row["point_forecast"] == 1.0, "the model's own prediction is kept"
    corpus = build_index(REGRESSION_UNIT / "corpus")
    [claim] = row["claims"]
    cited = corpus.doc_texts[claim["doc_id"]][claim["span_start"]:claim["span_end"]]
    assert cited == claim["claim"]
    assert answer["notes"]["fallback_claim_entities"] == [victim]
    assert answer["notes"]["fallback_entities"] == []


def test_with_nothing_retrieved_the_newest_eligible_document_is_quoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The document `baseline_agent` cites in the same case: newest doc_date on or before cutoff."""
    from baselines.strong_rag_baseline.retriever import BM25Index

    monkeypatch.setattr(BM25Index, "search", lambda self, query, top_k: [])
    answer = _run(REGRESSION_UNIT, tmp_path / "a.json", monkeypatch, bad={})
    _assert_schema_valid(answer, REGRESSION_UNIT)
    task = json.loads((REGRESSION_UNIT / "task.json").read_text(encoding="utf-8"))
    corpus = build_index(REGRESSION_UNIT / "corpus")
    eligible = {d: t for d, t in corpus.doc_dates.items() if t and t <= task["cutoff_date"]}
    newest = max(eligible.values())
    for row in answer["entity_predictions"]:
        [claim] = row["claims"]
        assert corpus.doc_dates[claim["doc_id"]] == newest
        assert claim["span_start"] == 0
        cited = corpus.doc_texts[claim["doc_id"]][: claim["span_end"]]
        assert cited == claim["claim"]


def test_with_no_eligible_document_nothing_is_cited_and_the_run_still_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing dated on or before the cutoff: no honest citation exists, so `claims` stays empty.

    Pins behaviour rather than a fix: `baseline_agent` cites a doc_id the corpus does not have
    here, which the scorer refuses too. This agent must not crash or invent a citation.
    """
    unit = tmp_path / "unit"
    shutil.copytree(REGRESSION_UNIT, unit)
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    task["cutoff_date"] = "1900-01-01"
    (unit / "task.json").write_text(json.dumps(task), encoding="utf-8")
    answer = _run(unit, tmp_path / "a.json", monkeypatch, bad={})
    assert all(row["claims"] == [] for row in answer["entity_predictions"])


# --------------------------------------------------------------------------- #
# Non-finite and oversized numbers, over-deep JSON                            #
# --------------------------------------------------------------------------- #

UNUSABLE_NUMBERS = {
    "point_nan": '{"point_forecast": NaN, "evidence": []}',
    "point_infinity": '{"point_forecast": Infinity, "evidence": []}',
    "point_band_overflows": '{"point_forecast": 1.7e308, "evidence": []}',
    # With a finite interval the band is never computed, so only the point check can catch these.
    "point_nan_finite_interval": (
        '{"point_forecast": NaN, "interval": {"level": 0.9, "lo": 0.0, "hi": 2.0}}'
    ),
    "point_infinity_finite_interval": (
        '{"point_forecast": Infinity, "interval": {"level": 0.9, "lo": 0.0, "hi": 2.0}}'
    ),
    "point_400_digits": '{"point_forecast": ' + "9" * 400 + ', "evidence": []}',
    "nested_100k_deep": '{"evidence": ' + "[" * 100_000 + "]" * 100_000 + "}",
}


@pytest.mark.parametrize("kind", sorted(UNUSABLE_NUMBERS))
def test_non_finite_oversized_or_too_deep_reply_gets_the_fallback_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    victim = _first_entity(REGRESSION_UNIT)
    answer = _run(REGRESSION_UNIT, tmp_path / "a.json", monkeypatch,
                  bad={victim: UNUSABLE_NUMBERS[kind]})
    assert answer["notes"]["fallback_entities"] == [victim], "reply not treated as unusable"
    _assert_schema_valid(answer, REGRESSION_UNIT)


UNUSABLE_BOUNDS = {
    "lo_nan": "NaN",
    "lo_minus_infinity": "-Infinity",
    "lo_400_digits": "-" + "9" * 400,
}


@pytest.mark.parametrize("kind", sorted(UNUSABLE_BOUNDS))
def test_an_unusable_interval_bound_keeps_the_models_forecast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """A bad bound is a missing interval (the fallback band), not a reason to drop the forecast."""
    victim = _first_entity(REGRESSION_UNIT)
    reply = (
        '{"point_forecast": 1.0, "interval": {"level": 0.9, "lo": '
        + UNUSABLE_BOUNDS[kind]
        + ', "hi": 2.0}, "evidence": []}'
    )
    answer = _run(REGRESSION_UNIT, tmp_path / "a.json", monkeypatch, bad={victim: reply})
    assert answer["notes"]["fallback_entities"] == [], "the model's forecast was dropped"
    _assert_schema_valid(answer, REGRESSION_UNIT)
    [row] = [r for r in answer["entity_predictions"] if r["entity_id"] == victim]
    assert row["point_forecast"] == 1.0
    assert row["interval"]["lo"] < 1.0 < row["interval"]["hi"]


INTERVAL_PROBLEMS = {
    "missing": "",
    "lo_nan": ', "interval": {"level": 0.9, "lo": NaN, "hi": 2.0}',
    "hi_infinity": ', "interval": {"level": 0.9, "lo": 0.5, "hi": Infinity}',
    "lo_above_hi": ', "interval": {"level": 0.9, "lo": 3.0, "hi": 2.0}',
}


@pytest.mark.parametrize("kind", sorted(INTERVAL_PROBLEMS))
def test_a_substituted_band_is_logged_and_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    """The band around the kept forecast is not the model's interval, so the answer says so."""
    victim = _first_entity(REGRESSION_UNIT)
    reply = '{"point_forecast": 1.0' + INTERVAL_PROBLEMS[kind] + ', "evidence": []}'
    answer = _run(REGRESSION_UNIT, tmp_path / "a.json", monkeypatch, bad={victim: reply})
    assert answer["notes"].get("interval_fallback_entities") == [victim], (
        "the substituted band is not recorded in notes"
    )
    stderr = capsys.readouterr().err
    assert f"{victim!r}" in stderr and "band" in stderr and "substituted" in stderr, (
        "the substituted band is not logged to stderr"
    )
    assert answer["notes"]["fallback_entities"] == []
    [row] = [r for r in answer["entity_predictions"] if r["entity_id"] == victim]
    assert row["point_forecast"] == 1.0
    assert (row["interval"]["lo"], row["interval"]["hi"]) == (0.0, 2.0)  # 1.0 +/- max(0.5, 1)


# --------------------------------------------------------------------------- #
# Transport failures, through the real HTTP client against a local server     #
# --------------------------------------------------------------------------- #
class _House(BaseHTTPRequestHandler):
    """A local chat-completions server: the victim entity fails, every other one gets the mock."""

    victim = ""  # "*" fails every entity
    failure = ""  # "budget_<n>": the first n requests are answered, every later one denied
    denial_body = ""
    requests: list[str] = []

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system, user = (m["content"] for m in body["messages"])
        match = _ENTITY_ID_RE.search(user)
        self.requests.append(match.group(1) if match else "")
        if self.failure.startswith("budget_") and len(self.requests) > int(
            self.failure.split("_")[1]
        ):
            # Past the request cap the House answers 403 with `grant_denied` in the body.
            denial = self.denial_body.encode()
            self.send_response(403)
            self.send_header("Content-Length", str(len(denial)))
            self.end_headers()
            self.wfile.write(denial)
            return
        if self.failure.startswith("http_") and self.denial_body:
            denial = self.denial_body.encode()
            self.send_response(int(self.failure[len("http_"):]))
            self.send_header("Content-Length", str(len(denial)))
            self.end_headers()
            self.wfile.write(denial)
            return
        if match and self.victim in (match.group(1), "*"):
            if self.failure == "hang":
                # Longer than the client's timeout. Not time.sleep: the test stubs that out
                # (the client's retry back-off calls it through the shared `time` module).
                threading.Event().wait(2.0)
                return
            elif self.failure.startswith("http_"):
                self.send_response(int(self.failure[len("http_"):]))
                self.end_headers()
                return
            elif self.failure == "drop":
                self.close_connection = True
                return  # no status line at all: http.client.RemoteDisconnected
        content = cli._mock_reply(system, user)
        payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:
        pass


def _serve_and_run(
    out: Path, monkeypatch: pytest.MonkeyPatch, victim: str, failure: str,
    endpoint: str | None = None, seen: list[str] | None = None, denial_body: str = "",
) -> int:
    """Run the CLI against a local server; `seen` collects the entity id of every request."""
    from baselines.strong_rag_baseline import client as client_module

    unit = REGRESSION_UNIT
    handler = type(
        "House", (_House,),
        {"victim": victim, "failure": failure, "requests": seen if seen is not None else [],
         "denial_body": denial_body},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(client_module.time, "sleep", lambda s: None)
        monkeypatch.setenv(
            "MODEL_ENDPOINT", endpoint or f"http://127.0.0.1:{server.server_address[1]}"
        )
        monkeypatch.setenv("T4_MODEL_TIMEOUT_S", "0.3")
        monkeypatch.setenv("T4_MODEL_RETRIES", "2")
        return cli.main(
            ["--task", str(unit / "task.json"), "--corpus", str(unit / "corpus"),
             "--out", str(out)]
        )
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("failure", ["hang", "http_500", "drop"])
def test_a_failed_model_call_costs_only_that_entity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    victim = _first_entity(REGRESSION_UNIT)
    out = tmp_path / "a.json"
    _serve_and_run(out, monkeypatch, victim=victim, failure=failure)
    answer = json.loads(out.read_text(encoding="utf-8"))
    _assert_schema_valid(answer, REGRESSION_UNIT)
    assert answer["notes"]["fallback_entities"] == [victim]


def test_a_missing_endpoint_still_ends_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configuration error is not a per-entity failure: all-placeholder output would hide it."""
    monkeypatch.delenv("MODEL_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_ENDPOINT is not set"):
        cli.main(
            ["--task", str(REGRESSION_UNIT / "task.json"),
             "--corpus", str(REGRESSION_UNIT / "corpus"), "--out", str(tmp_path / "a.json")]
        )


def test_when_every_model_call_fails_the_unit_finishes_with_noted_fallback_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every call failing after retries (not a configuration error) still writes fallback rows
    and exits 0: a non-zero exit is charged to the team. The notes say so, so the unit can be
    found later."""
    task = json.loads((REGRESSION_UNIT / "task.json").read_text(encoding="utf-8"))
    ids = [str(e["entity_id"]) for e in task["entities"]]
    out = tmp_path / "a.json"
    assert _serve_and_run(out, monkeypatch, victim="*", failure="http_500") == 0
    answer = json.loads(out.read_text(encoding="utf-8"))
    _assert_schema_valid(answer, REGRESSION_UNIT)
    notes = answer["notes"]
    assert notes["every_model_call_failed"] is True
    assert notes["call_failed_entities"] == ids
    assert notes["fallback_entities"] == ids
    assert notes["budget_denied_entities"] == []
    assert "every model call failed" in capsys.readouterr().err


def test_one_failed_call_is_noted_but_not_every_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: one failed call is listed in call_failed_entities; every_model_call_failed stays
    false."""
    victim = _first_entity(REGRESSION_UNIT)
    out = tmp_path / "a.json"
    assert _serve_and_run(out, monkeypatch, victim=victim, failure="http_500") == 0
    notes = json.loads(out.read_text(encoding="utf-8"))["notes"]
    assert notes["call_failed_entities"] == [victim]
    assert notes["every_model_call_failed"] is False


@pytest.mark.parametrize("unit", [CLASSIFICATION_UNIT, REGRESSION_UNIT, RANKING_UNIT],
                         ids=lambda u: u.name)
def test_when_every_reply_is_unusable_the_run_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unit: Path
) -> None:
    """A wrong model or no JSON on every entity must not pass as a placeholder-only answer."""
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(cli, "_mock_reply", lambda system, user: BAD_REPLIES["no_json"])
    out = tmp_path / "a.json"
    with pytest.raises(RuntimeError, match="no model reply was usable") as raised:
        cli.main(
            ["--task", str(unit / "task.json"), "--corpus", str(unit / "corpus"),
             "--out", str(out), "--mock"]
        )
    assert f"({len(task['entities'])} entities)" in str(raised.value)
    assert not out.exists()


def test_one_usable_reply_is_enough_to_write_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control for the all-unusable case: every entity but one unusable still writes an answer."""
    task = json.loads((REGRESSION_UNIT / "task.json").read_text(encoding="utf-8"))
    ids = [str(e["entity_id"]) for e in task["entities"]]
    bad = {entity_id: BAD_REPLIES["no_json"] for entity_id in ids[1:]}
    answer = _run(REGRESSION_UNIT, tmp_path / "a.json", monkeypatch, bad=bad)
    _assert_schema_valid(answer, REGRESSION_UNIT)
    assert answer["notes"]["fallback_entities"] == ids[1:]


@pytest.mark.parametrize("status", [401, 403, 404])
def test_a_rejected_endpoint_or_token_ends_the_run_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """Wrong endpoint or credential: one request, no retries, no placeholder rows, no answer."""
    out = tmp_path / "a.json"
    seen: list[str] = []
    with pytest.raises(RuntimeError, match="check MODEL_ENDPOINT and MODEL_TOKEN"):
        _serve_and_run(out, monkeypatch, victim="*", failure=f"http_{status}", seen=seen)
    assert len(seen) == 1, f"the server was asked {len(seen)} times"
    assert not out.exists()


def test_an_endpoint_with_no_scheme_ends_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "a.json"
    with pytest.raises(RuntimeError, match="check MODEL_ENDPOINT and MODEL_TOKEN"):
        _serve_and_run(out, monkeypatch, victim="", failure="", endpoint="127.0.0.1:1")
    assert not out.exists()


#: The House's refusal past the request cap (and for an expired or unknown grant), as the
#: organizers gave it: `error.code` "grant_denied" with `error.message` "request admission
#: refused". Key order and whitespace do not matter.
GRANT_DENIED_BODIES = {
    "house": json.dumps({"error": {"message": "request admission refused",
                                   "type": "invalid_request_error", "code": "grant_denied"}}),
    "reordered": json.dumps({"error": {"code": "grant_denied", "type": "invalid_request_error",
                                       "message": "request admission refused"}}, indent=2),
}

#: 403 bodies that are not that refusal: each is a configuration error and ends the run.
NOT_BUDGET_403_BODIES = {
    # A wrong model name: same code, another message.
    "model_mismatch": json.dumps({"error": {"message": "request model does not match credential",
                                            "type": "invalid_request_error",
                                            "code": "grant_denied"}}),
    "other_message": json.dumps({"error": {"code": "grant_denied",
                                           "message": "request cap reached"}}),
    "other_code": json.dumps({"error": {"code": "forbidden",
                                        "message": "request admission refused"}}),
    "no_message": json.dumps({"error": {"code": "grant_denied"}}),
    "error_not_object": json.dumps({"error": "grant_denied"}),
    "top_level_list": json.dumps([{"error": {"code": "grant_denied",
                                             "message": "request admission refused"}}]),
    "plain_text": "grant_denied: request admission refused",
    "truncated_json": '{"error": {"code": "grant_denied", "message": "request admission refused"',
}


@pytest.mark.parametrize("answered", [1, 2])
@pytest.mark.parametrize("body", sorted(GRANT_DENIED_BODIES))
def test_a_used_up_request_allowance_writes_fallback_rows_and_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    answered: int, body: str,
) -> None:
    """Past the cap, after at least one answered request, the House answers 403 grant_denied:
    the unit still finishes with an answer, the entities the model never answered get fallback
    rows, and no further request is sent."""
    task = json.loads((REGRESSION_UNIT / "task.json").read_text(encoding="utf-8"))
    ids = [str(e["entity_id"]) for e in task["entities"]]
    assert len(ids) > answered + 1
    out = tmp_path / "a.json"
    seen: list[str] = []
    exit_code = _serve_and_run(out, monkeypatch, victim="", failure=f"budget_{answered}",
                               seen=seen, denial_body=GRANT_DENIED_BODIES[body])
    assert exit_code == 0
    answer = json.loads(out.read_text(encoding="utf-8"))
    _assert_schema_valid(answer, REGRESSION_UNIT)
    denied = ids[answered:]
    assert answer["notes"]["fallback_entities"] == denied
    assert answer["notes"]["budget_denied_entities"] == denied
    assert len(seen) == answered + 1, "requests were sent after the allowance was used up"
    stderr = capsys.readouterr().err
    assert "grant_denied" in stderr and "allowance was used up" in stderr
    assert answer["notes"]["budget_refused_before_any_reply"] is False
    assert answer["notes"]["budget_refused_before_any_usable_reply"] is False


@pytest.mark.parametrize("body", sorted(GRANT_DENIED_BODIES))
def test_a_grant_denied_before_any_reply_writes_noted_fallback_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    body: str,
) -> None:
    """A refusal on the unit's first request still finishes the unit: fallback rows for every
    entity, exit 0 (a non-zero exit is charged to the team), no further request, and notes that
    record the refusal came before any reply."""
    task = json.loads((REGRESSION_UNIT / "task.json").read_text(encoding="utf-8"))
    ids = [str(e["entity_id"]) for e in task["entities"]]
    out = tmp_path / "a.json"
    seen: list[str] = []
    exit_code = _serve_and_run(out, monkeypatch, victim="", failure="budget_0", seen=seen,
                               denial_body=GRANT_DENIED_BODIES[body])
    assert exit_code == 0
    assert len(seen) == 1, "requests were sent after the refusal"
    answer = json.loads(out.read_text(encoding="utf-8"))
    _assert_schema_valid(answer, REGRESSION_UNIT)
    notes = answer["notes"]
    assert notes["budget_denied_entities"] == ids
    assert notes["fallback_entities"] == ids
    assert notes["budget_refused_before_any_reply"] is True
    assert notes["budget_refused_before_any_usable_reply"] is True
    stderr = capsys.readouterr().err
    assert "grant_denied" in stderr and "before any model reply was received" in stderr


def test_a_grant_denied_after_only_failed_calls_writes_noted_fallback_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed call is not a reply: calls that failed and then a refusal still mean no request
    in the unit was answered. The unit finishes with fallback rows and the notes say so."""
    calls: list[int] = []

    def reply(system: str, user: str) -> str:
        calls.append(1)
        if len(calls) == 1:
            raise ModelCallError("model call failed after 2 attempts")
        raise ModelBudgetExhausted("the unit's model request allowance is used up")

    out = tmp_path / "a.json"
    answer = cli.run(REGRESSION_UNIT / "task.json", REGRESSION_UNIT / "corpus", out,
                     MockModelClient(reply=reply), 5)
    _assert_schema_valid(answer, REGRESSION_UNIT)
    task = json.loads((REGRESSION_UNIT / "task.json").read_text(encoding="utf-8"))
    ids = [str(e["entity_id"]) for e in task["entities"]]
    notes = answer["notes"]
    assert notes["call_failed_entities"] == ids[:1]
    assert notes["budget_denied_entities"] == ids[1:]
    assert notes["budget_refused_before_any_reply"] is True
    assert notes["every_model_call_failed"] is False
    assert json.loads(out.read_text(encoding="utf-8")) == answer


def test_a_grant_denied_after_an_unusable_reply_writes_noted_fallback_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unusable reply then a refusal follows the refusal rule: the unit finishes with
    fallback rows, and the notes record that no usable reply came before the refusal."""
    calls: list[int] = []

    def reply(system: str, user: str) -> str:
        calls.append(1)
        if len(calls) == 1:
            return BAD_REPLIES["no_json"]
        raise ModelBudgetExhausted("the unit's model request allowance is used up")

    out = tmp_path / "a.json"
    answer = cli.run(REGRESSION_UNIT / "task.json", REGRESSION_UNIT / "corpus", out,
                     MockModelClient(reply=reply), 5)
    _assert_schema_valid(answer, REGRESSION_UNIT)
    task = json.loads((REGRESSION_UNIT / "task.json").read_text(encoding="utf-8"))
    ids = [str(e["entity_id"]) for e in task["entities"]]
    assert answer["notes"]["budget_denied_entities"] == ids[1:]
    assert answer["notes"]["fallback_entities"] == ids
    assert answer["notes"]["budget_refused_before_any_reply"] is False
    assert answer["notes"]["budget_refused_before_any_usable_reply"] is True


@pytest.mark.parametrize("body", sorted(NOT_BUDGET_403_BODIES))
@pytest.mark.parametrize("answered", [0, 1])
def test_a_403_that_is_not_the_budget_refusal_ends_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answered: int, body: str
) -> None:
    """A wrong model name (code grant_denied, message "request model does not match
    credential"), any other code or message, or a body that cannot be parsed is a configuration
    error: the run ends at once with no answer and no placeholder rows, before or after a reply."""
    out = tmp_path / "a.json"
    seen: list[str] = []
    with pytest.raises(RuntimeError, match="check MODEL_ENDPOINT and MODEL_TOKEN"):
        _serve_and_run(out, monkeypatch, victim="", failure=f"budget_{answered}", seen=seen,
                       denial_body=NOT_BUDGET_403_BODIES[body])
    assert len(seen) == answered + 1, "requests were sent after the configuration error"
    assert not out.exists()


@pytest.mark.parametrize(
    "denial_body", ["", json.dumps({"error": {"code": "forbidden"}})], ids=["empty", "other"]
)
def test_a_403_without_grant_denied_still_ends_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, denial_body: str
) -> None:
    """Control: only the grant_denied refusal finishes the unit; any other 403 is a wrong
    endpoint or credential and ends the run at once, as before."""
    out = tmp_path / "a.json"
    seen: list[str] = []
    with pytest.raises(RuntimeError, match="check MODEL_ENDPOINT and MODEL_TOKEN"):
        _serve_and_run(out, monkeypatch, victim="*", failure="http_403", seen=seen,
                       denial_body=denial_body)
    assert len(seen) == 1
    assert not out.exists()


# --------------------------------------------------------------------------- #
# Nothing retrieved: the document baseline_agent would cite, ties included    #
# --------------------------------------------------------------------------- #
def test_with_nothing_retrieved_a_date_tie_breaks_like_baseline_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from baselines.baseline_agent.cli import _newest_eligible
    from baselines.baseline_agent.indexer import build_index as agent_build_index
    from baselines.strong_rag_baseline.retriever import BM25Index

    unit = tmp_path / "unit"
    shutil.copytree(REGRESSION_UNIT, unit)
    cutoff = json.loads((unit / "task.json").read_text(encoding="utf-8"))["cutoff_date"]
    dated = []
    for path in sorted((unit / "corpus").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if path.name != "manifest.json" and isinstance(doc.get("doc_date"), str) \
                and doc["doc_date"] <= cutoff:
            dated.append((doc["doc_date"], path))
    assert len(dated) >= 2
    newest_date = max(d for d, _ in dated)
    # Give every eligible document the newest date: the tie-break alone picks the document.
    for _, path in dated:
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["doc_date"] = newest_date
        path.write_text(json.dumps(doc), encoding="utf-8")

    expected = _newest_eligible(agent_build_index(unit / "corpus"), cutoff)
    assert expected is not None
    monkeypatch.setattr(BM25Index, "search", lambda self, query, top_k: [])
    answer = _run(unit, tmp_path / "a.json", monkeypatch, bad={})
    cited = {c["doc_id"] for row in answer["entity_predictions"] for c in row["claims"]}
    assert cited == {expected.doc_id}
