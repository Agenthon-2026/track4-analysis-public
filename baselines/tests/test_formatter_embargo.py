"""The formatter's advertised "second embargo filter" must actually exist.

baselines/README.md §5 and formatter.py's own docstring both promise that the output
formatter "drops any citation whose doc_date > cutoff_date before writing output". Before
the fix, build_entity_prediction copied claims through verbatim — no date, no cutoff, no
corpus reference anywhere in the module — so anything reaching the formatter outside the
retriever's happy path was written out unfiltered, and an embargo violation is
ineligibility, not a score penalty.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_BASELINES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BASELINES))

from baseline_agent import cli as baseline_cli  # noqa: E402
from baseline_agent.formatter import (  # noqa: E402
    build_entity_prediction,
    filter_embargoed,
)
from baseline_agent.retriever import Retrieved  # noqa: E402

CUTOFF = "2024-02-10"
DOC_DATES = {"OLD": "2024-02-01", "STALE": "2024-03-01", "UNDATED": None}


def _claim(doc_id: str) -> dict[str, object]:
    return {"doc_id": doc_id, "span_start": 0, "span_end": 10, "claim": f"cites {doc_id}"}


def test_filter_drops_post_cutoff_citation() -> None:
    kept = filter_embargoed([_claim("OLD"), _claim("STALE")], CUTOFF, DOC_DATES)
    assert [c["doc_id"] for c in kept] == ["OLD"]


def test_filter_keeps_undated_and_unknown_docs() -> None:
    """Only ``doc_date > cutoff_date`` is advertised as dropped. An undated or unknown doc
    is not provably stale here — the scorer's embargo gate owns that judgment."""
    kept = filter_embargoed([_claim("UNDATED"), _claim("NEW-ID")], CUTOFF, DOC_DATES)
    assert [c["doc_id"] for c in kept] == ["UNDATED", "NEW-ID"]


def test_build_entity_prediction_applies_the_advertised_filter() -> None:
    pred = build_entity_prediction(
        "AAPL",
        "beat",
        1.0,
        0.5,
        1.5,
        [_claim("OLD"), _claim("STALE")],
        cutoff=CUTOFF,
        doc_dates=DOC_DATES,
    )
    assert [c["doc_id"] for c in pred["claims"]] == ["OLD"]


def test_cli_falls_back_when_filter_empties_claims(tmp_path: Path, monkeypatch) -> None:
    """If the only retrieved citation is post-cutoff, the filter must not leave the entity
    with zero claims (the schema requires >=1): the no-hit fallback re-fires and cites the
    newest embargo-eligible document instead."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for doc_id, date in (("OK_DOC", "2024-02-01"), ("STALE_DOC", "2024-03-01")):
        (corpus / f"{doc_id}.json").write_text(
            json.dumps(
                {"doc_id": doc_id, "doc_date": date, "text": "earnings text " * 30}
            ),
            encoding="utf-8",
        )
    task_path = tmp_path / "task.json"
    task_path.write_text(
        json.dumps(
            {
                "task_id": "t4-test",
                "cutoff_date": CUTOFF,
                "target": {"type": "classification"},
                "entities": [{"entity_id": "AAPL", "name": "Apple"}],
            }
        ),
        encoding="utf-8",
    )

    def stale_hit(query, docs, cutoff_date, max_span_chars=240):
        return Retrieved(
            doc_id="STALE_DOC", span_start=0, span_end=40, text="stale span", score=9.9
        )

    monkeypatch.setattr(baseline_cli, "retrieve", stale_hit)
    answer = baseline_cli.run(task_path, corpus, tmp_path / "answer.json")

    claims = answer["entity_predictions"][0]["claims"]
    assert len(claims) >= 1, "filter must never leave an entity schema-invalid"
    assert all(c["doc_id"] != "STALE_DOC" for c in claims)
    assert claims[0]["doc_id"] == "OK_DOC"
