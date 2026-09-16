"""Independent regressions for signed quantities and per-entity attribution."""

from __future__ import annotations

import pytest

from baselines.strong_rag_baseline.agent import run_entity_grounded
from baselines.strong_rag_baseline.indexer import Chunk, IndexedCorpus
from baselines.strong_rag_baseline.retriever import BM25Index
from baselines.strong_rag_baseline.tests.test_quantities import predict


@pytest.mark.parametrize(
    "current,prior,growth",
    [
        ("-1.20", "1.00", -220.0),
        ("(1.20)", "1.00", -220.0),
        ("1.20", "-1.00", -220.0),
        ("1.20", "(1.00)", -220.0),
        ("-1.20", "-1.00", 20.0),
    ],
)
def test_generic_growth_preserves_operand_signs(current, prior, growth):
    result = predict(
        "revenue_growth_pct",
        f"Widget A revenue was {current} compared to {prior} in the prior year.",
    )
    assert result["point_forecast"] == pytest.approx(growth)


@pytest.mark.parametrize("current", ["-1.20", "(1.20)"])
def test_eps_growth_preserves_signed_level_with_task_reference(current):
    result = predict(
        "eps_yoy_growth_pct",
        f"Widget A diluted EPS was {current} compared to 2.00 in the prior year.",
        entity={"prior_year_q_eps": 1.0},
    )
    assert result["point_forecast"] == pytest.approx(-220.0)


def _predict_company_b(text, *, doc_id="report"):
    entities = [
        {"entity_id": "WGA", "name": "Widget A"},
        {"entity_id": "WGB", "name": "Widget B", "consensus_eps": 3.0},
    ]
    task = {
        "cutoff_date": "2024-06-01",
        "entities": entities,
        "target": {
            "name": "diluted_eps",
            "type": "classification",
            "labels": ["beat", "miss", "inline"],
        },
    }
    chunk = Chunk(doc_id, "2024-05-01", 0, len(text), text)
    corpus = IndexedCorpus([chunk], {doc_id: text}, {doc_id: chunk.doc_date})
    return run_entity_grounded(
        task, entities[1], BM25Index([chunk], task["cutoff_date"]), corpus, 10
    )


@pytest.mark.parametrize("separator", [". ", "; ", "\n"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("doc_id", ["report", "Widget_B_filing"])
def test_multiple_companies_do_not_share_the_first_eps(separator, reverse, doc_id):
    sentences = ["Widget A diluted EPS was 1.20", "Widget B diluted EPS was 3.40"]
    if reverse:
        sentences.reverse()
    text = separator.join(sentences) + "."
    result = _predict_company_b(text, doc_id=doc_id)
    assert result.prediction["point_forecast"] == pytest.approx(3.4)
    assert result.prediction["label"] == "beat"
    claim = result.prediction["claims"][0]
    assert "Widget B" in claim["claim"]
    assert "Widget A" not in claim["claim"]
    assert claim["claim"] == text[claim["span_start"] : claim["span_end"]].strip()


def test_ambiguous_multiple_company_sentence_uses_explicit_fallback():
    result = _predict_company_b(
        "Widget A and Widget B diluted EPS were 1.20 and 3.40, respectively."
    )
    assert result.prediction["point_forecast"] == 0
    assert "fallback" in result.rationale
