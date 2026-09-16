"""Synthetic controls for evidence attribution, embargo, and context retention."""

from __future__ import annotations

import json

import pytest

from baselines.strong_rag_baseline.evidence import evidence_queries, prepare_evidence
from baselines.strong_rag_baseline.indexer import Chunk, IndexedCorpus
from baselines.strong_rag_baseline.retriever import BM25Index


def _packet(documents, *, target=None, entity=None, **budgets):
    entities = [
        {"entity_id": "ALP", "name": "Alpha Company"},
        {"entity_id": "BET", "name": "Beta Company"},
    ]
    entity = entity or entities[1]
    task = {
        "entities": entities,
        "cutoff_date": "2024-06-01",
        "resolution_date": "2024-08-01",
        "target": target or {"name": "diluted_eps", "type": "regression"},
    }
    texts = {doc_id: text for doc_id, _, text in documents}
    dates = {doc_id: date for doc_id, date, _ in documents}
    chunks = [
        Chunk(doc_id, date, 0, len(text), text) for doc_id, date, text in documents
    ]
    corpus = IndexedCorpus(chunks, texts, dates)
    # Deliberately permissive index: the packet must enforce its own task cutoff.
    index = BM25Index(chunks, "2026-01-01")
    return prepare_evidence(task, entity, index, corpus, **budgets), corpus


@pytest.mark.parametrize("separator", [". ", "; ", "\n"])
def test_other_entity_values_do_not_enter_evidence(separator):
    text = separator.join(
        ["Alpha Company diluted EPS was 1.25", "Beta Company diluted EPS was 3.40"]
    )
    packet, _ = _packet([("report", "2024-05-01", text)])
    assert packet.chunks
    assert all("1.25" not in chunk.text for chunk in packet.chunks)
    assert any("3.40" in chunk.text for chunk in packet.chunks)


def test_ambiguous_multi_entity_statement_is_not_attributed():
    packet, _ = _packet(
        [("report", "2024-05-01", "Alpha Company and Beta Company EPS were 1 and 3.")]
    )
    assert packet.chunks == []
    assert packet.ledger["records"] == []


@pytest.mark.parametrize("kind", ["classification", "regression", "ranking"])
@pytest.mark.parametrize("metric", ["eps_yoy_growth_pct", "diluted_EPS"])
def test_eps_queries_retrieve_per_share_values_before_generic_growth(kind, metric):
    text = (
        "Diluted earnings per common share were $2.40 for Q1 2024.\n"
        "Economic growth forecasts reflect historical risks and future expectations."
    )
    packet, corpus = _packet(
        [("Beta_Company_filing", "2024-05-01", text)],
        target={"name": metric, "type": kind},
        top_k=1,
    )
    [chunk] = packet.chunks
    assert "$2.40" in chunk.text
    assert "Economic growth" not in chunk.text
    assert corpus.doc_texts[chunk.doc_id][chunk.span_start : chunk.span_end] == chunk.text


def test_non_eps_growth_target_keeps_its_own_quantity():
    packet, _ = _packet(
        [
            (
                "Beta_Company_filing",
                "2024-05-01",
                "Diluted earnings per common share were $2.40.\n"
                "Steps growth was 12 percent; guidance forecasts further growth.",
            )
        ],
        target={"name": "steps_growth", "type": "regression"},
        top_k=1,
    )
    [chunk] = packet.chunks
    assert "12 percent" in chunk.text
    assert "$2.40" not in chunk.text


def test_cutoff_and_strict_document_dates_are_enforced():
    packet, _ = _packet(
        [
            ("old", "2024-06-01", "Beta Company EPS was 2.50."),
            ("future", "2024-06-02", "Beta Company EPS was 99.50."),
            ("undated", None, "Beta Company EPS was 88.50."),
            ("malformed", "20240601", "Beta Company EPS was 77.50."),
        ]
    )
    assert {chunk.doc_id for chunk in packet.chunks} == {"old"}


def test_exact_offsets_and_financial_heading_survive_table_retrieval():
    text = (
        "Market commentary.\n"
        "Diluted EPS | Q1 2024 | Q1 2023\n"
        "Beta Company | $3.40 | $2.80\n"
        "Alpha Company | $9.50 | $9.10"
    )
    packet, corpus = _packet([("report", "2024-05-15", text)])
    record = packet.ledger["records"][0]
    assert "Q1 2024 | Q1 2023" in record["text"]
    assert "$9.50" not in record["text"]
    assert {period["text"] for period in record["periods"]} == {"Q1 2024", "Q1 2023"}
    assert [quantity["value"] for quantity in record["quantities"]] == [3.4, 2.8]
    for item in [record, *record["periods"], *record["metrics"], *record["quantities"]]:
        assert (
            corpus.doc_texts[record["doc_id"]][item["span_start"] : item["span_end"]]
            == item["text"]
        )
    assert all(quantity["period"] is None for quantity in record["quantities"])


def test_unknown_target_units_and_periods_are_not_invented():
    packet, _ = _packet(
        [("report", "2024-05-01", "Beta Company reported a reading of 17.5.")],
        target={"name": "unpublished_indicator", "type": "regression"},
    )
    assert packet.ledger["unit"] is None
    record = packet.ledger["records"][0]
    assert record["metrics"] == []
    assert record["periods"] == []
    assert record["doc_date"] == "2024-05-01"
    quantity = record["quantities"][0]
    assert quantity["value"] == 17.5
    assert quantity["metric"] is quantity["unit"] is quantity["period"] is None
    assert json.loads(json.dumps(packet.ledger)) == packet.ledger


def test_requested_unit_does_not_override_literal_quantity_units():
    packet, _ = _packet(
        [("report", "2024-05-01", "Beta Company revenue was USD 2.5 billion, up 5%.")],
        target={"name": "revenue_growth", "type": "regression", "unit": "percent"},
    )
    assert packet.ledger["unit"] == "percent"
    quantities = packet.ledger["records"][0]["quantities"]
    assert [(q["value"], q["unit"], q["scale"]) for q in quantities] == [
        (2.5, "USD", "billion"),
        (5.0, "%", None),
    ]
    assert all(q["metric"] is None for q in quantities)


def test_dates_and_explicit_periods_are_not_numeric_observations():
    packet, _ = _packet(
        [
            (
                "report",
                "2024-05-01",
                "Beta Company FY2023 EPS was $2.50 as of 2024-04-01.",
            )
        ]
    )
    record = packet.ledger["records"][0]
    assert [q["value"] for q in record["quantities"]] == [2.5]
    assert [p["text"] for p in record["periods"]] == ["FY2023"]


@pytest.mark.parametrize("token", ["1.2e5", "1.2e-5", "1.2e+5", "-2E-10"])
def test_unsupported_numeric_forms_are_not_partially_parsed(token):
    packet, _ = _packet(
        [("report", "2024-05-01", f"Beta Company EPS estimate token was {token}.")]
    )
    assert packet.ledger["records"][0]["quantities"] == []


@pytest.mark.parametrize(
    ("token", "expected", "currency"),
    [
        ("-$2.50", -2.5, "$"),
        ("-USD 2.50", -2.5, "USD"),
        ("+$2.50", 2.5, "$"),
        ("USD -2.50", -2.5, "USD"),
    ],
)
def test_signed_currency_preserves_value_and_exact_offsets(token, expected, currency):
    packet, corpus = _packet(
        [("report", "2024-05-01", f"Beta Company EPS was {token}.")]
    )
    [record] = packet.ledger["records"]
    [quantity] = record["quantities"]
    assert quantity["text"] == token
    assert quantity["value"] == expected
    assert quantity["currency"] == quantity["unit"] == currency
    assert (
        corpus.doc_texts[record["doc_id"]][
            quantity["span_start"] : quantity["span_end"]
        ]
        == token
    )


def test_unrepresentable_quantity_remains_json_safe():
    packet, _ = _packet(
        [("report", "2024-05-01", "Beta Company EPS was " + "9" * 400 + ".")]
    )
    assert packet.ledger["records"][0]["quantities"][0]["value"] is None
    json.dumps(packet.ledger, allow_nan=False)


def test_document_binding_provenance_does_not_claim_text_mention():
    packet, _ = _packet([("Beta_Company_filing", "2024-05-01", "EPS was 3.50.")])
    assert packet.ledger["records"][0]["binding"] == {
        "source": "document_alias",
        "aliases": [],
    }


def test_budgets_deduplication_and_order_are_deterministic():
    documents = [
        (f"report-{i}", "2024-05-01", f"Beta Company EPS guidance was {i}. " * 20)
        for i in range(10)
    ]
    packet, corpus = _packet(documents, top_k=3, max_chars=200, max_span_chars=100)
    again, _ = _packet(
        list(reversed(documents)), top_k=3, max_chars=200, max_span_chars=100
    )
    assert packet == again
    assert 0 < len(packet.chunks) <= 3
    assert sum(len(chunk.text) for chunk in packet.chunks) <= 200
    assert all(len(chunk.text) <= 100 for chunk in packet.chunks)
    keys = [(c.doc_id, c.span_start, c.span_end) for c in packet.chunks]
    assert len(keys) == len(set(keys))
    for chunk in packet.chunks:
        assert (
            corpus.doc_texts[chunk.doc_id][chunk.span_start : chunk.span_end]
            == chunk.text
        )


def test_queries_follow_requested_target_without_unrelated_revenue_suffix():
    queries = evidence_queries(
        {"target": {"name": "yield_change_bps"}},
        {"entity_id": "BOND10", "name": "10-year Treasury"},
    )
    assert len(set(queries)) == 4
    assert all("yield change bps" in query for query in queries)
    assert all("revenue" not in query and "earnings" not in query for query in queries)


@pytest.mark.parametrize(
    "budgets", [{"top_k": 0}, {"max_chars": 0}, {"max_span_chars": 0}]
)
def test_invalid_budgets_fail_explicitly(budgets):
    with pytest.raises(ValueError, match="budgets"):
        _packet([], **budgets)
