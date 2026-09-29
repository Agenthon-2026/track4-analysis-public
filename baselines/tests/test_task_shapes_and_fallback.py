"""The two published `task.json` shapes, and the no-retrieval-hit citation fallback.

Both defects covered here were mutation-reverted against the full suite without turning a single
test red, which is the reason this file exists.

* **`target_type`.** `SUBMISSION_CLI.md` (the T4 interface row, and contract invariant 7) publishes
  `target_type` as a **top-level** field of `/input/task.json`; every `task.json` in this repo
  declares it nested at `target.type`. A baseline that reads one shape silently mislabels the
  other, and a mislabelled answer is not a small penalty: `align_predictions` refuses a declared
  type that disagrees with the trusted task as `t4.target_type_mismatch` ->
  `SCHEMA_INVALID_OUTPUT`, scored at the worst case `W = -0.27`.

* **The fallback citation.** When retrieval returns nothing the baseline still has to emit >= 1
  claim. It used to cite `docs[0]` -- the first corpus file in sorted order, whose date has nothing
  to do with the cutoff -- at offsets `(0, 1)`. Under `strict = true` embargo that is a cutoff
  violation whenever that first file post-dates the cutoff.

Standard library only, on purpose: this suite runs in the secret-free CI job, which installs
`pytest` and nothing else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BASELINES = Path(__file__).resolve().parents[1]
_REPO = _BASELINES.parent
sys.path.insert(0, str(_BASELINES))

from baseline_agent.cli import _newest_eligible, _target_type, run  # noqa: E402
from baseline_agent.indexer import IndexedDoc  # noqa: E402

CUTOFF = "2024-03-15"


# --------------------------------------------------------------------------------------------
# target_type: both published shapes
# --------------------------------------------------------------------------------------------
def _task(**overrides: object) -> dict:
    task = {
        "task_id": "t4-SHAPES",
        "cutoff_date": CUTOFF,
        "entities": [{"entity_id": "AAA", "name": "Issuer AAA"}],
    }
    task.update(overrides)
    return task


def test_the_documented_top_level_shape_is_read() -> None:
    """SUBMISSION_CLI.md:107 / invariant 7 publish `target_type` at the top level."""
    assert _target_type(_task(target_type="regression")) == "regression"


def test_the_nested_shape_every_task_json_in_this_repo_uses_is_read() -> None:
    assert _target_type(_task(target={"name": "y", "type": "ranking"})) == "ranking"


def test_the_documented_field_wins_when_a_task_carries_both() -> None:
    task = _task(target_type="regression", target={"name": "y", "type": "classification"})
    assert _target_type(task) == "regression"


def test_a_task_declaring_neither_yields_none_rather_than_a_guess() -> None:
    """`None` is the one safe answer: an ABSENT target_type is accepted and scored by the
    scorer, a WRONG one is t4.target_type_mismatch and loses the whole unit."""
    assert _target_type(_task()) is None
    assert _target_type(_task(target="not-an-object")) is None


def test_a_malformed_target_block_does_not_crash_the_agent() -> None:
    assert _target_type(_task(target=["classification"])) is None


@pytest.mark.parametrize("declared", ["classification", "regression", "ranking"])
def test_the_emitted_answer_carries_the_target_type_the_documented_shape_declares(
    tmp_path: Path, declared: str
) -> None:
    unit = _REPO / "units" / "t4-EXAMPLE-eps-beat"
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    task.pop("target", None)  # exactly the shape SUBMISSION_CLI.md publishes
    task["target_type"] = declared
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")

    answer = run(task_path, unit / "corpus", tmp_path / "answer.json")
    assert answer["target_type"] == declared


def test_the_answer_omits_target_type_when_the_task_declares_none(tmp_path: Path) -> None:
    unit = _REPO / "units" / "t4-EXAMPLE-eps-beat"
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    task.pop("target", None)
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")

    answer = run(task_path, unit / "corpus", tmp_path / "answer.json")
    assert "target_type" not in answer


# --------------------------------------------------------------------------------------------
# the no-hit fallback
# --------------------------------------------------------------------------------------------
def _doc(doc_id: str, date: str | None, text: str) -> IndexedDoc:
    return IndexedDoc(doc_id=doc_id, text=text, doc_date=date)


def test_newest_eligible_never_returns_a_post_cutoff_document() -> None:
    """`docs[0]` is sorted-order, not date-order. Here it is 78 days past the cutoff."""
    docs = [
        _doc("A_POST", "2024-06-01", "post"),  # sorts first; 78 days past CUTOFF
        _doc("B_PRE", "2024-01-02", "pre"),
    ]
    assert docs[0].doc_id == "A_POST"
    chosen = _newest_eligible(docs, CUTOFF)
    assert chosen is not None
    assert chosen.doc_id == "B_PRE"


def test_newest_eligible_picks_the_newest_of_the_eligible_ones() -> None:
    docs = [
        _doc("A", "2024-01-02", "old"),
        _doc("B", "2024-03-14", "newest eligible"),
        _doc("C", "2024-03-16", "one day too late"),
    ]
    chosen = _newest_eligible(docs, CUTOFF)
    assert chosen is not None
    assert chosen.doc_id == "B"


def test_newest_eligible_is_none_when_nothing_predates_the_cutoff() -> None:
    """No substitute is offered. Every alternative scores the same W = -0.27, and one of
    them (citing a post-cutoff document) is a rule violation the baseline would be
    authoring on the participant's behalf."""
    assert _newest_eligible([_doc("A", "2024-06-01", "post")], CUTOFF) is None
    assert _newest_eligible([], CUTOFF) is None


def test_an_undated_document_is_never_chosen() -> None:
    """The scorer refuses to load a unit containing an undated corpus document
    (T4OrganizerFault, "carries no doc_date"), so it can never be a legal citation."""
    assert _newest_eligible([_doc("A", None, "undated")], CUTOFF) is None


# A pre-cutoff document the lexical retriever cannot score: every token is <= 2 characters,
# so `retrieve` skips it and the fallback is what actually fires.
_UNRETRIEVABLE = "an is at on to by of " * 40


def _corpus(tmp_path: Path, docs: dict[str, tuple[str, str]]) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for doc_id, (date, text) in docs.items():
        (corpus / f"{doc_id}.json").write_text(
            json.dumps({"doc_id": doc_id, "doc_date": date, "text": text}), encoding="utf-8"
        )
    return corpus


def _run_on(tmp_path: Path, corpus: Path) -> dict:
    task_path = tmp_path / "task.json"
    task_path.write_text(
        json.dumps(
            {
                "task_id": "t4-FALLBACK",
                "cutoff_date": CUTOFF,
                "target": {"type": "classification"},
                "entities": [{"entity_id": "AAA", "name": "Issuer AAA"}],
            }
        ),
        encoding="utf-8",
    )
    return run(task_path, corpus, tmp_path / "answer.json")


def test_the_fallback_cites_an_eligible_document_with_a_usable_span(tmp_path: Path) -> None:
    corpus = _corpus(
        tmp_path,
        {
            # sorts first AND post-dates the cutoff: this is the document docs[0] handed out
            "AAA_POST_20240601": ("2024-06-01", "Rich retrievable earnings text about issuer."),
            "ZZZ_PRE_20240102": ("2024-01-02", _UNRETRIEVABLE),
        },
    )
    claim = _run_on(tmp_path, corpus)["entity_predictions"][0]["claims"][0]
    assert claim["doc_id"] == "ZZZ_PRE_20240102"
    # A one-character premise is t4.evidence_unsupported; the fallback quotes up to 240 chars.
    assert claim["span_end"] == 240


def test_the_fallback_says_unknown_rather_than_citing_a_post_cutoff_document(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path, {"AAA_POST_20240601": ("2024-06-01", _UNRETRIEVABLE)})
    claim = _run_on(tmp_path, corpus)["entity_predictions"][0]["claims"][0]
    assert claim["doc_id"] == "UNKNOWN"


# --------------------------------------------------------------------------------------------
# published prose that contradicts the scorer
# --------------------------------------------------------------------------------------------
def test_the_shipped_answer_template_declares_no_target_type() -> None:
    """`target_type` is a closed enum in analysis.schema.json, so there is no legal REPLACE
    marker for it: a placeholder string fails g1_schema outright. A copied-in literal
    "classification" is worse -- it is a DNF on every regression and ranking unit -- so the
    template omits the field and its notes tell the participant to add it."""
    template = json.loads((_REPO / "templates" / "answer.example.json").read_text("utf-8"))
    assert "target_type" not in template
    assert "target_type" in template["notes"]


def test_the_readme_does_not_tell_ranking_entries_to_put_the_rank_in_label() -> None:
    """`label` is never read on a ranking unit; the ordering comes from `point_forecast`.

    Measured: the rank INTEGER placed in `point_forecast` (1 = highest) inverts the ordering
    and scores predictive_quality 0.0 / composite -0.03, against 1.0 / 0.67 for the metric
    values. This paragraph used to end "a constant or absent `point_forecast` is scored by row
    order alone -- 1.0 or 0.0 for an answer that predicts nothing", which described the
    position-tie-breaking exploit and is no longer true. Re-measured 2026-08-29: a CONSTANT
    `point_forecast` ties every entity, and ties now rank as ties, so it scores the neutral
    predictive_quality 0.5 / composite 0.32 -- strictly below the honest answer, never equal to
    it (pinned by `test_a_constant_point_forecast_scores_the_neutral_value_not_full_marks`).
    OMITTING `point_forecast` on a ranking unit is not scored by row order either: it is
    refused at g1_schema for the whole submission at W = -0.27."""
    readme = (_REPO / "README.md").read_text(encoding="utf-8")
    assert "`label` carries the rank" not in readme


def test_no_published_doc_calls_a_schema_failure_a_coverage_penalty() -> None:
    """Measured: a missing lo/hi, an empty claims[] and an absent claims key are each
    t4.schema_invalid at W = -0.27 for the whole submission, with no coverage number
    computed at all. `strong_rag_baseline/agent.py` gave the same wrong reason for the
    right behaviour."""
    checked = [
        _REPO / "README.md",
        _REPO / "baselines" / "strong_rag_baseline" / "agent.py",
    ]
    # Control: the sweep must be able to see the surrounding text at all.
    assert any("coverage" in p.read_text(encoding="utf-8") for p in checked)
    for path in checked:
        text = path.read_text(encoding="utf-8")
        assert "= zero coverage score" not in text
        assert "scores zero coverage for the entity" not in text


def test_no_published_doc_still_calls_the_wall_clock_budget_cpu_only() -> None:
    """Every card in this repo declares `gpu = true`; "(CPU)" contradicted the card the same
    sentence cites as authoritative. Whether participants are PROMISED a GPU is an organizer
    decision, so the false claim is removed rather than replaced."""
    checked = [
        _REPO / "README.md",
        _REPO / "baselines" / "README.md",
        _REPO / "baselines" / "strong_rag_baseline" / "README.md",
        _REPO / "baselines" / "strong_rag_baseline" / "retriever.py",
        _REPO / "faithfulness" / "judge.py",
        _REPO / "units" / "t4-EXAMPLE-eps-beat" / "card.toml",
    ]
    # Control: the sweep must be able to see the string at all.
    assert any("10 minutes per unit" in p.read_text(encoding="utf-8") for p in checked)
    offenders = [
        p.name
        for p in checked
        if "(CPU)" in p.read_text(encoding="utf-8")
        or "CPU-only" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
