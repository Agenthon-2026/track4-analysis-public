"""The local rail's 5.2.0 claim rules (from a documentation audit, 2026-09-29).

`check_answer` must accept a valid task-table citation (``doc_id: "task"``) and flag an invalid
one; `check_claim_rules` must report the deterministic 5.2.0 claim verdicts (wrong entity, out of
range, malformed incl. the judge-token cap, every figure with the verbatim-quote pass, the
name/ticker exemption and the 8,000-character span cap) exactly as the scorer's own gate does;
`check_submitted_reasons` must flag content-duplicate reasons and task-table reason citations.

Synthetic units only (`scoring/tests/synthetic.py`); no model is loaded (the judge-token cap uses
a word-counting stand-in, passed explicitly).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.guardrails_example.citation_rail import (
    CorpusDoc,
    check_answer,
    check_claim_rules,
    check_submitted_reasons,
    claim_penalty_preview,
)

# These checks run the scorer's own claim gate, which needs the shared toolkit. The stdlib-only
# CI job installs nothing but pytest, so the module skips there instead of failing collection.
pytest.importorskip("qfbench2_common", reason="needs the shared toolkit (qfbench2_common)")

from qfbench2_track_analysis.corpus import task_table_text  # noqa: E402
from scoring.tests.synthetic import build_unit  # noqa: E402

A_TEXT = "Synthetic Issuer A reported revenue of $4,210 million and diluted EPS of $1.37 for the quarter."
B_TEXT = "Synthetic Issuer B reported revenue of $980 million for the quarter."
LONG_TEXT = ("Filler text without any amounts in it. " * 240) + "Segment revenue of $5,555 million was reported."
DOCS = {
    "SYNDOC_A_20260201": {"doc_id": "SYNDOC_A_20260201", "doc_date": "2026-02-01", "text": A_TEXT},
    "SYNDOC_B_20260201": {"doc_id": "SYNDOC_B_20260201", "doc_date": "2026-02-01", "text": B_TEXT},
    "SYNDOC_LONG_20260201": {"doc_id": "SYNDOC_LONG_20260201", "doc_date": "2026-02-01", "text": LONG_TEXT},
}
LABELS = {
    "SYNDOC_A_20260201": {"entity_ids": ["SYN-A"]},
    "SYNDOC_B_20260201": {"entity_ids": ["SYN-B"]},
}
ENTITIES = ("SYN-A", "SYN-B")


def words(text: str) -> int:
    """A stand-in judge tokenizer: one token per whitespace-separated word."""
    return len(text.split())


@pytest.fixture()
def unit(tmp_path: Path) -> Path:
    u = build_unit(tmp_path, entities=ENTITIES, docs=DOCS, labels=LABELS, with_outcome=True)
    task = json.loads((u / "task.json").read_text())
    task["entities"][0].update(name="Phillips 66 Synthetic", prior_value=12.5)
    task["entities"][1].update(prior_value=7.25)
    (u / "task.json").write_text(json.dumps(task, indent=1) + "\n")
    return u


def _task(unit: Path) -> dict:
    return json.loads((unit / "task.json").read_text())


def cite(doc_id: str, start: int, end: int, claim: str) -> dict:
    return {"doc_id": doc_id, "span_start": start, "span_end": end, "claim": claim}


def whole(doc_id: str, claim: str) -> dict:
    return cite(doc_id, 0, len(DOCS[doc_id]["text"]), claim)


def answer(claims_a: list[dict], claims_b: list[dict] | None = None) -> dict:
    rows = []
    for eid, claims in (("SYN-A", claims_a), ("SYN-B", claims_b or [whole("SYNDOC_B_20260201", "Revenue was $980 million.")])):
        rows.append({"entity_id": eid, "label": "beat", "point_forecast": 1.0,
                     "interval": {"level": 0.9, "lo": 0.5, "hi": 3.5}, "claims": claims})
    return {"task_id": "t4-SYNTH", "schema_version": "3", "target_type": "classification",
            "entity_predictions": rows}


def task_row_cite(unit: Path, entity: str, claim: str) -> dict:
    _, ranges = task_table_text(_task(unit))
    start, end = ranges[entity]
    return cite("task", start, end, claim)


# ---------------------------------------------------------------- task-table citations in claims


def test_a_task_citation_inside_the_own_row_is_clean(unit: Path) -> None:
    ans = answer([task_row_cite(unit, "SYN-A", "The prior value was 12.5.")])
    corpus = {d: CorpusDoc(d, v["text"], v["doc_date"]) for d, v in DOCS.items()}
    assert [f.code for f in check_answer(ans, corpus, "2026-02-15", task=_task(unit))] == []


def test_a_task_citation_of_another_row_or_across_rows_or_out_of_range_is_flagged(unit: Path) -> None:
    corpus = {d: CorpusDoc(d, v["text"], v["doc_date"]) for d, v in DOCS.items()}
    text, ranges = task_table_text(_task(unit))
    other = task_row_cite(unit, "SYN-B", "The prior value was 7.25.")
    across = cite("task", ranges["SYN-A"][0], ranges["SYN-B"][1], "Two rows.")
    past = cite("task", 0, len(text) + 1, "Past the end.")
    got = [f.code for f in check_answer(answer([other, across, past]), corpus, "2026-02-15", task=_task(unit))]
    assert got == ["task_row", "task_row", "bad_span"]


def test_without_the_task_a_task_citation_is_not_called_unknown(unit: Path) -> None:
    corpus = {d: CorpusDoc(d, v["text"], v["doc_date"]) for d, v in DOCS.items()}
    got = [f.code for f in check_answer(answer([task_row_cite(unit, "SYN-A", "x")]), corpus, "2026-02-15")]
    assert got == ["task_unchecked"]


# ---------------------------------------------------------------- deterministic claim rules

CASES = [
    # (claim, expected checker code or None)
    (whole("SYNDOC_A_20260201", "Revenue was $4,210 million."), None),
    (whole("SYNDOC_A_20260201", "Revenue grew 5% to $4,210 million."), "claim_unanchored"),
    (whole("SYNDOC_B_20260201", "Revenue was $980 million."), "claim_wrong_entity"),
    (cite("SYNDOC_A_20260201", 0, len(A_TEXT) + 5, "Revenue was reported."), "claim_out_of_range"),
    # the 8,000-character cap: a paraphrase is not anchored by an over-cap span ...
    (whole("SYNDOC_LONG_20260201", "Segment revenue was $5,555 million."), "claim_unanchored"),
    # ... but a verbatim quote passes the figure rule in a span of any length
    (whole("SYNDOC_LONG_20260201", "Segment revenue of $5,555 million was reported."), None),
    # the unit's own entity name carries a number that is not a figure; an altered one is
    (whole("SYNDOC_A_20260201", "Phillips 66 Synthetic reported revenue of $4,210 million."), None),
    (whole("SYNDOC_A_20260201", "Phillips 67 Synthetic reported revenue of $4,210 million."), "claim_unanchored"),
    (whole("SYNDOC_A_20260201", " ".join(["word"] * 401)), "claim_malformed"),
    (whole("SYNDOC_A_20260201", ""), "claim_malformed"),
]


def _checker(unit: Path, claims: list[dict], counter=words) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for f in check_claim_rules(answer(claims), unit, token_counter=counter):
        if f.entity_id == "SYN-A":
            out.setdefault(f.claim_index, set()).add(f.code)
    return out


@pytest.mark.parametrize("case", range(len(CASES)))
def test_each_rule(unit: Path, case: int) -> None:
    claim, expected = CASES[case]
    got = _checker(unit, [claim])
    assert got == ({} if expected is None else {0: {expected}})


def test_a_task_citation_is_checked_like_the_scorer(unit: Path) -> None:
    own = task_row_cite(unit, "SYN-A", "The prior value was 12.5.")
    other = task_row_cite(unit, "SYN-B", "The prior value was 7.25.")
    invented = task_row_cite(unit, "SYN-A", "The prior value was 13.5.")
    assert _checker(unit, [own, other, invented]) == {1: {"claim_wrong_entity"}, 2: {"claim_unanchored"}}


def test_without_a_tokenizer_the_token_cap_is_reported_unchecked(unit: Path) -> None:
    got = check_claim_rules(answer([whole("SYNDOC_A_20260201", "Revenue was $4,210 million.")]), unit,
                            token_counter=None)
    assert [(f.code, f.claim_index) for f in got] == [("claim_tokens_unchecked", -1)]


def test_a_unit_the_scorer_refuses_is_reported_as_refused(unit: Path) -> None:
    got = check_claim_rules(answer([whole("SYNDOC_A_20260201", "x")])
                            | {"entity_predictions": answer([])["entity_predictions"][:1]}, unit,
                            token_counter=words)
    assert [f.code for f in got] == ["unit_refused"]


def test_checker_and_scorer_agree_claim_by_claim(unit: Path) -> None:
    """Parity: every claim's deterministic verdict equals the one the scorer's own gate
    (`score_unit`, g0-g3) records, with the same token counter."""
    from qfbench2_track_analysis.judge_factory import build_smoke_judge
    from qfbench2_track_analysis.scoring import score_unit

    claims = [c for c, _ in CASES] + [task_row_cite(unit, "SYN-B", "The prior value was 7.25.")]
    ans = answer(claims)
    checker = _checker(unit, claims)

    smoke, prov = build_smoke_judge()

    class Counting:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def claim_tokens(self, hypothesis: str) -> int:
            return words(hypothesis)

    out = unit.parent / "out"
    out.mkdir()
    (out / "answer.json").write_text(json.dumps(ans))
    outcome = score_unit({"unit_dir": str(unit), "output_dir": str(out)}, judge=Counting(smoke),
                         judge_provenance=prov)
    scorer: dict[int, set[str]] = {}
    texts = [c["claim"] for c in claims]
    for fc in outcome.diagnostics["false_claims"]:
        if fc["entity_id"] != "SYN-A":
            continue
        reasons = {f"claim_{r}" for r in fc["reasons"] if r != "contradicted"}
        if reasons:
            scorer.setdefault(texts.index(fc["claim"]), set()).update(reasons)
    assert checker == scorer
    assert len(scorer) == 8  # positive control: the planted false claims are there on both sides
    preview = claim_penalty_preview(ans, unit, token_counter=words)
    assert preview["factor"] == outcome.diagnostics["faithfulness_factor"]
    assert preview["entities"] == 2 and 0.0 < preview["factor"] < 1.0


# ---------------------------------------------------------------- reasons


def _reason(rid: str, premise: str = "p", **extra) -> dict:
    return {"reason_id": rid, "premise": premise, "mechanism": "m", "answer_implication": "a", **extra}


def test_a_content_duplicate_reason_is_flagged() -> None:
    corpus = {"d": CorpusDoc("d", "text", "2026-01-01")}
    ans = {"entity_predictions": [], "submitted_reasons": [
        _reason("r1", "Deposits rose."), _reason("r2", "  deposits   ROSE. "), _reason("r3", "Deposits fell.")]}
    got = [(f.code, f.claim_index) for f in check_submitted_reasons(ans, corpus, "2026-02-15")]
    assert got == [("duplicate_reason", 1)]


def test_a_task_citation_in_a_reason_is_flagged_with_its_own_message() -> None:
    corpus = {"d": CorpusDoc("d", "text", "2026-01-01")}
    ans = {"entity_predictions": [], "submitted_reasons": [
        _reason("r1", citations=[{"doc_id": "task", "span_start": 0, "span_end": 3}])]}
    (f,) = check_submitted_reasons(ans, corpus, "2026-02-15")
    assert f.code == "reason_citation" and "task table" in f.message


@pytest.mark.parametrize(("pad", "factor"), [(0, 1 - 1 / 2), (5, 1 - 1 / 7), (40, 1 - 1 / 7)])
def test_the_preview_factor_is_the_soft_floor(unit: Path, pad: int, factor: float) -> None:
    """One false claim for SYN-A and one true claim for SYN-B, plus `pad` neutral claims for
    SYN-A: padding dilutes the false claim only up to 3 x E = 6 claims in total (E = 2)."""
    neutral = [whole("SYNDOC_A_20260201", f"Issuer A published results, note {chr(97 + i % 26)}.") for i in range(pad)]
    ans = answer([whole("SYNDOC_A_20260201", "Revenue was $9,999 million.")] + neutral)
    got = claim_penalty_preview(ans, unit, token_counter=words)
    assert (got["false"], got["claims"], got["entities"]) == (1, pad + 2, 2)
    assert got["factor"] == pytest.approx(factor)
