"""Scorer 5.3.0: only the first 20 claims about each entity are checked and counted.

Scorer 5.3.0 rule (adopted 2026-10-03): claims after the 20th about an entity, in the order they appear in
`answer.json`, are IGNORED, not penalised. No per-claim check runs on them, the judge is not
asked about them, they are not counted in F or T, and the recorded `prediction_relevance`
diagnostic does not read their citations. The cap is per entity, not per unit, and it is applied
before anything groups claims per entity, so the Final, Development and the local checker see the
same counted claims. The answer as a whole is still checked structurally: the citation date check
applies to every claim.

Every scored comparison below is anchored by a positive control: the same five planted claims,
placed where they ARE counted, change the score. A cap test that only showed "no change" could
pass because the planted claims were harmless to begin with.
"""

from __future__ import annotations

import importlib
import json
import pathlib
from typing import Any

from qfbench2_track_analysis import judge_factory
from qfbench2_track_analysis.alignment import (
    DATE_CHECK_COVERS_IGNORED_CLAIMS,
    MAX_CLAIMS_PER_ENTITY,
)
from qfbench2_track_analysis.scoring import UnitOutcome, score_unit

from .synthetic import (
    POST_CUTOFF_DOC,
    PRE_CUTOFF_DOC,
    SUPPORTING_TEXT,
    TRIVIA_DOC,
    StubJudge,
    answer_for,
    build_unit,
)

ROSTER = ("SYN-A", "SYN-B", "SYN-C")
#: TRIVIA_DOC is labelled for SYN-C only, so SYN-A or SYN-B citing it is a wrong-entity claim.
LABELS = {TRIVIA_DOC: {"entity_ids": ["SYN-C"]}}

GOOD = "Synthetic Issuer A beat the published consensus."
#: A figure no cited span carries: false (`unanchored`) wherever it is counted.
FABRICATED = "Synthetic Issuer A reported revenue of $9,999 million."


def _claim(text: str, doc_id: str = PRE_CUTOFF_DOC) -> dict[str, Any]:
    span_end = len(SUPPORTING_TEXT) if doc_id == PRE_CUTOFF_DOC else 20
    return {"doc_id": doc_id, "span_start": 0, "span_end": span_end, "claim": text}


#: The first 20 claims for an entity: one false (a fabricated figure) and 19 neutral, so the
#: counted claims already carry a penalty and an equal score is not equality at factor 1.
FIRST_20 = [_claim(FABRICATED)] + [_claim(GOOD) for _ in range(19)]
#: Five claims that are false if counted: fabricated figures and wrong-entity citations.
PLANTED = [
    _claim(FABRICATED),
    _claim(GOOD, TRIVIA_DOC),
    _claim("Synthetic Issuer A grew sales 77% to $12.5 billion."),
    _claim(GOOD, TRIVIA_DOC),
    _claim(FABRICATED),
]


def _answer(claims_by_entity: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    answer = answer_for(ROSTER)
    for row in answer["entity_predictions"]:
        if row["entity_id"] in claims_by_entity:
            row["claims"] = claims_by_entity[row["entity_id"]]
    return answer


def _score(
    tmp_path: pathlib.Path, answer: dict[str, Any], judge: Any | None = None
) -> tuple[UnitOutcome, StubJudge]:
    unit = build_unit(tmp_path, entities=ROSTER, with_outcome=True, labels=LABELS)
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    judge = judge or StubJudge((SUPPORTING_TEXT,))
    _, provenance = judge_factory.build_smoke_judge()
    outcome = score_unit(
        {"unit_dir": unit, "output_dir": out}, judge=judge, judge_provenance=provenance
    )
    return outcome, judge


def test_the_cap_is_twenty() -> None:
    assert MAX_CLAIMS_PER_ENTITY == 20


def test_the_date_check_covers_ignored_claims_by_choice() -> None:
    """The one switch for whether the date check reads ignored claims' citations (see the next tests)."""
    assert DATE_CHECK_COVERS_IGNORED_CLAIMS is True


def test_twenty_five_claims_on_one_entity_score_as_the_first_twenty(
    tmp_path: pathlib.Path,
) -> None:
    """The 25-claim case: claims 21-25 are false if counted, and change nothing."""
    twenty, _ = _score(tmp_path / "a", _answer({"SYN-A": FIRST_20}))
    capped, _ = _score(tmp_path / "b", _answer({"SYN-A": FIRST_20 + PLANTED}))
    assert twenty.state == capped.state == "participant_success"
    assert capped.score == twenty.score
    for key in (
        "faithfulness",
        "faithfulness_factor",
        "claim_count",
        "false_claim_count",
        "unanchored_claim_count",
        "wrong_entity_claim_count",
        "prediction_relevance",
    ):
        assert capped.diagnostics[key] == twenty.diagnostics[key], key
    assert (
        twenty.diagnostics["claim_count"] == 22
    )  # 20 for SYN-A, 1 each for SYN-B, SYN-C
    assert twenty.diagnostics["false_claim_count"] == 1
    assert twenty.diagnostics["faithfulness_factor"] < 1.0
    assert capped.diagnostics["ignored_claim_count"] == 5
    assert capped.diagnostics["ignored_claims_by_entity"] == {"SYN-A": 5}
    assert twenty.diagnostics["ignored_claim_count"] == 0
    assert twenty.diagnostics["ignored_claims_by_entity"] == {}


def test_file_order_decides_which_claims_count(tmp_path: pathlib.Path) -> None:
    """Positive control: the same 25 claims with the planted five FIRST put them in the
    counted twenty, and every one of them is false."""
    late, _ = _score(tmp_path / "a", _answer({"SYN-A": FIRST_20 + PLANTED}))
    early, _ = _score(tmp_path / "b", _answer({"SYN-A": PLANTED + FIRST_20}))
    assert early.diagnostics["ignored_claim_count"] == 5
    assert (
        early.diagnostics["false_claim_count"] == 6
    )  # the five planted + the fabricated #6
    assert early.diagnostics["wrong_entity_claim_count"] == 2
    assert late.diagnostics["false_claim_count"] == 1
    assert early.score < late.score


def test_the_cap_is_per_entity_not_per_unit(tmp_path: pathlib.Path) -> None:
    """Two entities with 25 claims each: 20 counted for EACH (40 in all), 5 ignored for each."""
    both, _ = _score(
        tmp_path / "a",
        _answer({"SYN-A": FIRST_20 + PLANTED, "SYN-B": FIRST_20 + PLANTED}),
    )
    first, _ = _score(tmp_path / "b", _answer({"SYN-A": FIRST_20, "SYN-B": FIRST_20}))
    assert both.diagnostics["claim_count"] == 41  # 20 + 20 + SYN-C's one
    assert both.diagnostics["ignored_claims_by_entity"] == {"SYN-A": 5, "SYN-B": 5}
    assert both.diagnostics["ignored_claim_count"] == 10
    assert both.diagnostics["false_claim_count"] == 2
    assert both.score == first.score


def test_ignored_claims_make_no_judge_call_at_all(tmp_path: pathlib.Path) -> None:
    """Counting stub: the 25-claim answer asks the judge exactly the questions the 20-claim
    answer asks, contradiction AND the `prediction_relevance` entailments (which read every
    citation of the entity; capped too)."""
    _, twenty = _score(tmp_path / "a", _answer({"SYN-A": FIRST_20}))
    _, capped = _score(tmp_path / "b", _answer({"SYN-A": FIRST_20 + PLANTED}))
    assert capped.calls == twenty.calls
    assert capped.contradiction_calls == twenty.contradiction_calls
    entail_calls = len(twenty.calls) - len(twenty.contradiction_calls)
    # prediction_relevance: 3 phrasings x (20 + 1 + 1) citations; contradiction: the 19
    # neutral SYN-A claims and SYN-B's and SYN-C's (the fabricated figure is not judged).
    assert entail_calls == 3 * 22
    assert len(twenty.contradiction_calls) == 21
    # Positive control: uncapped (planted first), the planted claims' citations ARE read.
    _, early = _score(tmp_path / "c", _answer({"SYN-A": PLANTED + FIRST_20}))
    assert early.calls != twenty.calls


def test_an_ignored_claim_is_still_checked_for_its_date(tmp_path: pathlib.Path) -> None:
    """Structural, not per-claim: a post-cutoff citation anywhere in the answer refuses the unit,
    an ignored claim's included (the answer used material it may not have read)."""
    late = _claim(GOOD, POST_CUTOFF_DOC)
    outcome, judge = _score(tmp_path, _answer({"SYN-A": FIRST_20 + [late]}))
    assert outcome.state == "participant_failure"
    assert outcome.diagnostics["reason"] == "t4.citation_post_cutoff"
    assert judge.calls == []


def test_the_local_checker_applies_the_same_cap(tmp_path: pathlib.Path) -> None:
    """`check_claim_rules` / `claim_penalty_preview` (the participant's local checker) on the
    25-claim answer: the same counted claims, the same false ones, the same factor, and a
    `claims_ignored` notice for the entity instead of findings on claims 21-25."""
    # Imported by name: the checker lives in the baselines tree, outside the typed packages.
    rail = importlib.import_module("baselines.guardrails_example.citation_rail")
    check_claim_rules, claim_penalty_preview = (
        rail.check_claim_rules,
        rail.claim_penalty_preview,
    )

    answer = _answer({"SYN-A": FIRST_20 + PLANTED})
    outcome, _ = _score(tmp_path, answer)
    unit = tmp_path / "t4-SYNTH"
    findings = check_claim_rules(answer, unit, token_counter=lambda t: len(t.split()))
    per_claim = sorted(
        (f.entity_id, f.claim_index, f.code) for f in findings if f.claim_index >= 0
    )
    assert per_claim == [("SYN-A", 0, "claim_unanchored")]
    notices = [(f.entity_id, f.code) for f in findings if f.claim_index < 0]
    assert notices == [("SYN-A", "claims_ignored")]
    preview = claim_penalty_preview(
        answer, unit, token_counter=lambda t: len(t.split())
    )
    assert preview["claims"] == outcome.diagnostics["claim_count"] == 22
    assert preview["false"] == outcome.diagnostics["false_claim_count"] == 1
    assert preview["ignored"] == outcome.diagnostics["ignored_claim_count"] == 5
    assert preview["factor"] == outcome.diagnostics["faithfulness_factor"]
    # Positive control: with the planted claims counted, the checker reports each of them.
    early = check_claim_rules(
        _answer({"SYN-A": PLANTED + FIRST_20}),
        unit,
        token_counter=lambda t: len(t.split()),
    )
    assert sum(1 for f in early if f.claim_index >= 0) == 6
