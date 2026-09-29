"""The judge reads the cited passage against the participant's CLAIM.

5.2.0: the question is no longer "does the passage entail the claim above tau, for 80% of
claims" but "does the passage CONTRADICT the claim" (three-way P(contradiction) > bar), and a false
claim costs its share of the unit instead of refusing it. The hypothesis is still the claim.

The alternative -- asking whether the passage entails a canonical sentence built from the
submitted prediction -- would stop accurate prose from making a wrong forecast look "faithful",
but it is not a question the corpus can answer: a pre-cutoff passage does not state a
post-cutoff outcome, so nothing in the corpus entails a forecast and the score falls back to
topical consistency, which is as high for an unrelated passage of the cited document as for the
honestly cited one. So the hypothesis is the claim, the premise is the cited span, and admission
is the fraction of claims supported. The prediction-derived score is still computed and recorded
as `prediction_relevance`, and these tests hold it to being recorded and NOT enforced.

What the gate is for, stated so the tests below read as a set: it catches fabricated or
misattributed evidence. An accurate claim about a passage that says nothing about the forecast is
an accurate claim; whether the evidence supports the forecast is reasoning grading's question.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from qfbench2_track_analysis.alignment import EntityRoster, align_predictions
from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import (
    DOMAIN_MIN,
    UnitOutcome,
    _g3_domain_semantics,
    evaluate_claims,
    hydrate,
    score_unit,
)

from .synthetic import (
    MISS_MARKER,
    PRE_CUTOFF_DOC,
    SUPPORTING_TEXT,
    TRIVIA_DOC,
    TRIVIA_TEXT,
    HypothesisAwareJudge,
    StubJudge,
    answer_for,
    build_unit,
)

ROSTER = ("SYN-A",)

#: Prose that accurately describes `SUPPORTING_TEXT`.
ACCURATE_PROSE = (
    # A paraphrase, not a verbatim quote: from 5.2.0 a word-for-word quote of the
    # cited passage is not put to the judge at all, and these tests need the judge's question.
    "Synthetic Issuer A's quarterly earnings came in above the published consensus."
)
#: Prose that contradicts it.
CONTRADICTING_PROSE = (
    "Synthetic Issuer A reported quarterly earnings below the published consensus."
)


def _run(tmp_path: pathlib.Path, answer: dict[str, Any], judge: object) -> UnitOutcome:
    unit = build_unit(tmp_path, entities=ROSTER, with_outcome=True)
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out}, judge=judge, judge_provenance=provenance
    )


def _accuracy_judge() -> HypothesisAwareJudge:
    """Entails the supporting passage only for a hypothesis that describes it, and finds it
    contradicted by a hypothesis that says the opposite."""
    return HypothesisAwareJudge(
        SUPPORTING_TEXT,
        required_in_hypothesis="above the published consensus",
        contradicting_in_hypothesis="below the published consensus",
    )


# --- the hypothesis is the claim ---------------------------------------------------------------
def test_an_accurate_claim_is_neutral_and_costs_nothing(tmp_path: pathlib.Path) -> None:
    judge = _accuracy_judge()
    outcome = _run(
        tmp_path, answer_for(entities=ROSTER, claim_text=ACCURATE_PROSE), judge
    )
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["faithfulness"] == 1.0
    assert outcome.diagnostics["faithfulness_factor"] == 1.0
    assert outcome.score == outcome.diagnostics["composite_before_penalty"]
    # The question the judge was asked: premise = the cited span, hypothesis = the claim.
    assert (SUPPORTING_TEXT, ACCURATE_PROSE) in judge.contradiction_calls


def test_a_contradicted_claim_is_penalised_not_refused(tmp_path: pathlib.Path) -> None:
    judge = _accuracy_judge()
    outcome = _run(
        tmp_path, answer_for(entities=ROSTER, claim_text=CONTRADICTING_PROSE), judge
    )
    # 5.2.0: never a refusal; the unit's only claim is false, so the factor is (1 - 1/1) = 0.
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["false_claim_count"] == 1
    assert outcome.diagnostics["contradicted_claim_count"] == 1
    assert outcome.diagnostics["faithfulness"] == 0.0
    assert outcome.diagnostics["faithfulness_factor"] == 0.0
    assert outcome.diagnostics["composite_before_penalty"] > 0.0
    assert outcome.score == DOMAIN_MIN
    assert (SUPPORTING_TEXT, CONTRADICTING_PROSE) in judge.contradiction_calls


def test_a_quoted_span_as_a_claim_is_admitted(tmp_path: pathlib.Path) -> None:
    """Quoting the passage is an accurate claim. Weak reasoning is the reasoning grader's to
    score; the penalty falls only on what is demonstrably wrong."""
    judge = StubJudge((SUPPORTING_TEXT,))
    outcome = _run(
        tmp_path, answer_for(entities=ROSTER, claim_text=SUPPORTING_TEXT), judge
    )
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["faithfulness"] == 1.0


def test_the_hypothesis_is_the_participants_text_verbatim(
    tmp_path: pathlib.Path,
) -> None:
    marker = "PARTICIPANT-AUTHORED-MARKER-STRING"
    judge = StubJudge((SUPPORTING_TEXT,))
    _run(tmp_path, answer_for(entities=ROSTER, claim_text=marker), judge)
    hypotheses = {hypothesis for _, hypothesis in judge.calls}
    assert marker in hypotheses


def test_changing_only_the_prediction_does_not_change_the_gate_question(
    tmp_path: pathlib.Path,
) -> None:
    """The gate's question is the claim; the prediction-derived question is the diagnostic's."""
    beat_judge = StubJudge((SUPPORTING_TEXT,))
    _run(tmp_path / "beat", answer_for(entities=ROSTER, label="beat"), beat_judge)
    miss_judge = StubJudge((SUPPORTING_TEXT,))
    _run(tmp_path / "miss", answer_for(entities=ROSTER, label="miss"), miss_judge)
    # First call in each run is the gate's: same premise, same claim.
    assert beat_judge.calls[0] == miss_judge.calls[0]
    # The diagnostic's questions differ, because they are built from the prediction.
    assert {h for _, h in beat_judge.calls} != {h for _, h in miss_judge.calls}


# --- prediction relevance: recorded, never enforced --------------------------------------------
def test_accurate_prose_about_an_irrelevant_passage_is_admitted_and_the_relevance_is_recorded(
    tmp_path: pathlib.Path,
) -> None:
    """The submission predicts `miss`; its one claim accurately describes a passage that says
    nothing about the forecast. That is an accurate claim, so the gate admits it, and the
    diagnostic records that the passage does not entail the prediction."""
    judge = HypothesisAwareJudge(TRIVIA_TEXT, required_in_hypothesis="public holiday")
    answer = answer_for(
        entities=ROSTER,
        doc_id=TRIVIA_DOC,
        label="miss",
        claim_text="The synthetic exchange observes a public holiday in February.",
    )
    outcome = _run(tmp_path, answer, judge)
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["faithfulness"] == 1.0
    assert outcome.diagnostics["prediction_relevance"] == 0.0
    # The diagnostic was computed: the judge was asked the prediction-derived question too.
    assert any(MISS_MARKER in h for _, h in judge.calls)


def test_prediction_relevance_is_none_when_the_judge_was_never_reached(
    tmp_path: pathlib.Path,
) -> None:
    judge = StubJudge()
    answer = answer_for(entities=ROSTER)
    answer["entity_predictions"][0]["claims"][0]["doc_id"] = "SYNTHDOC_NOT_IN_MANIFEST"
    outcome = _run(tmp_path, answer, judge)
    assert outcome.state == "participant_failure"
    assert outcome.diagnostics["prediction_relevance"] is None
    assert outcome.diagnostics["faithfulness"] is None
    assert judge.calls == []


# --- ordering and structure -----------------------------------------------------------------
def test_a_wrong_entity_citation_makes_that_claim_false_not_the_unit(
    tmp_path: pathlib.Path,
) -> None:
    """5.2.0: the wrong-entity citation is a per-claim falsehood (it was a whole-unit refusal).
    SYN-A cites SYN-B's document; SYN-B's own claim is untouched, so half the claims are false."""
    roster = ("SYN-A", "SYN-B")
    judge = StubJudge((SUPPORTING_TEXT,))
    unit = build_unit(
        tmp_path, entities=roster, labels={PRE_CUTOFF_DOC: {"entity_ids": ["SYN-B"]}}
    )
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    answer = answer_for(entities=roster, claim_text=ACCURATE_PROSE)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    ctx: dict[str, Any] = {"unit_dir": unit, "output_dir": out, "judge": judge}
    hydrate(ctx)
    ctx["_answer"] = answer
    _g3_domain_semantics(ctx)
    report = ctx["_claim_report"]
    assert [(v.entity_id, v.reasons) for v in report.verdicts] == [
        ("SYN-A", ("wrong_entity",)),
        ("SYN-B", ()),
    ]
    assert ctx["_wrong_entity_citations"] == 1
    assert ctx["_faithfulness_factor"] == 0.5


def test_the_denominator_is_the_claim_count() -> None:
    """Founding definition: fraction of CLAIMS supported. Two claims, one supported -> 0.5."""
    roster = EntityRoster(entity_ids=ROSTER)
    answer = answer_for(entities=ROSTER, claim_text=ACCURATE_PROSE)
    second = dict(answer["entity_predictions"][0]["claims"][0])
    second["claim"] = CONTRADICTING_PROSE
    answer["entity_predictions"][0]["claims"].append(second)
    aligned = align_predictions(
        answer, roster, target_type="classification", interval_level=0.90
    )
    report = evaluate_claims(
        aligned,
        lambda _doc_id: {"text": SUPPORTING_TEXT},
        _accuracy_judge(),
        target_type="classification",
        interval_scored=True,
        contradiction_bar=0.9,
    )
    assert report.claim_count == 2
    assert [v.status for v in report.verdicts] == ["neutral", "contradicted"]
    assert report.false_count == 1
    assert report.faithfulness == 0.5
    assert report.penalty_factor(1.0, entity_count=1) == 0.5
    assert report.penalty_factor(2.0, entity_count=1) == 0.25


@pytest.mark.parametrize(
    ("prose", "status"),
    [(ACCURATE_PROSE, "neutral"), (CONTRADICTING_PROSE, "contradicted")],
)
def test_a_claim_with_several_citations_is_read_on_its_most_contradicting_one(
    prose: str, status: str
) -> None:
    """Max over the claim's passages: one passage that contradicts the claim makes it false,
    whatever its other citations say (an unrelated passage is merely neutral)."""
    roster = EntityRoster(entity_ids=ROSTER)
    answer = answer_for(entities=ROSTER, claim_text=prose)
    claim = answer["entity_predictions"][0]["claims"][0]
    answer["entity_predictions"][0]["claims"] = [
        {
            "claim": prose,
            "citations": [
                {"doc_id": "TRIVIA", "span_start": 0, "span_end": len(TRIVIA_TEXT)},
                {
                    "doc_id": claim["doc_id"],
                    "span_start": 0,
                    "span_end": len(SUPPORTING_TEXT),
                },
            ],
        }
    ]
    aligned = align_predictions(
        answer, roster, target_type="classification", interval_level=0.90
    )
    texts = {"TRIVIA": TRIVIA_TEXT, claim["doc_id"]: SUPPORTING_TEXT}
    report = evaluate_claims(
        aligned,
        lambda doc_id: {"text": texts[doc_id]},
        _accuracy_judge(),
        target_type="classification",
        interval_scored=True,
        contradiction_bar=0.9,
    )
    assert report.verdicts[0].status == status
    assert report.verdicts[0].score == pytest.approx(
        0.99 if status == "contradicted" else 0.01
    )


def test_an_empty_claim_is_false_and_never_reaches_the_judge() -> None:
    roster = EntityRoster(entity_ids=ROSTER)
    answer = answer_for(entities=ROSTER, claim_text="   ")
    aligned = align_predictions(
        answer, roster, target_type="classification", interval_level=0.90
    )
    judge = StubJudge((SUPPORTING_TEXT,))
    report = evaluate_claims(
        aligned,
        lambda _doc_id: {"text": SUPPORTING_TEXT},
        judge,
        target_type="classification",
        interval_scored=True,
        contradiction_bar=0.9,
    )
    assert [v.status for v in report.verdicts] == ["malformed"]
    assert report.verdicts[0].reasons == ("malformed",)
    assert report.faithfulness == 0.0
    assert judge.calls == []


def test_no_claims_is_factor_one() -> None:
    """5.2.0: a report with no claims states nothing false, so the factor is 1. (Under
    the retired gate it was 0.0, "grounded nothing"; the penalty never rewards or charges
    silence -- alignment still requires claims[] on every entity, so this is the arithmetic
    edge, not a submission shape.)"""
    from qfbench2_track_analysis.scoring import ClaimReport

    assert ClaimReport(()).faithfulness == 1.0
    assert ClaimReport(()).penalty_factor(1.0, entity_count=1) == 1.0
    assert ClaimReport(()).penalty_factor(3.0, entity_count=1) == 1.0
