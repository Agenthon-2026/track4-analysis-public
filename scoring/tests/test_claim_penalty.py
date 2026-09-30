"""Scorer 5.2.0: faithfulness is a per-claim PENALTY, not an 80% admission gate.

A claim is FALSE when it cites a document not labelled for its entity, cites an out-of-range
slice, is malformed, is ``unanchored`` under the unchanged any-figure backstop, or when the
judge's three-way P(contradiction) for (judged passage, claim) exceeds ``contradiction_bar``.
Every other claim is NEUTRAL. The unit scores ``composite * factor ** penalty_k`` with the soft
floor ``factor = 1 - F/(F + min(T, 3E))`` (F false, T the other claims, E the roster count);
no claims, or no false claims, means factor 1 (see test_soft_floor.py). Only structural errors refuse a unit.

The planted classes (a)-(f) are the synthetic analogues of the ones the penalty probe measured on
real units with the real ensemble: (a) content-free,
(b) false and figure-free, (c) true with figures, (d) true with a figure altered, (e) one real and
one fabricated figure, (f) true with a derived figure. The stub judge here plays the ensemble's
part for (b) only (it contradicts exactly the claims it is told to); everything else is decided
by exact code, which is what these tests pin. (e) and (f) are NOT false under the penalty alone: the
any-figure backstop anchors them on their real figure, and the probe found the real ensemble
contradicts few of them. Catching them is the every-figure rule, not this one.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any

import pytest

from qfbench2_track_analysis.codes import T4OrganizerFault, T4Reason
from qfbench2_track_analysis.judge_factory import (
    JudgeProvenance,
    WindowedEnsembleNLIJudge,
    build_smoke_judge,
)
from qfbench2_track_analysis.scoring import (
    DEFAULT_CONTRADICTION_BAR,
    DEFAULT_PENALTY_K,
    FAITHFULNESS_RULE,
    ClaimReport,
    ScoringParams,
    UnitOutcome,
    _apply_faithfulness_penalty,
    judged_contradiction,
    score_unit,
)

from .synthetic import PRE_CUTOFF_DOC, _doc, answer_for, build_unit

SPAN = (
    "Synthetic Issuer A reported revenue of $5.2 billion in the quarter, up 8% from the prior "
    "year, and operating margin of 14.5%."
)

PLANTED: dict[str, list[str]] = {
    "a": [
        "Pre-cutoff evidence selected for SYN-A.",
        "The cited document is relevant evidence.",
    ],
    "b": ["Synthetic Issuer A reported lower revenue than a year earlier."],
    "c": ["Revenue was $5.2 billion, up 8% from the prior year."],
    "d": ["Revenue was $5.9 billion, up 11% from the prior year."],
    "e": ["Revenue was $5.2 billion and operating margin was 17.0%."],
    "f": [
        "Revenue was $5.2 billion, up 8%, so prior-year revenue was about $4.8 billion."
    ],
}
#: What the penalty must decide for each class, with the reasons it must give.
EXPECTED: dict[str, tuple[str, ...]] = {
    "a": ("content_free",),
    "b": ("contradicted",),
    "c": (),
    "d": ("unanchored",),
    "e": ("unanchored",),
    "f": ("unanchored",),
}


class TextJudge:
    """Contradicts exactly the hypotheses it is told to (P = `high`), everything else at `low`.

    Entailment is irrelevant to 5.2.0 and is answered 0.99 throughout, so nothing in these tests
    can pass because a claim was "supported".
    """

    def __init__(
        self,
        contradicted: tuple[str, ...] = (),
        *,
        high: float = 0.97,
        low: float = 0.02,
    ) -> None:
        self.contradicted = set(contradicted)
        self.high = high
        self.low = low
        self.calls: list[tuple[str, str]] = []

    def entail(self, premise: str, hypothesis: str) -> float:
        return 0.99

    def contradiction(self, premise: str, hypothesis: str) -> float:
        self.calls.append((premise, hypothesis))
        return self.high if hypothesis in self.contradicted else self.low


def _unit(
    tmp_path: pathlib.Path,
    entities: tuple[str, ...] = ("SYN-A",),
    *,
    extra_params: str = "",
    labels: dict[str, dict[str, Any]] | None = None,
) -> pathlib.Path:
    unit = build_unit(
        tmp_path,
        entities=entities,
        with_outcome=True,
        docs={PRE_CUTOFF_DOC: _doc(PRE_CUTOFF_DOC, "2026-02-01", SPAN)},
        labels=labels,
    )
    if extra_params:
        card = unit / "card.toml"
        text = card.read_text(encoding="utf-8")
        anchor = "tau_citation           = 0.5\n"
        assert anchor in text
        card.write_text(text.replace(anchor, anchor + extra_params), encoding="utf-8")
    return unit


def _answer(claims_by_entity: dict[str, list[str]]) -> dict[str, Any]:
    answer = answer_for(entities=tuple(claims_by_entity))
    for row in answer["entity_predictions"]:
        first = row["claims"][0]
        row["claims"] = [
            {**first, "claim": text, "span_end": len(SPAN)}
            for text in claims_by_entity[row["entity_id"]]
        ]
    return answer


def _score(
    tmp_path: pathlib.Path,
    unit: pathlib.Path,
    answer: dict[str, Any],
    judge: object,
    provenance: JudgeProvenance | None = None,
) -> UnitOutcome:
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    if provenance is None:
        _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out}, judge=judge, judge_provenance=provenance
    )


# --- the planted classes -------------------------------------------------------------------------
def test_the_planted_classes_are_penalised_exactly_as_step_one_rules(
    tmp_path: pathlib.Path,
) -> None:
    claims = [text for cls in "abcdef" for text in PLANTED[cls]]
    judge = TextJudge(contradicted=tuple(PLANTED["b"]))
    outcome = _score(tmp_path, _unit(tmp_path), _answer({"SYN-A": claims}), judge)

    # Never a refusal: 6 of 7 claims false (5.2.2: the two content-free claims too) leaves
    # 1 - 6/(6 + 1) = 1/7 of the score at k = 1.
    assert outcome.state == "participant_success"
    assert outcome.failure_code is None
    got = {c["claim"]: tuple(c["reasons"]) for c in outcome.diagnostics["false_claims"]}
    for cls in "abcdef":
        for text in PLANTED[cls]:
            assert got.get(text, ()) == EXPECTED[cls], (cls, text)
    assert outcome.diagnostics["claim_count"] == 7
    assert outcome.diagnostics["false_claim_count"] == 6
    assert outcome.diagnostics["faithfulness_rule"] == FAITHFULNESS_RULE
    factor = outcome.diagnostics["faithfulness_factor"]
    assert factor == pytest.approx(1 / 7)
    before = outcome.diagnostics["composite_before_penalty"]
    assert before > 0.0
    assert outcome.score == pytest.approx(before * 1 / 7)
    # Every judged claim was put to the judge as (the cited passage, the claim's own text); a
    # content-free claim is not put to it (5.2.2).
    judged = [t for cls in "bc" for t in PLANTED[cls]]
    assert sorted(h for _, h in judge.calls) == sorted(judged)
    assert {p for p, _ in judge.calls} == {SPAN}


def test_content_free_claims_are_false(tmp_path: pathlib.Path) -> None:
    """5.2.2: a content-free claim asserts nothing a passage could
    support, so it is false and is not put to the judge. A unit of nothing else scores 0."""
    judge = TextJudge()
    outcome = _score(tmp_path, _unit(tmp_path), _answer({"SYN-A": PLANTED["a"]}), judge)
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["false_claim_count"] == 2
    assert outcome.diagnostics["content_free_claim_count"] == 2
    assert judge.calls == []
    assert outcome.score == 0.0


# --- wrong-entity citations: partial credit, not a cliff -----------------------------------------
def test_one_wrong_entity_citation_costs_one_claim_not_the_unit(
    tmp_path: pathlib.Path,
) -> None:
    """SYN-A and SYN-B, two claims each. The document is labelled for SYN-B only, so SYN-A's two
    claims are false (wrong entity) and SYN-B's two are neutral: factor 0.5, not 0."""
    labels = {PRE_CUTOFF_DOC: {"entity_ids": ["SYN-B"]}}
    unit = _unit(tmp_path, ("SYN-A", "SYN-B"), labels=labels)
    true_claims = PLANTED["c"] + ["Synthetic Issuer A reported revenue."]
    answer = _answer({"SYN-A": true_claims, "SYN-B": true_claims})
    outcome = _score(tmp_path, unit, answer, TextJudge())
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["wrong_entity_claim_count"] == 2
    assert [c["entity_id"] for c in outcome.diagnostics["false_claims"]] == [
        "SYN-A",
        "SYN-A",
    ]
    assert outcome.diagnostics["faithfulness_factor"] == 0.5
    assert outcome.score == pytest.approx(
        0.5 * outcome.diagnostics["composite_before_penalty"]
    )


# --- the penalty's two parameters ----------------------------------------------------------------
@pytest.mark.parametrize(("k", "factor"), [(1, 0.5), (2, 0.25), (0.5, math.sqrt(0.5))])
def test_penalty_k_is_the_exponent_of_the_factor(k: float, factor: float) -> None:
    """The arithmetic of `factor ** k` (one of two claims false, within the cap, so the factor is
    1/2); in 5.2.0 k is fixed at 1 (see below)."""
    from qfbench2_track_analysis.scoring import ClaimVerdict

    report = ClaimReport(
        (
            ClaimVerdict("E", "a", "neutral", 0.0),
            ClaimVerdict("E", "b", "unanchored", 0.0),
        )
    )
    assert report.penalty_factor(k, entity_count=1) == pytest.approx(factor)


def test_a_unit_scores_with_k_one(tmp_path: pathlib.Path) -> None:
    claims = PLANTED["c"] + PLANTED["d"]  # one neutral, one unanchored
    outcome = _score(tmp_path, _unit(tmp_path), _answer({"SYN-A": claims}), TextJudge())
    assert outcome.diagnostics["faithfulness_factor"] == pytest.approx(0.5)
    assert outcome.score == pytest.approx(
        0.5 * outcome.diagnostics["composite_before_penalty"]
    )


@pytest.mark.parametrize(
    ("p_contra", "is_false"),
    [(0.9, False), (0.9000001, True), (0.8999999, False), (0.97, True)],
)
def test_the_contradiction_bar_is_strict(
    tmp_path: pathlib.Path, p_contra: float, is_false: bool
) -> None:
    """False only ABOVE the bar (default 0.9): a probability equal to the bar is neutral."""
    claim = PLANTED["b"][0]
    judge = TextJudge(contradicted=(claim,), high=p_contra)
    outcome = _score(tmp_path, _unit(tmp_path), _answer({"SYN-A": [claim]}), judge)
    assert outcome.diagnostics["false_claim_count"] == int(is_false)
    assert outcome.diagnostics["faithfulness_factor"] == (0.0 if is_false else 1.0)


@pytest.mark.parametrize(
    "line",
    [
        "penalty_k              = 2\n",
        "contradiction_bar      = 0.7\n",
        "penalty_k              = 1\n",
    ],
)
def test_a_card_cannot_set_the_fixed_constants(
    tmp_path: pathlib.Path, line: str
) -> None:
    """k = 1 and bar = 0.9 are scorer constants in 5.2.0. A card naming either
    (even with the default value) is refused, never read."""
    unit = _unit(tmp_path, extra_params=line)
    with pytest.raises(T4OrganizerFault, match="fixes penalty_k"):
        _score(tmp_path, unit, _answer({"SYN-A": PLANTED["c"]}), TextJudge())


def test_the_defaults_are_the_measured_recommendation() -> None:
    assert (DEFAULT_PENALTY_K, DEFAULT_CONTRADICTION_BAR) == (1.0, 0.9)
    params = ScoringParams.from_sources(trusted=None, card_params=None)
    assert (params.penalty_k, params.contradiction_bar) == (1.0, 0.9)


_BASE = {
    "target_type": "classification",
    "interval_level": 0.9,
    "faithfulness_threshold": 0.8,
    "tau_citation": 0.5,
    "composite_weights": (0.7, 0.3),
}


@pytest.mark.parametrize(
    "override",
    [
        {"penalty_k": 0},
        {"penalty_k": 1.0},
        {"penalty_k": True},
        {"contradiction_bar": 0.9},
        {"contradiction_bar": float("nan")},
    ],
)
def test_a_plan_cannot_set_the_fixed_constants(override: dict[str, Any]) -> None:
    with pytest.raises(T4OrganizerFault, match="fixes penalty_k"):
        ScoringParams.from_sources(trusted={**_BASE, **override}, card_params=None)
    with pytest.raises(T4OrganizerFault, match="fixes penalty_k"):
        ScoringParams.from_sources(trusted=_BASE, card_params=override)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"penalty_k": 0}, "penalty_k must be > 0"),
        ({"penalty_k": float("inf")}, "penalty_k must be finite"),
        ({"contradiction_bar": 1.0}, r"contradiction_bar must be in \(0, 1\)"),
    ],
)
def test_the_constants_are_still_validated_where_constructed(
    override: dict[str, Any], message: str
) -> None:
    fields = {**_BASE, "composite_weights": (0.7, 0.3), **override}
    with pytest.raises(T4OrganizerFault, match=message):
        ScoringParams(**fields)


# --- the retired threshold: mapped explicitly, never read silently -------------------------------
def test_the_legacy_threshold_maps_to_the_penalty_and_says_so(
    tmp_path: pathlib.Path,
) -> None:
    """Every shipped card and C1 plan carries faithfulness_threshold = 0.80; 5.2.0 maps exactly
    that to the penalty with its defaults and records the rule it applied."""
    unit = _unit(tmp_path)
    assert "faithfulness_threshold = 0.8" in (unit / "card.toml").read_text()
    outcome = _score(tmp_path, unit, _answer({"SYN-A": PLANTED["c"]}), TextJudge())
    assert outcome.state == "participant_success"
    assert (
        outcome.diagnostics["faithfulness_rule"] == "claim_penalty_soft_floor_3e/5.2.0"
    )


@pytest.mark.parametrize("threshold", [0.5, 0.75, 0.9, 1.0])
def test_any_other_legacy_threshold_is_refused_not_ignored(threshold: float) -> None:
    with pytest.raises(T4OrganizerFault, match="retired the 80% admission gate"):
        ScoringParams.from_sources(
            trusted={**_BASE, "faithfulness_threshold": threshold}, card_params=None
        )
    with pytest.raises(T4OrganizerFault, match="retired the 80% admission gate"):
        ScoringParams.from_sources(
            trusted=None, card_params={"faithfulness_threshold": threshold}
        )


# --- no claims: factor 1 -------------------------------------------------------------------------
def test_no_claims_is_factor_one_through_the_composite() -> None:
    report = ClaimReport(())
    factor = report.penalty_factor(2.0, entity_count=1)
    assert factor == 1.0
    parts = _apply_faithfulness_penalty(
        {"composite": 0.42, "predictive_quality": 0.6}, {"_faithfulness_factor": factor}
    )
    assert parts["composite"] == parts["composite_before_penalty"] == 0.42
    assert parts["faithfulness_factor"] == 1.0


def test_a_missing_factor_is_an_organizer_fault_not_a_free_pass() -> None:
    with pytest.raises(T4OrganizerFault, match="faithfulness factor"):
        _apply_faithfulness_penalty({"composite": 0.42}, {})


# --- structural refusals are unchanged -----------------------------------------------------------
def _structural(case: str) -> tuple[dict[str, Any], T4Reason]:
    answer = _answer({"SYN-A": PLANTED["c"], "SYN-B": PLANTED["c"]})
    if case == "schema":
        answer["entity_predictions"][0]["claims"][0]["span_start"] = "zero"
        return answer, T4Reason.SCHEMA_INVALID
    if case == "missing_entity":
        answer["entity_predictions"].pop()
        return answer, T4Reason.ENTITY_MISSING
    if case == "unresolved_citation":
        answer["entity_predictions"][0]["claims"][0]["doc_id"] = "SYNTHDOC_NOT_DECLARED"
        return answer, T4Reason.CITATION_UNRESOLVED
    if case == "no_claims":
        answer["entity_predictions"][1]["claims"] = []
        return answer, T4Reason.SCHEMA_INVALID
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case", ["schema", "missing_entity", "unresolved_citation", "no_claims"]
)
def test_structural_errors_still_refuse_the_unit(
    tmp_path: pathlib.Path, case: str
) -> None:
    answer, reason = _structural(case)
    unit = _unit(tmp_path, ("SYN-A", "SYN-B"))
    outcome = _score(tmp_path, unit, answer, TextJudge())
    assert outcome.state == "participant_failure"
    assert outcome.score == 0.0
    assert outcome.diagnostics["reason"] == reason.value


# --- the judge side: contradiction is a separate question, entail is untouched -------------------
class _Member:
    def __init__(self, value: float) -> None:
        self.value = value

    def entail(self, premise: str, hypothesis: str) -> float:
        return 0.5

    def contradiction(self, premise: str, hypothesis: str) -> float:
        return self.value


def test_the_ensemble_contradiction_is_the_member_mean() -> None:
    ensemble = WindowedEnsembleNLIJudge([_Member(0.95), _Member(0.80)])
    assert ensemble.contradiction("p", "h") == pytest.approx(0.875)
    assert ensemble.entail("p", "h") == 0.5


def test_an_ensemble_member_without_contradiction_is_an_organizer_fault() -> None:
    class EntailOnly:
        def entail(self, premise: str, hypothesis: str) -> float:
            return 0.5

    ensemble = WindowedEnsembleNLIJudge([_Member(0.95), EntailOnly()])
    with pytest.raises(T4OrganizerFault, match="no three-way contradiction"):
        ensemble.contradiction("p", "h")


def test_a_rankable_judge_without_contradiction_is_an_organizer_fault() -> None:
    class EntailOnly:
        def entail(self, premise: str, hypothesis: str) -> float:
            return 0.5

    with pytest.raises(T4OrganizerFault, match="no three-way contradiction"):
        judged_contradiction(EntailOnly(), "p", "h", require_window=True)
    # A non-rankable stand-in with no such method never finds a contradiction.
    assert judged_contradiction(EntailOnly(), "p", "h") == 0.0
    smoke, _ = build_smoke_judge()
    assert judged_contradiction(smoke, "p", "h") == 0.0


def test_a_contradiction_outside_the_unit_interval_is_an_organizer_fault() -> None:
    with pytest.raises(T4OrganizerFault, match=r"outside \[0, 1\]"):
        judged_contradiction(_Member(1.5), "p", "h")
