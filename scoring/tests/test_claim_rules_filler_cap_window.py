"""Scorer 5.2.2: three claim rules.

- **Content-free claims are false.** A claim that states no figure and, once the unit's own entity
  names are blanked, consists only of function words and evidence/meta words
  (`scoring.CONTENT_FREE_META_WORDS`), with nothing but function words left or a filler anchor
  (`scoring.CONTENT_FREE_FILLER_ANCHORS`) among them, asserts nothing a passage could support. It
  is false and is not put to the judge. Any other word makes it contentful, and so do finance
  words alone ("AAPL has no forecast.").
- **A citation over the 8,000-character cap is false**, whatever the claim states, a verbatim quote
  of the span included.
- **A long span is judged on its best-matching window.** The judge reads a window of about 500
  tokens. A cited span longer than that is judged on the window that shares the most claim words,
  the first window on a tie, so a contradiction deep in the passage is read.

Every text here is synthetic.
"""

from __future__ import annotations

from typing import Any

import pytest

from qfbench2_track_analysis.alignment import EntityRoster, align_predictions
from qfbench2_track_analysis.scoring import (
    CONTENT_FREE_FILLER_ANCHORS,
    CONTENT_FREE_META_WORDS,
    FIGURE_SPAN_CAP,
    best_matching_window,
    content_free,
    evaluate_claims,
)

from .synthetic import answer_for

ROSTER = ("SYN-A",)
NAMES = ("Synthetic Issuer A", "SYN-A")


# --- content-free ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "claim",
    [
        "Pre-cutoff evidence selected for the submitted prediction.",
        "Evidence for SYN-A from the cited pre-cutoff passage.",
        "Evidence for Synthetic Issuer A from the cited pre-cutoff passage.",
        "The cited document is relevant evidence.",
        "Fallback: nearest retrieved passage used as context.",
        "Synthetic Issuer A",
    ],
)
def test_a_meta_only_claim_is_content_free(claim: str) -> None:
    assert content_free(claim, figure_status="no_figures", names=NAMES)


@pytest.mark.parametrize(
    "claim",
    [
        "Guidance was cut.",
        "Rates rose.",
        "Hiring slowed.",
        "Job gains slowed.",
        "Energy prices fell.",
        "The Fed cut rates.",
        "Zent flagged going-concern doubt.",
        "Severance costs were recorded.",
        "Evidence shows demand weakened.",
    ],
)
def test_a_short_contentful_claim_is_not_content_free(claim: str) -> None:
    assert not content_free(claim, figure_status="no_figures", names=NAMES)


def test_a_claim_with_a_figure_is_never_content_free() -> None:
    assert not content_free(
        "Pre-cutoff evidence selected: 5.2", figure_status="anchored", names=NAMES
    )


# --- the gate ---------------------------------------------------------------------------------------
class WindowJudge:
    """Reads a `window`-character prefix of each premise and contradicts any premise that holds
    `marker`. Records every premise it is asked about."""

    def __init__(self, marker: str = "", *, window: int = 200) -> None:
        self.marker = marker
        self.window = window
        self.premises: list[str] = []

    def judged_premise(self, premise: str, hypothesis: str) -> str:
        return premise[: self.window]

    def contradiction(self, premise: str, hypothesis: str) -> float:
        self.premises.append(premise)
        return 0.99 if self.marker and self.marker in premise else 0.01

    def entail(self, premise: str, hypothesis: str) -> float:
        return 0.5


def _verdict(claim: str, span: str, judge: Any) -> Any:
    answer = answer_for(entities=ROSTER, claim_text=claim)
    answer["entity_predictions"][0]["claims"][0]["span_end"] = len(span)
    aligned: Any = align_predictions(
        answer,
        EntityRoster(entity_ids=ROSTER),
        target_type="classification",
        interval_level=0.9,
    )
    report = evaluate_claims(
        aligned,
        lambda _doc_id: {"text": span},
        judge,
        target_type="classification",
        interval_scored=False,
        contradiction_bar=0.9,
        entity_names=NAMES,
    )
    (verdict,) = report.verdicts
    return verdict


def test_a_content_free_claim_is_false_and_not_judged() -> None:
    judge = WindowJudge()
    verdict = _verdict(
        "Pre-cutoff evidence selected for SYN-A.", "Zent sales rose.", judge
    )
    assert verdict.status == "content_free" and verdict.reasons == ("content_free",)
    assert judge.premises == []


FILLER = "Zent note line. " * 20  # 320 characters, no claim word


def test_a_citation_over_the_cap_is_false() -> None:
    span = ("Zent sales rose. " * 600)[: FIGURE_SPAN_CAP + 1]
    verdict = _verdict(span[:40].strip(), span, WindowJudge())
    assert verdict.status == "over_cap" and verdict.reasons == ("over_cap",)


def test_a_citation_at_the_cap_is_not_over_it() -> None:
    span = ("Zent sales rose. " * 600)[:FIGURE_SPAN_CAP]
    verdict = _verdict(span[:40].strip(), span, WindowJudge())
    assert verdict.status == "neutral"


def test_a_contradiction_outside_the_first_window_is_caught() -> None:
    span = FILLER + "Zent quarterly sales fell sharply in March. " + FILLER
    judge = WindowJudge("fell sharply")
    verdict = _verdict("Zent quarterly sales rose sharply in March.", span, judge)
    assert verdict.status == "contradicted"
    (judged,) = judge.premises
    assert judged != span[:200] and "fell sharply" in judged


def test_a_tie_keeps_the_first_window() -> None:
    """No window shares a claim word: the first window is judged, as before 5.2.2."""
    span = FILLER + "Zent quarterly sales fell sharply in March. " + FILLER
    judge = WindowJudge("fell sharply")
    verdict = _verdict("Growth was strong.", span, judge)
    assert verdict.status == "neutral"
    assert judge.premises == [span[:200]]


def test_best_matching_window_returns_an_uncut_span_unchanged() -> None:
    assert best_matching_window(
        WindowJudge(), "short", "short", "claim", require_window=False
    ) == ("short")


@pytest.mark.parametrize(
    "claim",
    [
        pytest.param("\u5229\u7387\u5c06\u4e0a\u5347\u3002", id="non-latin-script"),
        pytest.param("SYN-A \u2191", id="arrow-only"),
        pytest.param("Synthetic Issuer A failed.", id="entity-failed"),
        pytest.param("SYN-A at 1.25", id="own-value-only"),
    ],
)
def test_latent_false_positives_are_contentful(claim: str) -> None:
    assert not content_free(claim, figure_status="no_figures", names=NAMES)


@pytest.mark.parametrize(
    "claim",
    [
        "Evidence for SYN-A from the cited pre-cutoff passage.",
        "Pre-cutoff evidence selected for SYN-A.",
        "The cited document is relevant evidence.",
        "No model-entailed quote could be grounded for SYN-A; top-retrieved passage used.",
        "This pre-cutoff passage is available context only. Model inference failed; it does not"
        " establish the placeholder forecast.",
        "Pre-cutoff evidence selected for the submitted prediction.",
        "fallback",
        "Synthetic Issuer A",
        "Model inference failed",
        "No model-entailed quote could be grounded for SYN-A; citing the nearest top-retrieved"
        " passage as available evidence.",
    ],
)
def test_the_filler_templates_are_still_content_free(claim: str) -> None:
    assert content_free(claim, figure_status="no_figures", names=NAMES)


# --- organizer review of 5.2.2: finance words are not filler on their own ---------------------------
REVIEW_NAMES = ("Apple Inc.", "AAPL")


@pytest.mark.parametrize(
    "claim",
    [
        "AAPL has no forecast.",
        "No forecast available.",
        "There will be no support.",
        "No quotes were submitted.",
        "No forecasts were submitted.",
        "Quotes are available.",
        "AAPL has no quote.",
        "Support is available for AAPL.",
        "The forecast was submitted.",
        "No supports were available.",
        "There is support.",
        "Apple Inc. has no forecast available.",
    ],
)
def test_finance_meta_words_without_a_filler_anchor_are_contentful(claim: str) -> None:
    assert not content_free(claim, figure_status="no_figures", names=REVIEW_NAMES)


def test_the_filler_anchors_are_the_reviewed_list() -> None:
    assert CONTENT_FREE_FILLER_ANCHORS == frozenset(
        """evidence passage passages excerpt pre-cutoff cutoff cite cited citing retrieved
        top-retrieved nearest placeholder fallback inference context contextual wording
        document documents source sources model-entailed""".split()
    )
    assert CONTENT_FREE_FILLER_ANCHORS <= CONTENT_FREE_META_WORDS


@pytest.mark.parametrize("anchor", sorted(CONTENT_FREE_FILLER_ANCHORS))
def test_each_filler_anchor_makes_a_meta_only_claim_content_free(anchor: str) -> None:
    """The same finance words are contentful without the anchor and content-free with it."""
    assert not content_free(
        "No forecast for SYN-A.", figure_status="no_figures", names=NAMES
    )
    assert content_free(
        f"No forecast for SYN-A; {anchor} only.",
        figure_status="no_figures",
        names=NAMES,
    )
    assert content_free(
        f"{anchor.capitalize()}.", figure_status="no_figures", names=NAMES
    )


@pytest.mark.parametrize("anchor", sorted(CONTENT_FREE_FILLER_ANCHORS))
def test_a_filler_anchor_does_not_flag_a_claim_with_a_content_word(anchor: str) -> None:
    assert not content_free(
        f"Demand weakened, per the {anchor}.", figure_status="no_figures", names=NAMES
    )
