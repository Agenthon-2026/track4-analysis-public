"""Two guards on the passage a claim cites (scorer 5.1.1).

The offsets guard. A citation whose `span_start`/`span_end` do not name a real slice of the
document -- a negative start, an end past the document, or a start at or after the end -- names
no passage. It contributes nothing to the claim, exactly as a citation of an unknown document
does, and the gate records how many there were. Before 5.1.1 Python slicing clamped such offsets
silently: `[0, N)` with N past the end read the whole document, and a negative start read the
document's tail.

The judged-window guard. The NLI judge reads at most its tokenizer's window (512 tokens, premise
and claim together, the premise truncated first). The numeric backstop used to read the whole
span, so a figure the judge never saw could anchor a claim. Both now read the one text
`judged_passage` returns, so they cannot disagree about what the passage says.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from qfbench2_track_analysis.alignment import EntityRoster, align_predictions
from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import (
    evaluate_claims,
    judged_passage,
    score_unit,
)

from .synthetic import PRE_CUTOFF_DOC, StubJudge, answer_for, build_unit

ROSTER = ("SYN-A",)
TEXT = "Synthetic Issuer A reported revenue of $5.2 billion, up 8% year over year."
CLAIM = "Revenue was $5.2 billion"


def _aligned(citations: list[tuple[int, int]], claim: str = CLAIM) -> Any:
    answer = answer_for(entities=ROSTER, claim_text=claim)
    first = answer["entity_predictions"][0]["claims"][0]
    # One claim carrying every citation (the explicit `citations[]` shape).
    answer["entity_predictions"][0]["claims"] = [
        {
            "claim": claim,
            "citations": [
                {"doc_id": first["doc_id"], "span_start": start, "span_end": end}
                for start, end in citations
            ],
        }
    ]
    return align_predictions(
        answer,
        EntityRoster(entity_ids=ROSTER),
        target_type="classification",
        interval_level=0.90,
    )


def _lookup(_doc_id: str) -> dict[str, str]:
    return {"text": TEXT}


class WindowJudge(StubJudge):
    """A stub judge that, like the production ensemble, reads only a window of the premise.

    `judged_premise` is the hook `judged_passage` asks: here the window is the first `window`
    characters. `entail` and `contradiction` record what they were handed, so a test can see
    the judge's input.
    """

    def __init__(self, window: int, score: float = 0.99) -> None:
        super().__init__((), score)
        self.window = window

    def judged_premise(self, premise: str, hypothesis: str) -> str:
        return premise[: self.window]

    def entail(self, premise: str, hypothesis: str) -> float:
        self.calls.append((premise, hypothesis))
        return self.score


# --- Offsets in range -----------------------------------------------------------------------------
def test_a_span_past_the_end_of_the_document_names_no_passage() -> None:
    judge = StubJudge((TEXT,))
    report = evaluate_claims(
        _aligned([(0, len(TEXT) + 10)]),
        _lookup,
        judge,
        target_type="classification",
        interval_scored=True,
        contradiction_bar=0.9,
    )
    # No passage: the claim's figure is anchored nowhere, and the judge is never asked.
    assert [v.status for v in report.verdicts] == ["unanchored"]
    assert judge.calls == []
    assert report.out_of_range_citation_count == 1


def test_a_negative_start_names_no_passage() -> None:
    judge = StubJudge((TEXT[-40:],))
    report = evaluate_claims(
        _aligned([(-40, len(TEXT))]),
        _lookup,
        judge,
        target_type="classification",
        interval_scored=True,
        contradiction_bar=0.9,
    )
    assert [v.status for v in report.verdicts] == ["unanchored"]
    assert judge.calls == []
    assert report.out_of_range_citation_count == 1


def test_a_start_at_or_after_the_end_names_no_passage() -> None:
    judge = StubJudge((TEXT,))
    report = evaluate_claims(
        _aligned([(10, 10), (20, 5)]),
        _lookup,
        judge,
        target_type="classification",
        interval_scored=True,
        contradiction_bar=0.9,
    )
    assert [v.status for v in report.verdicts] == ["unanchored"]
    assert judge.calls == []
    assert report.out_of_range_citation_count == 2


def test_an_in_range_citation_beside_a_bad_one_still_counts() -> None:
    """The guard drops the bad citation, not the claim: the good passage is still judged. From
    5.2.0 the bad citation makes the claim false (`out_of_range`) all the same."""
    judge = StubJudge((TEXT,))
    report = evaluate_claims(
        _aligned([(0, len(TEXT) + 1), (0, len(TEXT))]),
        _lookup,
        judge,
        target_type="classification",
        interval_scored=True,
        contradiction_bar=0.9,
    )
    assert [v.status for v in report.verdicts] == ["neutral"]
    assert judge.calls == [(TEXT, CLAIM)]
    assert report.out_of_range_citation_count == 1
    assert report.verdicts[0].reasons == ("out_of_range",)
    assert report.false_count == 1


def test_the_out_of_range_count_reaches_the_review_queue(
    tmp_path: pathlib.Path,
) -> None:
    document = {
        "doc_id": PRE_CUTOFF_DOC,
        "doc_date": "2026-02-01",
        "source": "synthetic",
        "spans": [{"span_id": 0, "start": 0, "end": len(TEXT), "text": TEXT}],
        "text": TEXT,
    }
    unit = build_unit(
        tmp_path, entities=ROSTER, with_outcome=True, docs={PRE_CUTOFF_DOC: document}
    )
    answer = answer_for(entities=ROSTER, claim_text=CLAIM)
    answer["entity_predictions"][0]["claims"][0]["span_end"] = len(TEXT) + 500
    out = tmp_path / "res"
    out.mkdir()
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    outcome = score_unit(
        {"unit_dir": unit, "output_dir": out},
        judge=StubJudge((TEXT,)),
        judge_provenance=provenance,
    )
    # 5.2.0: a penalty, never a refusal. The one claim is false twice over (out of range, and
    # its figure is then anchored nowhere), so the factor is 0 and so is the score.
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["out_of_range_citation_count"] == 1
    assert outcome.diagnostics["out_of_range_claim_count"] == 1
    assert outcome.diagnostics["unanchored_claim_count"] == 1
    assert outcome.diagnostics["false_claims"] == [
        {
            "entity_id": "SYN-A",
            "claim": CLAIM,
            "reasons": ["out_of_range", "unanchored"],
        }
    ]
    assert outcome.score == 0.0


# --- The figure check reads what the judge reads --------------------------------------------------
def test_a_figure_outside_the_judged_window_anchors_on_the_whole_span_and_the_judge_reads_the_cut() -> (
    None
):
    # The window ends before "$5.2 billion": the judge never sees the figure. From 5.2.0 (the
    # every-figure rule) figures are checked against the WHOLE cited span, so the claim is anchored;
    # the judge still reads only its window.
    window = TEXT.index("$5.2")
    judge = WindowJudge(window)
    report = evaluate_claims(
        _aligned([(0, len(TEXT))]),
        _lookup,
        judge,
        target_type="classification",
        interval_scored=True,
        contradiction_bar=0.9,
    )
    assert [v.status for v in report.verdicts] == ["neutral"]
    assert judge.calls == [(TEXT[:window], CLAIM)]
    assert report.window_cut_citation_count == 1


def test_a_figure_inside_the_judged_window_still_anchors_and_the_judge_reads_the_cut() -> (
    None
):
    window = TEXT.index(" billion") + len(" billion")
    judge = WindowJudge(window)
    report = evaluate_claims(
        _aligned([(0, len(TEXT))]),
        _lookup,
        judge,
        target_type="classification",
        interval_scored=True,
        contradiction_bar=0.9,
    )
    assert [v.status for v in report.verdicts] == ["neutral"]
    # The judge's premise is exactly the text the figure check read.
    assert judge.calls == [(TEXT[:window], CLAIM)]
    assert report.window_cut_citation_count == 1


def test_a_passage_inside_the_window_reaches_the_judge_unchanged() -> None:
    judge = WindowJudge(10_000)
    report = evaluate_claims(
        _aligned([(0, len(TEXT))]),
        _lookup,
        judge,
        target_type="classification",
        interval_scored=True,
        contradiction_bar=0.9,
    )
    assert [v.status for v in report.verdicts] == ["neutral"]
    assert judge.calls == [(TEXT, CLAIM)]
    assert report.window_cut_citation_count == 0


def test_judged_passage_is_the_identity_for_a_judge_without_a_window() -> None:
    """The smoke judge and the test stubs read the whole premise; nothing is cut for them."""
    assert judged_passage(StubJudge(()), TEXT, CLAIM) == TEXT
    smoke, _ = build_smoke_judge()
    assert judged_passage(smoke, TEXT, CLAIM) == TEXT
