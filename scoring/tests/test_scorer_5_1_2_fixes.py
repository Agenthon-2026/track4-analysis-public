"""Scorer 5.1.2: the review fixes to the figure reader and the citation offsets.

1. An unknown scale suffix ("12kn") raised KeyError from `figures`, and one claim's prose aborted
   the whole evaluation. An unknown suffix is now no scale.
2. The month pattern had no word boundary, so "Decreased", "Declined" and "Margin" read as months
   and "Decreased 12.5%" lost its integer part: an honest verbatim claim came out unanchored.
3. The year pattern blanked every bare 4-digit number from 1900 to 2099, so "$2030 million" in a
   span was deleted (a false refusal) and "$2045 million" in a claim became no figure at all
   (silent credit). A currency sign, a decimal part or a scale word now says "amount".
5. The offsets guard dropped a citation whose offsets were whole-number floats (``120.0``), which the published
   schema admits as integers. They are now read as the integer; any other non-integer offset is
   refused by name.
"""

from __future__ import annotations

import json
import pathlib
from decimal import Decimal
from typing import Any

import pytest

from qfbench2_track_analysis.alignment import EntityRoster, align_predictions
from qfbench2_track_analysis.codes import T4ParticipantFailure, T4Reason
from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.numeric import claim_number_status, figures
from qfbench2_track_analysis.scoring import UnitOutcome, score_unit

from .synthetic import PRE_CUTOFF_DOC, StubJudge, answer_for, build_unit

ROSTER = ("SYN-A",)


# --- 1. an unknown suffix is no scale, never an exception ---------------------------------------
@pytest.mark.parametrize(
    ("text", "value"), [("growth of 12kn", "12"), ("revenue 5Kn", "5")]
)
def test_an_unknown_suffix_reads_as_no_scale(text: str, value: str) -> None:
    assert figures(text) == (Decimal(value),)


def test_a_claim_with_an_unknown_suffix_is_classified_not_raised() -> None:
    assert (
        claim_number_status("Output rose 12kn", ["output rose 12 units"]) == "anchored"
    )


# --- 2. a month is a month name, not a capitalised word starting with one -----------------------
@pytest.mark.parametrize(
    ("claim", "span"),
    [
        (
            "Decreased 12.5% year over year",
            "net revenue decreased 12.5% year over year",
        ),
        ("Declined 12.5% in the quarter", "deposits declined 12.5% in the quarter"),
        ("Margin 12.5% for the quarter", "operating margin was 12.5% for the quarter"),
        ("Market share 31.5%", "a market share of 31.5%"),
    ],
)
def test_a_capitalised_word_is_not_a_month(claim: str, span: str) -> None:
    assert claim_number_status(claim, [span]) == "anchored"


def test_no_integer_part_is_lost_after_a_month_like_word() -> None:
    assert figures("Declined 12% to 3.4") == (Decimal("12"), Decimal("3.4"))
    assert figures("Novel 9 and Separately 4") == (Decimal("9"), Decimal("4"))


@pytest.mark.parametrize(
    "text",
    [
        "Sept. 30, 2024",
        "Mar. 5",
        "Dec 31",
        "May 6-7, 2025",
        "31 March 2025",
        "6-7 May",
        "on September 17",
    ],
)
def test_month_dates_are_still_not_figures(text: str) -> None:
    assert figures(text) == ()


# --- 3. a 4-digit amount is not a year when the context says amount -----------------------------
@pytest.mark.parametrize(
    ("claim", "span"),
    [
        ("Revenue of $2,030 million", "revenue of $2030 million"),
        ("Revenue of $2030 million", "revenue of $2,030 million"),
        ("Revenue of $1.95 billion", "revenue of $1950 million"),
        ("Revenue of $1950 million", "revenue of 1950 million dollars"),
        ("An index of 1987.5", "the index closed at 1987.5"),
    ],
)
def test_a_four_digit_amount_anchors(claim: str, span: str) -> None:
    assert claim_number_status(claim, [span]) == "anchored"


def test_a_fabricated_four_digit_amount_is_a_figure_the_span_must_carry() -> None:
    """Before 5.1.2 "$2045 million" was blanked as a year: no figure, the judge alone decided."""
    assert (
        claim_number_status("Revenue of $2045 million", ["revenue of $2030 million"])
        == "unanchored"
    )


@pytest.mark.parametrize(
    "text",
    [
        "in 2024",
        "fiscal 2030 outlook",
        "FY2024",
        "Q1 2026",
        "2026Q1",
        "the 2019 annual report",
        "as of 2025, revenue",
    ],
)
def test_years_are_still_not_figures(text: str) -> None:
    assert figures(text) == ()


# --- 5. Offsets: a whole-number float is its integer; nothing else is an offset --------------
TEXT = "Synthetic Issuer A reported revenue of $5.2 billion, up 8% year over year."
CLAIM = "Revenue was $5.2 billion"


def _score(tmp_path: pathlib.Path, start: Any, end: Any) -> UnitOutcome:
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
    answer["entity_predictions"][0]["claims"][0].update(span_start=start, span_end=end)
    out = tmp_path / "res"
    out.mkdir()
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out},
        judge=StubJudge((TEXT,)),
        judge_provenance=provenance,
    )


def test_whole_number_float_offsets_score_as_their_integers(
    tmp_path: pathlib.Path,
) -> None:
    as_int = _score(tmp_path / "int", 0, len(TEXT))
    as_float = _score(tmp_path / "float", 0.0, float(len(TEXT)))
    assert as_int.state == as_float.state == "participant_success"
    assert as_float.score == as_int.score
    assert as_float.diagnostics["out_of_range_citation_count"] == 0
    assert as_float.diagnostics["claim_count"] == 1
    assert as_float.diagnostics["false_claim_count"] == 0


@pytest.mark.parametrize(("start", "end"), [(0.5, len(TEXT)), (False, True)])
def test_non_integer_offsets_are_refused_not_dropped(
    tmp_path: pathlib.Path, start: Any, end: Any
) -> None:
    outcome = _score(tmp_path, start, end)
    assert outcome.state == "participant_failure"
    # The schema gate refuses them first; the claim gate never sees a dropped citation.
    assert outcome.diagnostics["reason"] == T4Reason.SCHEMA_INVALID.value
    assert outcome.diagnostics.get("out_of_range_citation_count") is None


def _nested(start: Any, end: Any) -> dict[str, Any]:
    """The explicit `citations[]` shape, which the schema's offset types do not reach."""
    answer = answer_for(entities=ROSTER, claim_text=CLAIM)
    first = answer["entity_predictions"][0]["claims"][0]
    answer["entity_predictions"][0]["claims"] = [
        {
            "claim": CLAIM,
            "citations": [
                {"doc_id": first["doc_id"], "span_start": start, "span_end": end}
            ],
        }
    ]
    return answer


def _align(answer: dict[str, Any]) -> Any:
    return align_predictions(
        answer,
        EntityRoster(entity_ids=ROSTER),
        target_type="classification",
        interval_level=0.90,
    )


def test_alignment_reads_a_whole_number_float_offset_as_its_integer() -> None:
    aligned = _align(_nested(0.0, 57.0))
    cite = aligned.claims_by_entity[0][0].citations[0]
    assert (cite["span_start"], cite["span_end"]) == (0, 57)
    assert not any(isinstance(cite[k], float) for k in ("span_start", "span_end"))


@pytest.mark.parametrize(
    ("start", "end"),
    [(0.5, 57), (0, 57.25), (True, 57), (0, "57"), (0, float("nan"))],
)
def test_alignment_refuses_a_non_integer_offset_as_malformed(
    start: Any, end: Any
) -> None:
    with pytest.raises(T4ParticipantFailure) as excinfo:
        _align(_nested(start, end))
    assert excinfo.value.reason is T4Reason.CITATION_MALFORMED
