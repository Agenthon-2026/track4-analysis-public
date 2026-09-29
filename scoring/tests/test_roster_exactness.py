"""The trusted roster is the denominator, and answering less can never help.

The measured exploit, on the pre-fix scorer: a three-entity unit, one correct prediction available.
The honest full-roster answer scored ``0.2033``; answering only the entity it was sure of scored
``0.6700`` — **3.3x for answering less** — because alignment iterated the participant's rows and
skipped anything with no matching truth, so the truth vector was rebuilt from the submission.
Duplicates re-consumed the same truth row and reweighted the metric; unknown ids were dropped.

The regressions below reproduce all three and require each to be refused *before* a metric runs.
The positive control is the honest answer, which must still score, and must score exactly what it
scored before — the fix removes the exploit, it does not move the honest number.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from qfbench2_track_analysis.alignment import EntityRoster, align_predictions
from qfbench2_track_analysis.codes import T4ParticipantFailure, T4Reason
from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import DOMAIN_MIN, UnitOutcome, score_unit

from .synthetic import SUPPORTING_TEXT, StubJudge, answer_for, build_unit

ROSTER = ("SYN-A", "SYN-B", "SYN-C")


def _score(tmp_path: pathlib.Path, answer: dict[str, Any]) -> UnitOutcome:
    unit = build_unit(tmp_path, with_outcome=True)
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out},
        judge=StubJudge((SUPPORTING_TEXT,)),
        judge_provenance=provenance,
    )


# --- positive control -------------------------------------------------------------------------
def test_honest_full_roster_answer_scores(tmp_path: pathlib.Path) -> None:
    outcome = _score(tmp_path, answer_for())
    assert outcome.state == "participant_success"
    # 5.1.0 interval leg: answer band == naive band -> iq 0.5; 5.2.0: the answer's labels equal
    # the naive rule's (accuracy 1/3 each), so the anchored prediction leg is exactly 0.5
    assert outcome.diagnostics["raw_predictive_quality"] == pytest.approx(1 / 3)
    assert outcome.score == pytest.approx(0.7 * 0.5 + 0.3 * 0.5)
    assert outcome.diagnostics["expected_entity_count"] == 3
    assert outcome.diagnostics["graded_entity_count"] == 3


# --- regressions ------------------------------------------------------------------------------
def test_answering_a_favourable_subset_is_a_failure_not_a_higher_score(
    tmp_path: pathlib.Path,
) -> None:
    """Pre-fix this returned (admissible, 0.67). It must now be W, in the denominator."""
    outcome = _score(tmp_path, answer_for(entities=("SYN-A",)))
    assert outcome.state == "participant_failure"
    assert outcome.score == DOMAIN_MIN
    assert outcome.detail["missing_count"] == 2
    assert outcome.detail["expected_count"] == 3
    assert outcome.diagnostics["reason"] == T4Reason.ENTITY_MISSING.value


def test_a_subset_answer_never_beats_the_honest_full_answer(
    tmp_path: pathlib.Path,
) -> None:
    """The property, stated directly, because the counts above are only evidence for it."""
    honest = _score(tmp_path / "a", answer_for())
    gamed = _score(tmp_path / "b", answer_for(entities=("SYN-A",)))
    assert gamed.score < honest.score


def test_duplicate_entity_is_refused_before_any_metric(tmp_path: pathlib.Path) -> None:
    outcome = _score(
        tmp_path, answer_for(entities=("SYN-A", "SYN-A", "SYN-B", "SYN-C"))
    )
    assert outcome.state == "participant_failure"
    assert outcome.diagnostics["reason"] == T4Reason.ENTITY_DUPLICATE.value
    assert outcome.score == DOMAIN_MIN


def test_unknown_entity_is_refused_rather_than_silently_dropped(
    tmp_path: pathlib.Path,
) -> None:
    outcome = _score(
        tmp_path, answer_for(entities=("SYN-A", "SYN-B", "SYN-C", "SYN-Z"))
    )
    assert outcome.state == "participant_failure"
    assert outcome.diagnostics["reason"] == T4Reason.ENTITY_UNKNOWN.value
    assert outcome.detail["extra_count"] == 1


# --- ordering ----------------------------------------------------------------------------------
def test_alignment_uses_the_trusted_order_not_the_participants(
    tmp_path: pathlib.Path,
) -> None:
    roster = EntityRoster(entity_ids=ROSTER, labels=("beat", "miss", "inline"))
    shuffled = answer_for(entities=("SYN-C", "SYN-A", "SYN-B"))
    for index, row in enumerate(shuffled["entity_predictions"]):
        row["label"] = ("inline", "beat", "miss")[index]
    aligned = align_predictions(
        shuffled, roster, target_type="classification", interval_level=0.90
    )
    assert aligned.entity_ids == ROSTER
    assert aligned.pred_labels == ("beat", "miss", "inline")


def test_ranking_requires_a_full_permutation(tmp_path: pathlib.Path) -> None:
    roster = EntityRoster(entity_ids=ROSTER)
    answer = answer_for()
    for index, row in enumerate(answer["entity_predictions"]):
        row["rank"] = 1 if index == 0 else 1  # duplicated rank, not a permutation
        row["point_forecast"] = float(index)
    with pytest.raises(T4ParticipantFailure):
        align_predictions(answer, roster, target_type="ranking", interval_level=0.90)


def test_a_malformed_entity_row_is_a_failure_not_a_skipped_row(
    tmp_path: pathlib.Path,
) -> None:
    roster = EntityRoster(entity_ids=ROSTER)
    answer = answer_for()
    answer["entity_predictions"].append({"no_entity_id": True})
    with pytest.raises(T4ParticipantFailure) as excinfo:
        align_predictions(
            answer, roster, target_type="classification", interval_level=0.90
        )
    assert excinfo.value.reason is T4Reason.SCHEMA_INVALID
