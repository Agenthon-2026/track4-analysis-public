"""What the published participant-facing artifacts promise, measured against the real scorer.

Three claims that shipped in `README.md` and `templates/answer.example.json` were wrong in the
same direction -- each described a soft penalty where the scorer actually returns the worst-case
`W = -0.27` for the whole submission -- and every one of them survived a mutation revert against
the full suite. They are pinned here by running `score_unit`, not by reading the prose:

* a `target_type` copied verbatim out of the answer template is `t4.target_type_mismatch` on every
  regression and ranking unit (`SCHEMA_INVALID_OUTPUT`), while an absent one is admitted and
  scored -- so the template ships without the field;
* a ranking unit is scored on `point_forecast`; `label` is never read, so "`label` carries the
  rank" cost a perfect ordering its entire score;
* a missing `interval` bound is `t4.schema_invalid` for the whole unit, not "zero coverage".
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from qfbench2_track_analysis.codes import T4Reason
from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import DOMAIN_MIN, UnitOutcome, score_unit

from .synthetic import (
    SUPPORTING_TEXT,
    PRE_CUTOFF_DOC,
    StubJudge,
    answer_for,
    build_unit,
    write_naive_answer,
)

_REPO = pathlib.Path(__file__).resolve().parents[2]
_TEMPLATE = _REPO / "templates" / "answer.example.json"

#: 1 = highest. The synthetic outcome is y = 1.0, 2.0, 3.0 for SYN-A, SYN-B, SYN-C.
_PERFECT_RANK = {"SYN-A": 3, "SYN-B": 2, "SYN-C": 1}


def _score(
    tmp_path: pathlib.Path,
    answer: dict[str, Any],
    *,
    target_type: str = "classification",
    entities: tuple[str, ...] = ("SYN-A", "SYN-B", "SYN-C"),
) -> UnitOutcome:
    unit = build_unit(
        tmp_path, target_type=target_type, with_outcome=True, entities=entities
    )
    if target_type == "regression":
        write_naive_answer(unit, entities)
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out},
        judge=StubJudge((SUPPORTING_TEXT,)),
        judge_provenance=provenance,
    )


def _template_answer() -> dict[str, Any]:
    """The shipped template with only its REPLACE markers resolved, as a participant would."""
    answer: dict[str, Any] = json.loads(_TEMPLATE.read_text(encoding="utf-8"))
    answer.pop("_comment", None)
    answer.pop("notes", None)
    answer["task_id"] = "t4-SYNTH"
    row: dict[str, Any] = answer["entity_predictions"][0]
    row["entity_id"] = "SYN-A"
    row["claims"] = [
        {
            "doc_id": PRE_CUTOFF_DOC,
            "span_start": 0,
            "span_end": len(SUPPORTING_TEXT),
            "claim": "Synthetic Issuer A beat consensus.",
        }
    ]
    answer["entity_predictions"] = [row]
    answer["evidence_trace"] = "synthetic"
    return answer


# --- the answer template ------------------------------------------------------------------------
@pytest.mark.parametrize("target_type", ["classification", "regression", "ranking"])
def test_the_shipped_answer_template_is_scoreable_on_every_target_type(
    tmp_path: pathlib.Path, target_type: str
) -> None:
    """Copied verbatim, the template must not DNF a unit merely by existing."""
    outcome = _score(
        tmp_path, _template_answer(), target_type=target_type, entities=("SYN-A",)
    )
    assert outcome.state == "participant_success"


@pytest.mark.parametrize("target_type", ["regression", "ranking"])
def test_a_target_type_baked_into_the_template_would_dnf_two_of_the_three(
    tmp_path: pathlib.Path, target_type: str
) -> None:
    """The control for the test above: this is exactly what a literal
    `"target_type": "classification"` in the template buys, and the shipped exemplar being a
    classification unit is why local practice never surfaces it."""
    answer = _template_answer()
    answer["target_type"] = "classification"
    outcome = _score(tmp_path, answer, target_type=target_type, entities=("SYN-A",))
    assert outcome.state == "participant_failure"
    assert outcome.score == DOMAIN_MIN
    assert outcome.diagnostics["reason"] == T4Reason.TARGET_TYPE_MISMATCH.value


# --- ranking is scored on point_forecast, never on label -----------------------------------------
def _ranking_answer(*, rank_into: str) -> dict[str, Any]:
    answer = answer_for()
    answer.pop("target_type", None)
    for row in answer["entity_predictions"]:
        rank = _PERFECT_RANK[row["entity_id"]]
        if rank_into == "label":
            row["label"] = str(rank)
            row["point_forecast"] = float(rank)
        else:
            row["label"] = "ignored"
            row["point_forecast"] = float(4 - rank)  # the predicted metric value
    return answer


def test_the_rank_integer_in_point_forecast_inverts_the_ordering(
    tmp_path: pathlib.Path,
) -> None:
    """`_ranking_answer(rank_into="label")` writes the rank into BOTH fields, so what this
    measures is the rank INTEGER sitting in `point_forecast` (1 = highest) against a metric
    where larger is better -- a perfectly inverted ordering, hence 0.0.

    It was previously named ...`_written_into_label_is_not_read`, which it does not show:
    `label` being unread is proved by `test_label_cannot_change_a_ranking_score_at_all`
    below, which holds `point_forecast` fixed and varies only `label`. Naming a test for a
    claim it does not isolate is how a wrong sentence ends up in the README citing it.
    """
    outcome = _score(
        tmp_path, _ranking_answer(rank_into="label"), target_type="ranking"
    )
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["predictive_quality"] == pytest.approx(0.0)
    # 5.0.0 measured -0.03 (= 0 - 0.3*0.1). 5.1.0: 0.7*0 + 0.3*iq, iq 0.5 (band == naive band).
    assert outcome.score == pytest.approx(0.15)


def test_the_same_ordering_in_point_forecast_scores_full_marks(
    tmp_path: pathlib.Path,
) -> None:
    outcome = _score(
        tmp_path, _ranking_answer(rank_into="point_forecast"), target_type="ranking"
    )
    assert outcome.diagnostics["predictive_quality"] == pytest.approx(1.0)
    # 5.0.0: 0.67. 5.1.0: 0.7*1 + 0.3*0.5.
    assert outcome.score == pytest.approx(0.85)


def test_label_cannot_change_a_ranking_score_at_all(tmp_path: pathlib.Path) -> None:
    good = _ranking_answer(rank_into="point_forecast")
    garbled = json.loads(json.dumps(good))
    for row in garbled["entity_predictions"]:
        row["label"] = "not-a-rank"
    assert _score(tmp_path / "a", good, target_type="ranking").score == pytest.approx(
        _score(tmp_path / "b", garbled, target_type="ranking").score
    )


def test_a_constant_point_forecast_scores_the_neutral_value_not_full_marks(
    tmp_path: pathlib.Path,
) -> None:
    """An answer that expresses no ordering must NOT tie a perfect one.

    RENAMED from `test_a_constant_point_forecast_ties_a_perfect_one`, which asserted the
    exploit rather than the contract. That test required a constant `point_forecast` to score
    predictive_quality 1.0 / composite 0.67 -- byte-identical to the honest answer -- and it
    passed, because the ranking metric broke ties by POSITION: every tie resolved to the
    roster's own order, so a submitter who expressed no opinion inherited the organizer's
    ordering and was graded on it as if it were a prediction. On this fixture the roster order
    happens to match the outcome, which is what turned "no information" into full marks.

    The shared ranking metric now ranks ties AS ties (average ranks), so a constant vector has
    no rank variance, Spearman's rho is 0.0, and the rescaling `(rho + 1) / 2` puts it at the
    NEUTRAL 0.5 -- neither rewarded nor punished for saying nothing. Measured on this fixture
    2026-08-29 against the shared toolkit:

        constant point_forecast -> predictive_quality 0.5, composite 0.32
        honest (true metric values) -> predictive_quality 1.0, composite 0.67

    Expressing no opinion stays ADMISSIBLE -- neutral is not a failure, and this test pins that
    too, so a later "fix" cannot quietly turn a flat answer into a DNF.

    THIS TEST GOES RED IF THE EXPLOIT REOPENS. Verified by execution, not by assertion: run
    against the pre-fix ranking code (position tie-breaking) the constant answer measures
    predictive_quality 1.0 / composite 0.67, and the two `pytest.approx` assertions below fail.

    Row-order independence is still pinned here: `align_predictions` re-indexes against
    the roster, so serialisation order cannot move the score. An earlier draft asserted the
    opposite and failed, which is how the roster-ordering guarantee got pinned rather than
    assumed.
    """
    flat = _ranking_answer(rank_into="point_forecast")
    for row in flat["entity_predictions"]:
        row["point_forecast"] = 0.0
    forward = _score(tmp_path / "fwd", flat, target_type="ranking")

    reordered = json.loads(json.dumps(flat))
    reordered["entity_predictions"] = list(reversed(reordered["entity_predictions"]))
    backward = _score(tmp_path / "rev", reordered, target_type="ranking")

    honest = _score(
        tmp_path / "honest",
        _ranking_answer(rank_into="point_forecast"),
        target_type="ranking",
    )

    assert (
        forward.diagnostics["predictive_quality"]
        == backward.diagnostics["predictive_quality"]
    ), "submission row order moved a ranking score; align_predictions should have prevented that"

    # Neutral, not full marks. These are the assertions the exploit fails.
    assert forward.diagnostics["predictive_quality"] == pytest.approx(0.5)
    # 5.0.0: 0.32 and 0.67 below. 5.1.0 adds 0.3*iq (iq 0.5) instead of subtracting 0.03.
    assert forward.score == pytest.approx(0.50)

    # ... and the honest answer is strictly better, which is the whole point.
    assert honest.diagnostics["predictive_quality"] == pytest.approx(1.0)
    assert honest.score == pytest.approx(0.85)
    assert (
        forward.diagnostics["predictive_quality"]
        < honest.diagnostics["predictive_quality"]
    )
    assert forward.score < honest.score

    # Saying nothing is neutral, not a DNF: the submission is still admitted and scored.
    assert forward.state == "participant_success"


def test_omitting_point_forecast_on_a_ranking_unit_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """README.md once told ranking entries that `label` was the primary prediction and
    `point_forecast` belonged to regression. Followed literally, that is a DNF."""
    answer = _ranking_answer(rank_into="label")
    for row in answer["entity_predictions"]:
        row.pop("point_forecast")
    outcome = _score(tmp_path, answer, target_type="ranking")
    assert outcome.state == "participant_failure"
    assert outcome.score == DOMAIN_MIN


# --- a missing interval bound is not a coverage penalty ------------------------------------------
@pytest.mark.parametrize("bound", ["lo", "hi"])
def test_a_missing_interval_bound_fails_the_whole_submission(
    tmp_path: pathlib.Path, bound: str
) -> None:
    answer = answer_for()
    answer.pop("target_type", None)
    del answer["entity_predictions"][0]["interval"][bound]
    outcome = _score(tmp_path, answer)
    assert outcome.state == "participant_failure"
    assert outcome.score == DOMAIN_MIN
    assert outcome.diagnostics["reason"] == T4Reason.SCHEMA_INVALID.value
    # Not a coverage number at all -- the unit never reaches the coverage leg.
    assert outcome.diagnostics.get("interval_coverage") is None


# --- a null point_forecast on a classification unit ----------------------------------------------
def test_a_classification_row_may_omit_point_forecast_but_not_send_null(
    tmp_path: pathlib.Path,
) -> None:
    """On a classification unit `point_forecast` is optional, so an absent one is scored; a
    `null` one fails the published schema for the whole submission. The strong RAG baseline
    leaves a missing value out for this reason (its tests pin that side)."""
    absent = answer_for()
    absent.pop("target_type", None)
    del absent["entity_predictions"][0]["point_forecast"]
    admitted = _score(tmp_path / "absent", absent)
    assert admitted.state == "participant_success"

    null = answer_for()
    null.pop("target_type", None)
    null["entity_predictions"][0]["point_forecast"] = None
    refused = _score(tmp_path / "null", null)
    assert refused.state == "participant_failure"
    assert refused.score == DOMAIN_MIN
    assert refused.diagnostics["reason"] == T4Reason.SCHEMA_INVALID.value
