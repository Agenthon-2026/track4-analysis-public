"""A refused submission keeps the number that says how far it was from grounding.

The defect. `score_unit` computed faithfulness, and then the participant_failure branch
returned `diagnostics={"reason": ...}` and nothing else — so on an EVIDENCE_UNSUPPORTED refusal,
where `_faithfulness` IS populated before the raise, the one number saying HOW FAR the
submission was from grounding its predictions was discarded at exactly the moment it mattered.

A missing key is worse than it looks, because the natural repair downstream is to substitute
`0.0` for it. That makes every refusal — including every refusal that never reached the judge —
indistinguishable from a submission the judge measured at exactly zero, and any conclusion drawn
from the resulting distribution is an artifact of the absence rather than a property of the
submissions.

Two states, and they must stay distinguishable:

    measured       the gate ran and produced a number -> a float
    not computed   the unit failed before g3 -> None, NEVER 0.0

`diagnostics` is OPERATOR-ONLY — the dataclass says "Never serialized into a participant-visible
artifact" — and the participant projection is `detail`, which admits an enum code and
non-negative integers only. So carrying this discloses nothing to a participant, and in
particular does not tell one how close it came.
"""

from __future__ import annotations

import json
import pathlib

from typing import Any

from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import UnitOutcome, score_unit

from .synthetic import SUPPORTING_TEXT, StubJudge, answer_for, build_unit, outcome_for


def _score(
    tmp_path: pathlib.Path, answer: dict[str, Any], *, judge: Any
) -> UnitOutcome:
    unit = build_unit(tmp_path, with_outcome=False, with_naive=True)
    (unit / "reference").mkdir(exist_ok=True)
    (unit / "reference" / "outcome.json").write_text(
        json.dumps(outcome_for()), encoding="utf-8"
    )
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out}, judge=judge, judge_provenance=provenance
    )


def test_a_penalised_unit_reports_the_measured_value(
    tmp_path: pathlib.Path,
) -> None:
    """THE GUARD. 5.2.0 no longer refuses on faithfulness: a judge that contradicts every claim
    costs the unit its whole score, and the diagnostics say how far it was (a measured 0.0, not
    a missing value)."""
    outcome = _score(
        tmp_path,
        answer_for(),
        judge=StubJudge((), contradicted_premises=(SUPPORTING_TEXT,)),
    )
    assert outcome.state == "participant_success"
    faithfulness = outcome.diagnostics.get("faithfulness")
    assert faithfulness is not None, "the penalty discarded the measured faithfulness"
    assert isinstance(faithfulness, float)
    assert faithfulness == 0.0
    assert outcome.diagnostics["faithfulness_factor"] == 0.0
    assert outcome.score == 0.0


def test_a_failure_before_the_judge_reports_none_not_zero(
    tmp_path: pathlib.Path,
) -> None:
    """THE OTHER HALF, and the one the 0.0 substitution destroyed.

    An unresolvable doc_id fails at citation resolution, before any entailment is computed. The
    honest value is None. If this ever returns 0.0, a consumer cannot tell a submission that
    grounded nothing from one that was never assessed.
    """
    answer = answer_for()
    for row in answer["entity_predictions"]:
        for claim in row["claims"]:
            claim["doc_id"] = "SYNTHDOC_NOT_DECLARED"
    outcome = _score(tmp_path, answer, judge=StubJudge((SUPPORTING_TEXT,)))
    assert outcome.state == "participant_failure"
    assert outcome.diagnostics.get("faithfulness") is None


def test_a_scored_unit_still_reports_it(tmp_path: pathlib.Path) -> None:
    """POSITIVE CONTROL: the success path is unchanged."""
    outcome = _score(tmp_path, answer_for(), judge=StubJudge((SUPPORTING_TEXT,)))
    assert outcome.state == "participant_success"
    assert isinstance(outcome.diagnostics.get("faithfulness"), float)


def test_the_participant_projection_never_carries_it(tmp_path: pathlib.Path) -> None:
    """`detail` is the redacted public projection: enum code plus non-negative integers only."""
    outcome = _score(tmp_path, answer_for(), judge=StubJudge(()))
    assert "faithfulness" not in outcome.detail
    for key, value in outcome.detail.items():
        assert key == "code" or (isinstance(value, int) and not isinstance(value, bool))
