"""Regression tests for ``g3_domain_semantics`` — the embargo and faithfulness checks.

These pin the *call conventions* into the shared toolkit rather than any model behaviour, so they
need no weights and no network. They exist because both halves of g3 were previously inert, each
for the same underlying reason: a predicate keyed on a field that appears in zero real answers.

1. **Faithfulness was identically 0.0.** The gate handed raw claim dicts to
   ``citation_faithfulness``, which iterates ``claim["citations"]``. In the multi-entity format a
   claim element bundles the text and its single citation in one flat object with no ``citations``
   key, so the inner loop never ran and every well-formed answer failed.
2. **Embargo was unconditionally empty.** ``embargo_violations`` was called without a corpus
   lookup, leaving the citation's own self-reported ``doc_date`` as the only source of truth — a
   field ``analysis.schema.json`` does not define.

Both are covered here against a real synthetic unit tree, so the trusted corpus index and the
prediction-bound hypothesis are exercised rather than mocked out. The legacy flat single-entity
answer shape, which the pre-fix scorer accepted through a fallback branch, is now refused: it
carries no ``entity_predictions`` and therefore cannot be aligned to a roster, which is the whole
mechanism that fixes the denominator.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from qfbench2_common.failure_labels import FailureLabel

from qfbench2_track_analysis.codes import T4ParticipantFailure, T4Reason
from qfbench2_track_analysis.scoring import ClaimReport, _g3_domain_semantics, hydrate

from .synthetic import (
    POST_CUTOFF_DOC,
    PRE_CUTOFF_DOC,
    SUPPORTING_TEXT,
    StubJudge,
    answer_for,
    build_unit,
)

ROSTER = ("SYN-A",)


def _ctx(
    tmp_path: pathlib.Path, answer: dict[str, Any], judge: object | None
) -> dict[str, object]:
    unit = build_unit(tmp_path, entities=ROSTER)
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    ctx: dict[str, object] = {"unit_dir": unit, "output_dir": out, "judge": judge}
    hydrate(ctx)
    ctx["_answer"] = answer
    return ctx


def test_faithfulness_is_scored_for_bundled_citation_claims(
    tmp_path: pathlib.Path,
) -> None:
    """A supported prediction in the multi-entity format must score 1.0, not 0.0."""
    judge = StubJudge((SUPPORTING_TEXT,))
    ctx = _ctx(tmp_path, answer_for(entities=ROSTER, doc_id=PRE_CUTOFF_DOC), judge)

    result = _g3_domain_semantics(ctx)

    assert result.passed
    assert ctx["_faithfulness"] == 1.0
    # The judge must actually be consulted, with the cited span as premise. Zero calls is the
    # signature of the original defect.
    assert judge.calls
    assert judge.calls[0][0] == SUPPORTING_TEXT


def test_a_contradicted_claim_still_costs_the_unit(tmp_path: pathlib.Path) -> None:
    """The fix must not make the check vacuous in the other direction. 5.2.0: a claim the cited
    passage contradicts is false and the unit's factor drops with it (no refusal)."""
    judge = StubJudge((), contradicted_premises=(SUPPORTING_TEXT,))
    ctx = _ctx(tmp_path, answer_for(entities=ROSTER, doc_id=PRE_CUTOFF_DOC), judge)

    assert _g3_domain_semantics(ctx).passed
    report = ctx["_claim_report"]
    assert isinstance(report, ClaimReport)
    assert report.contradicted_count == report.claim_count > 0
    assert ctx["_faithfulness"] == 0.0
    assert ctx["_faithfulness_factor"] == 0.0


def test_embargo_resolves_the_date_from_the_corpus_not_the_answer(
    tmp_path: pathlib.Path,
) -> None:
    """A citation to a post-cutoff document is a violation even though the answer, like every
    real answer, carries no ``doc_date`` of its own."""
    judge = StubJudge()
    answer = answer_for(entities=ROSTER, doc_id=POST_CUTOFF_DOC)
    assert "doc_date" not in answer["entity_predictions"][0]["claims"][0]

    with pytest.raises(T4ParticipantFailure) as excinfo:
        _g3_domain_semantics(_ctx(tmp_path, answer, judge))

    assert excinfo.value.reason is T4Reason.CITATION_POST_CUTOFF
    assert excinfo.value.label is FailureLabel.T4_STALE_EVIDENCE
    # Counts only: naming the offending document would put a per-unit diagnostic in public output.
    assert excinfo.value.public_detail() == {
        "code": "cutoff_violation",
        "observed_count": 1,
        "violation_count": 1,
    }


def test_pre_cutoff_citation_is_not_an_embargo_violation(
    tmp_path: pathlib.Path,
) -> None:
    judge = StubJudge((SUPPORTING_TEXT,))
    ctx = _ctx(tmp_path, answer_for(entities=ROSTER, doc_id=PRE_CUTOFF_DOC), judge)
    assert _g3_domain_semantics(ctx).passed


def test_the_embargo_gate_runs_before_the_judge(tmp_path: pathlib.Path) -> None:
    """A post-cutoff citation must not reach the judge at all; ordering is part of the contract."""
    judge = StubJudge()
    with pytest.raises(T4ParticipantFailure):
        _g3_domain_semantics(
            _ctx(tmp_path, answer_for(entities=ROSTER, doc_id=POST_CUTOFF_DOC), judge)
        )
    assert judge.calls == []


def test_legacy_flat_answer_is_refused(tmp_path: pathlib.Path) -> None:
    """The single-entity fallback bypassed the roster entirely; it is gone on purpose."""
    legacy = {
        "task_id": "t4-SYNTH",
        "label": "beat",
        "interval": {"level": 0.90, "lo": 1.48, "hi": 1.78},
        "claims": [
            {
                "doc_id": PRE_CUTOFF_DOC,
                "span_start": 0,
                "span_end": len(SUPPORTING_TEXT),
                "claim": "Synthetic Issuer A beat consensus.",
            }
        ],
    }
    with pytest.raises(T4ParticipantFailure):
        _g3_domain_semantics(_ctx(tmp_path, legacy, StubJudge((SUPPORTING_TEXT,))))
