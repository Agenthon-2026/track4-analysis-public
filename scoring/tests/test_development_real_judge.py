"""Scorer 5.3.0: Development scores with the real NLI judge, as the Final does.

The platform driver runs `build_smoke_verifier` for a development-signed plan. Before 5.3.0 that
factory always used the lexical stand-in and did not charge contradiction verdicts, so the
Development board ran without the NLI contradiction check. From 5.3.0, whenever the organizer's
judge spec is configured (`QFBENCH2_T4_JUDGE_SPEC`, as in the scoring images) it builds the pinned
production judge and charges the contradiction check exactly as `build_verifier` does; its
results stay non-rankable. With no spec configured (a participant's machine) it is the lexical
preview, as before.

The judge here is a counting stand-in built through the real `build_production_judge` (synthetic
spec and cache), so no weights are loaded.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from qfbench2_track_analysis import judge_factory
from qfbench2_track_analysis import scoring as scoring_module
from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.scoring import build_smoke_verifier

from .synthetic import SUPPORTING_TEXT, StubJudge, answer_for, build_unit


class WindowedStubJudge(StubJudge):
    """A stand-in with the window and token count a rankable run requires."""

    def judged_premise(self, premise: str, hypothesis: str) -> str:
        return premise

    def claim_tokens(self, hypothesis: str) -> int:
        return len(hypothesis.split())


def _spec(tmp_path: pathlib.Path) -> judge_factory.JudgeSpec:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "synthetic-weights").write_bytes(b"synthetic cache contents")
    return judge_factory.JudgeSpec.from_mapping(
        {
            "model_ids": ["synthetic/nli-a"],
            "model_revisions": {"synthetic/nli-a": "a" * 40},
            "tokenizer_digest": "sha256:" + "1" * 64,
            "cache_tree_digest": judge_factory.compute_cache_tree_digest(cache),
            "cache_dir": str(cache),
        },
        source="synthetic development fixture",
    )


def _ctx(tmp_path: pathlib.Path) -> dict[str, Any]:
    unit = build_unit(tmp_path, with_outcome=True)
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer_for()), encoding="utf-8")
    return {"unit_dir": unit, "output_dir": out, "failure_map": tmp_path / "fmap.jsonl"}


def _configure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> WindowedStubJudge:
    """A configured judge spec, and a production judge that contradicts every claim."""
    judge = WindowedStubJudge(
        (SUPPORTING_TEXT,), contradicted_premises=(SUPPORTING_TEXT,)
    )
    spec = _spec(tmp_path)
    built = judge_factory.build_production_judge(spec, judge_builder=lambda *_: judge)
    calls: list[int] = []

    def production() -> tuple[Any, judge_factory.JudgeProvenance]:
        calls.append(1)
        return built

    monkeypatch.setenv(judge_factory.ENV_JUDGE_SPEC, str(tmp_path / "spec.json"))
    monkeypatch.setattr(scoring_module, "build_production_judge", production)
    judge.factory_calls = calls  # type: ignore[attr-defined]
    return judge


def test_development_charges_the_contradiction_check_with_a_configured_judge(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = _configure(tmp_path, monkeypatch)
    ctx = _ctx(tmp_path)
    verdict = build_smoke_verifier(ctx).run(ctx)
    assert judge.factory_calls == [1]  # type: ignore[attr-defined]
    assert (
        judge.contradiction_calls
    )  # the real judge's contradiction question was asked
    assert verdict.detail["faithfulness_gate_applied"] is True
    assert verdict.detail["judge_mode"] == "production"
    assert verdict.detail["faithfulness_factor"] == 0.0  # every claim contradicted
    assert verdict.detail["score"] == 0.0
    # Development stays unofficial whatever judge scored it.
    assert verdict.detail["rankable"] is False


def test_development_equals_the_final_on_the_same_judge(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same judge, same answer: the Development factory's score is the Final factory's."""
    judge = _configure(tmp_path, monkeypatch)
    ctx = _ctx(tmp_path)
    development = build_smoke_verifier(ctx).run(ctx)
    (tmp_path / "final").mkdir()
    monkeypatch.setattr(
        scoring_module,
        "build_production_judge",
        lambda: judge_factory.build_production_judge(
            _spec(tmp_path / "final"), judge_builder=lambda *_: judge
        ),
    )
    ctx = _ctx(tmp_path / "final-unit")
    final = scoring_module.build_verifier(ctx).run(ctx)
    for key in ("score", "faithfulness", "faithfulness_factor", "prediction_relevance"):
        assert development.detail[key] == final.detail[key], key
    assert final.detail["rankable"] is True


def test_a_configured_judge_that_cannot_load_is_an_organizer_fault(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never a quiet fall back to the lexical judge on a host configured for the real one."""
    monkeypatch.setenv(judge_factory.ENV_JUDGE_SPEC, str(tmp_path / "missing.json"))
    with pytest.raises(T4OrganizerFault):
        build_smoke_verifier(_ctx(tmp_path))


def test_a_scoring_host_without_a_judge_spec_is_an_organizer_fault(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The platform driver passes a signed C1 plan (`ctx["plan"]`); a Development host that
    gets one with no judge configured fails closed instead of scoring lexically."""
    monkeypatch.delenv(judge_factory.ENV_JUDGE_SPEC, raising=False)
    ctx = _ctx(tmp_path)
    ctx["plan"] = object()  # any signed plan; the refusal comes before the plan is read
    with pytest.raises(T4OrganizerFault, match="Development scoring host"):
        build_smoke_verifier(ctx)


def test_without_a_configured_judge_the_preview_is_the_lexical_one(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A participant's machine (no signed plan, no judge spec): unchanged from 5.2.2."""
    monkeypatch.delenv(judge_factory.ENV_JUDGE_SPEC, raising=False)
    ctx = _ctx(tmp_path)
    verdict = build_smoke_verifier(ctx).run(ctx)
    assert verdict.detail["judge_mode"] == "smoke"
    assert verdict.detail["faithfulness_gate_applied"] is False
    assert verdict.detail["rankable"] is False
