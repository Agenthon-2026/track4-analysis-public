"""A rankable factory constructs a production judge, or there is no score.

The pre-fix behaviour, measured: under the driver's three-key context the official factory built no
judge, ``ctx.get("judge")`` was ``None``, the faithfulness block was skipped entirely, and
``_score`` read ``ctx.get("_faithfulness", 1.0)`` — a **missing judge defaulted faithfulness to
perfect** and an admissible ``0.67`` came back for a submission that cited a claim the corpus
contradicts. The regression below is the direct inverse: with no judge configured, the official
factory must refuse.

Positive control: the smoke factory builds, scores, and is visibly non-rankable — a gate that
rejects the legitimate case makes every rejection beside it uninterpretable.
"""

from __future__ import annotations

import importlib
import json
import pathlib
from typing import Any

import pytest

from qfbench2_common.contracts import JudgeRecord, OrganizerFault
from qfbench2_common.smoke import resolve_verifier_factory, run_smoke

from qfbench2_track_analysis import judge_factory
from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.scoring import (
    build_smoke_verifier,
    build_verifier,
    score_unit,
)

from .synthetic import SUPPORTING_TEXT, StubJudge, answer_for, build_unit


def _ctx(tmp_path: pathlib.Path, *, with_outcome: bool = True) -> dict[str, Any]:
    unit = build_unit(tmp_path, with_outcome=with_outcome)
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer_for()), encoding="utf-8")
    return {"unit_dir": unit, "output_dir": out, "failure_map": tmp_path / "fmap.jsonl"}


# --- regression: the official factory refuses without a judge --------------------------------
def test_official_factory_refuses_when_no_judge_artifact_is_configured(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-fix this returned a working verifier that scored 0.67 with faithfulness defaulted to 1."""
    monkeypatch.delenv(judge_factory.ENV_JUDGE_SPEC, raising=False)
    ctx = _ctx(tmp_path)
    with pytest.raises(OrganizerFault):
        build_verifier(ctx)


def test_official_factory_refuses_an_unpinned_model_revision(
    tmp_path: pathlib.Path,
) -> None:
    spec = tmp_path / "judge.json"
    spec.write_text(
        json.dumps(
            {
                "model_ids": ["synthetic/nli-a"],
                "model_revisions": {"synthetic/nli-a": "main"},
                "tokenizer_digest": "sha256:" + "0" * 64,
                "cache_tree_digest": "sha256:" + "0" * 64,
                "cache_dir": str(tmp_path / "cache"),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(T4OrganizerFault, match="40-hex commit sha"):
        judge_factory.load_judge_spec(spec)


def test_official_factory_refuses_a_cache_that_does_not_match_its_digest(
    tmp_path: pathlib.Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "weights.bin").write_bytes(b"synthetic weights")
    spec = judge_factory.JudgeSpec(
        model_ids=("synthetic/nli-a",),
        model_revisions={"synthetic/nli-a": "a" * 40},
        tokenizer_digest="sha256:" + "1" * 64,
        cache_tree_digest="sha256:" + "2" * 64,  # deliberately wrong
        cache_dir=str(cache),
    )
    with pytest.raises(T4OrganizerFault, match="does not match the digest"):
        judge_factory.build_production_judge(spec, judge_builder=lambda *_: StubJudge())


def test_a_judge_that_fails_to_load_is_an_organizer_fault_not_a_stub_fallback(
    tmp_path: pathlib.Path,
) -> None:
    """The other measured defect: a construction exception fell back to a lexical judge silently."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "weights.bin").write_bytes(b"synthetic weights")
    digest = judge_factory.compute_cache_tree_digest(cache)
    spec = judge_factory.JudgeSpec(
        model_ids=("synthetic/nli-a",),
        model_revisions={"synthetic/nli-a": "a" * 40},
        tokenizer_digest="sha256:" + "1" * 64,
        cache_tree_digest=digest,
        cache_dir=str(cache),
    )

    def _explode(*_: object) -> object:
        raise RuntimeError("weights not found")

    with pytest.raises(T4OrganizerFault, match="never a silent fallback"):
        judge_factory.build_production_judge(spec, judge_builder=_explode)


def test_scoring_without_a_judge_in_context_is_an_organizer_fault(
    tmp_path: pathlib.Path,
) -> None:
    ctx = _ctx(tmp_path)
    _, provenance = judge_factory.build_smoke_judge()
    with pytest.raises(OrganizerFault, match="no NLI judge"):
        score_unit(ctx, judge=None, judge_provenance=provenance)


# --- positive controls ------------------------------------------------------------------------
def test_production_provenance_parses_as_a_c4_judge_record(
    tmp_path: pathlib.Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "weights.bin").write_bytes(b"synthetic weights")
    spec = judge_factory.JudgeSpec(
        model_ids=("synthetic/nli-a", "synthetic/nli-b"),
        model_revisions={"synthetic/nli-a": "a" * 40, "synthetic/nli-b": "b" * 40},
        tokenizer_digest="sha256:" + "1" * 64,
        cache_tree_digest=judge_factory.compute_cache_tree_digest(cache),
        cache_dir=str(cache),
    )
    judge, provenance = judge_factory.build_production_judge(
        spec, judge_builder=lambda *_: StubJudge()
    )
    assert judge is not None
    assert provenance.judge_mode == "production"
    assert provenance.rankable is True
    record = JudgeRecord.from_mapping(provenance.to_mapping())
    assert record.is_rankable is True
    assert record.model_revisions["synthetic/nli-a"] == "a" * 40


def test_smoke_factory_scores_and_is_visibly_non_rankable(
    tmp_path: pathlib.Path,
) -> None:
    """The positive control. It must WORK, and it must be unmistakably not a production run."""
    ctx = _ctx(tmp_path)
    verifier = build_smoke_verifier(ctx)
    verdict = verifier.run(ctx)
    assert verdict.admissible is True
    assert verdict.detail["judge_mode"] == "smoke"
    assert verdict.detail["rankable"] is False
    assert ctx["judge_provenance"].rankable is False
    assert (
        JudgeRecord.from_mapping(ctx["judge_provenance"].to_mapping()).is_rankable
        is False
    )


def test_no_environment_variable_selects_the_smoke_judge(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`QFBENCH2_ANALYSIS_JUDGE=stub` used to downgrade a ranked run. Nothing may do that now."""
    for name in (
        "QFBENCH2_ANALYSIS_JUDGE",
        "T4_JUDGE_BACKEND",
        "QFBENCH2_T4_JUDGE_MODE",
    ):
        monkeypatch.setenv(name, "stub")
    monkeypatch.delenv(judge_factory.ENV_JUDGE_SPEC, raising=False)
    ctx = _ctx(tmp_path)
    with pytest.raises(OrganizerFault):
        build_verifier(ctx)


def test_scored_run_records_the_judge_sub_record_on_every_outcome(
    tmp_path: pathlib.Path,
) -> None:
    ctx = _ctx(tmp_path)
    judge = StubJudge((SUPPORTING_TEXT,))
    _, provenance = judge_factory.build_smoke_judge()
    outcome = score_unit(ctx, judge=judge, judge_provenance=provenance)
    assert outcome.state == "participant_success"
    assert outcome.judge["judge_mode"] == "smoke"
    assert outcome.rankable is False


def test_smoke_mode_records_that_the_faithfulness_gate_was_not_applied(
    tmp_path: pathlib.Path,
) -> None:
    """A smoke "admissible" must never be readable as a production one.

    The lexical proxy's number is on a different scale from the pinned ensemble's, so the smoke
    factory reports faithfulness and does not gate on it. That is only safe because the artifact
    says so: `rankable=False`, `judge_mode="smoke"`, `faithfulness_gate_applied=False`.
    """
    ctx = _ctx(tmp_path)
    verdict = build_smoke_verifier(ctx).run(ctx)
    assert verdict.detail["faithfulness_gate_applied"] is False
    assert verdict.detail["rankable"] is False
    assert isinstance(verdict.detail["faithfulness"], float)


def test_the_production_path_always_applies_the_faithfulness_gate(
    tmp_path: pathlib.Path,
) -> None:
    """The inverse: nothing outside `build_smoke_verifier` may set the non-gating flag."""
    ctx = _ctx(tmp_path)
    _, provenance = judge_factory.build_smoke_judge()
    judge = StubJudge(("nothing entails this",))
    # A judge that contradicts every passage it reads: applied, the penalty zeroes the unit.
    judge.contradicted_premises = {premise for premise in _all_corpus_texts(ctx)}
    outcome = score_unit(ctx, judge=judge, judge_provenance=provenance)
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["faithfulness_factor"] == 0.0
    assert outcome.score == 0.0


def _all_corpus_texts(ctx: dict[str, Any]) -> list[str]:
    import json as _json
    import pathlib as _pathlib

    corpus = _pathlib.Path(ctx["unit_dir"]) / "corpus"
    return [
        _json.loads(path.read_text(encoding="utf-8"))["text"]
        for path in sorted(corpus.glob("*.json"))
    ]


def test_the_shared_runner_resolves_the_smoke_factory_by_profile() -> None:
    """Track 4's `python -m qfbench2_track_analysis.smoke` stopgap is deleted; this replaces it.

    The stopgap existed because `qfbench2-smoke` called `build_verifier` by NAME, so the
    participant preview path ran the rankable factory and refused without a pinned production
    judge. The hub now selects by profile. What Track 4 must keep true is that BOTH factory names
    stay exported under the names the shared resolver looks for — that is the whole contract
    between the two repos, and it is asserted here rather than assumed.
    """
    import qfbench2_track_analysis.scoring as scoring_module

    name, factory = resolve_verifier_factory(scoring_module, "smoke")
    assert name == "build_smoke_verifier"
    assert factory is build_smoke_verifier
    name, factory = resolve_verifier_factory(scoring_module, "production")
    assert name == "build_verifier"
    assert factory is scoring_module.build_verifier
    with pytest.raises(SystemExit):
        resolve_verifier_factory(scoring_module, "rankable")


def test_the_smoke_profile_runs_the_example_unit_without_a_production_judge(
    tmp_path: pathlib.Path,
) -> None:
    """End to end through the SHARED runner, the way a participant reaches it."""
    unit = pathlib.Path(__file__).resolve().parents[2] / "units/t4-EXAMPLE-eps-beat"
    out = tmp_path / "res"
    out.mkdir()
    (out / "answer.json").write_text(
        json.dumps(
            {
                "task_id": "t4-EXAMPLE-eps-beat",
                "target_type": "classification",
                "entity_predictions": [
                    {
                        "entity_id": "AAPL",
                        "label": "beat",
                        "interval": {"level": 0.9, "lo": 0.4, "hi": 0.95},
                        "claims": [
                            {
                                "doc_id": "EDGAR_0000320193_10Q_20240202",
                                "span_start": 0,
                                "span_end": 20,
                                "claim": "Apple beat consensus.",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _name, factory = resolve_verifier_factory(
        importlib.import_module("qfbench2_track_analysis.scoring"), "smoke"
    )
    verdict = run_smoke(unit, out, factory)
    assert verdict.admissible is True
    assert verdict.detail["rankable"] is False
    assert verdict.detail["faithfulness_gate_applied"] is False
