"""The contradiction question on both judge backends (scorer 5.2.0).

The served endpoint returns one model's two-way entailment only (`POST /entail -> {"entailment": ...}`), so a served
ensemble cannot answer the three-way contradiction question. In the local preview (`check_answer`) and any other
NON-rankable run it is marked "contradiction not applied" and non-rankable, and only the deterministic reasons are
checked: never an organizer fault. The local ensemble answers it. A rankable run without it is refused.
No weights, no network: stand-in pipelines and the word-level tokenizer of `test_judge_window.py`.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest
import transformers

from faithfulness import judge as judge_module
from faithfulness.judge import (
    DeBERTaNLIJudge,
    ServedNLIJudge,
    build_judge,
    build_unit_context,
    check_answer,
)
from faithfulness.tests.test_judge_window import _tokenizer
from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.judge_factory import (
    JudgeProvenance,
    WindowedEnsembleNLIJudge,
    build_smoke_judge,
)
from qfbench2_track_analysis.scoring import contradiction_supported, score_unit
from scoring.tests.synthetic import PRE_CUTOFF_DOC, _doc, answer_for, build_unit

ROSTER = ("SYN-A",)
SPAN = "revenue rose 8 percent to 5.2 billion"
TRUE_CLAIM = "revenue was 5.2 billion"
FALSE_CLAIM = "revenue fell sharply"  # figure-free: only the judge could make it false
WRONG_FIGURE = (
    "revenue was 9.9 billion"  # unanchored: false by exact code on either backend
)


def _served(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _tokenizer(64)
    )
    # No network: the endpoint's two-way entailment (read only by the recorded prediction_relevance diagnostic).
    monkeypatch.setattr(ServedNLIJudge, "entail", lambda self, premise, hypothesis: 0.5)
    monkeypatch.setenv(judge_module.ENV_JUDGE_BACKEND, "served")
    monkeypatch.setenv(judge_module.ENV_JUDGE_URL, "http://127.0.0.1:9")
    ensemble = build_judge(cache_dir="/synthetic-cache")
    assert all(isinstance(m, ServedNLIJudge) for m in ensemble._judges)
    return ensemble


class _Logits:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def float(self) -> _Logits:  # noqa: A003 - mirrors torch.Tensor.float
        return self

    def reshape(self, *_: Any) -> _Logits:
        return self

    def tolist(self) -> list[Any]:
        return list(self.values)


class _Pipeline:
    """A stand-in zero-shot pipeline: contradicts FALSE_CLAIM, neutral on anything else."""

    tokenizer = _tokenizer(64)
    model = type(
        "M",
        (),
        {
            "config": type(
                "C",
                (),
                {"id2label": {0: "entailment", 1: "neutral", 2: "contradiction"}},
            )()
        },
    )()

    def preprocess(self, sequences: Any, **kwargs: Any) -> Any:
        yield {"hypothesis": kwargs["candidate_labels"][0]}

    def forward(self, inputs: Any) -> dict[str, Any]:
        contra = inputs["hypothesis"] == FALSE_CLAIM
        return {"logits": _Logits([0.0, 0.0, 9.0] if contra else [0.0, 9.0, 0.0])}

    def postprocess(
        self, *_: Any, **__: Any
    ) -> dict[str, Any]:  # `entail` (prediction_relevance only)
        return {"scores": [0.5]}


def _local() -> WindowedEnsembleNLIJudge:
    members = []
    for model_id in ("synthetic/nli-a", "synthetic/nli-b"):
        member = DeBERTaNLIJudge(model_id)
        member._pipeline = _Pipeline()
        members.append(member)
    return WindowedEnsembleNLIJudge(members)


def _unit_and_answer(tmp_path: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    unit = build_unit(
        tmp_path,
        entities=ROSTER,
        with_outcome=True,
        docs={PRE_CUTOFF_DOC: _doc(PRE_CUTOFF_DOC, "2026-02-01", SPAN)},
    )
    answer = answer_for(entities=ROSTER)
    first = answer["entity_predictions"][0]["claims"][0]
    answer["entity_predictions"][0]["claims"] = [
        {**first, "claim": text, "span_end": len(SPAN)}
        for text in (TRUE_CLAIM, FALSE_CLAIM, WRONG_FIGURE)
    ]
    return unit, answer


def test_the_served_ensemble_cannot_answer_and_the_local_one_can(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert contradiction_supported(_served(monkeypatch)) is False
    assert contradiction_supported(_local()) is True


def test_served_preview_marks_contradiction_not_applied_and_is_not_rankable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit, answer = _unit_and_answer(tmp_path)
    result = check_answer(answer, build_unit_context(unit), _served(monkeypatch))
    assert result.contradiction_applied is False and result.rankable is False
    # Only the deterministic reason fired: the false figure-free claim is not caught here.
    assert [c.reasons for c in result.claims] == [(), (), ("unanchored",)]
    assert result.penalty_factor == pytest.approx(2 / 3)


def test_local_preview_applies_the_contradiction_check(tmp_path: pathlib.Path) -> None:
    unit, answer = _unit_and_answer(tmp_path)
    result = check_answer(answer, build_unit_context(unit), _local())
    assert result.contradiction_applied is True and result.rankable is True
    assert [c.reasons for c in result.claims] == [
        (),
        ("contradicted",),
        ("unanchored",),
    ]
    assert result.penalty_factor == pytest.approx(1 / 3)


def _score(tmp_path: pathlib.Path, judge: Any, provenance: JudgeProvenance) -> Any:
    unit, answer = _unit_and_answer(tmp_path)
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    return score_unit(
        {"unit_dir": unit, "output_dir": out}, judge=judge, judge_provenance=provenance
    )


def test_a_non_rankable_served_run_records_not_applied_and_is_no_organizer_fault(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, smoke = build_smoke_judge()
    outcome = _score(tmp_path, _served(monkeypatch), smoke)
    assert outcome.state == "participant_success" and outcome.rankable is False
    assert outcome.diagnostics["contradiction_applied"] is False
    assert outcome.diagnostics["false_claim_count"] == 1


def test_a_rankable_run_without_contradiction_is_refused(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production = JudgeProvenance(
        judge_mode="production",
        model_ids=("synthetic/nli-a",),
        model_revisions={"synthetic/nli-a": "a" * 40},
        tokenizer_digest="sha256:" + "1" * 64,
        local_cache_tree_digest="sha256:" + "2" * 64,
    )
    with pytest.raises(T4OrganizerFault, match="contradiction"):
        _score(tmp_path, _served(monkeypatch), production)
    # ...while the local ensemble under the same rankable provenance scores and applies it.
    outcome = _score(tmp_path / "local", _local(), production)
    assert outcome.diagnostics["contradiction_applied"] is True
    assert outcome.diagnostics["false_claim_count"] == 2


def test_local_preview_exempts_the_interval_level_like_the_gate(
    tmp_path: pathlib.Path,
) -> None:
    """A claim that names the unit's interval level ("our 90% band") states the task's parameter,
    not evidence: the local check exempts it exactly as the gate does (scorer 5.2.1)."""
    unit, answer = _unit_and_answer(tmp_path)
    first = answer["entity_predictions"][0]["claims"][0]
    answer["entity_predictions"][0]["claims"] = [{**first, "claim": "Our 90% band."}]
    result = check_answer(answer, build_unit_context(unit), _local())
    assert [c.status for c in result.claims] == ["neutral"]
