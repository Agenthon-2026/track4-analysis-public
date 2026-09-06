"""Production model loading must fail before participant evaluation begins.

All artifacts and exception text here are synthetic; no real model is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from faithfulness import judge as judge_module
from qfbench2_common.contracts import OrganizerFault
from qfbench2_track_analysis import judge_factory
from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.scoring import build_verifier

from .synthetic import answer_for, build_unit


def _production_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    cache = tmp_path / "model-cache"
    cache.mkdir()
    (cache / "synthetic-weights").write_bytes(b"synthetic model artifact")
    spec = tmp_path / "judge-spec.json"
    spec.write_text(
        json.dumps(
            {
                "model_ids": ["synthetic/nli-a", "synthetic/nli-b"],
                "model_revisions": {
                    "synthetic/nli-a": "a" * 40,
                    "synthetic/nli-b": "b" * 40,
                },
                "tokenizer_digest": "sha256:" + "1" * 64,
                "cache_tree_digest": judge_factory.compute_cache_tree_digest(cache),
                "cache_dir": str(cache),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(judge_factory.ENV_JUDGE_SPEC, str(spec))
    monkeypatch.delenv(judge_factory.ENV_JUDGE_CACHE_DIR, raising=False)
    unit = build_unit(tmp_path, with_outcome=True)
    output = tmp_path / "res"
    output.mkdir()
    (output / "answer.json").write_text(json.dumps(answer_for()), encoding="utf-8")
    return {"unit_dir": unit, "output_dir": output, "failure_map": tmp_path / "fmap"}


@pytest.mark.parametrize("failure_index", [0, 1])
@pytest.mark.parametrize("error_type", [OSError, ImportError, RuntimeError, ValueError])
def test_actual_factory_classifies_each_member_load_failure_as_organizer_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_index: int,
    error_type: type[Exception],
) -> None:
    ctx = _production_context(tmp_path, monkeypatch)
    loads: list[str] = []
    detail = "synthetic-loader-only-detail"
    cause = error_type(detail)

    def never_infer(**_: Any) -> dict[str, Any]:
        pytest.fail("participant inference started before the ensemble was loaded")

    def loader(**kwargs: Any) -> Any:
        loads.append(kwargs["model"])
        if len(loads) - 1 == failure_index:
            raise cause
        return never_infer

    monkeypatch.setattr(judge_module, "_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(judge_module, "pipeline", loader, raising=False)
    with pytest.raises(T4OrganizerFault) as failure:
        build_verifier(ctx)
    assert isinstance(failure.value, OrganizerFault)
    assert failure.value.__cause__ is cause
    assert detail not in str(failure.value)
    assert loads == ["synthetic/nli-a", "synthetic/nli-b"][: failure_index + 1]


def test_missing_tensor_runtime_is_an_organizer_fault_at_factory_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _production_context(tmp_path, monkeypatch)
    monkeypatch.setattr(judge_module, "_TRANSFORMERS_AVAILABLE", False)
    with pytest.raises(T4OrganizerFault) as failure:
        build_verifier(ctx)
    assert isinstance(failure.value.__cause__, ImportError)


def test_typed_organizer_fault_from_loader_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _production_context(tmp_path, monkeypatch)
    cause = T4OrganizerFault("synthetic typed organizer fault")

    def loader(**_: Any) -> Any:
        raise cause

    monkeypatch.setattr(judge_module, "_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(judge_module, "pipeline", loader, raising=False)
    with pytest.raises(T4OrganizerFault) as failure:
        build_verifier(ctx)
    assert failure.value is cause


def test_standalone_judge_stays_lazy_and_preserves_its_exception_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    cause = OSError("synthetic standalone load failure")

    def loader(**_: Any) -> Any:
        calls.append(True)
        raise cause

    monkeypatch.setattr(judge_module, "_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(judge_module, "pipeline", loader, raising=False)
    judge = judge_module.DeBERTaNLIJudge("synthetic/nli-a")
    assert calls == []
    with pytest.raises(OSError) as failure:
        judge.entail("synthetic premise", "synthetic hypothesis")
    assert failure.value is cause
    assert calls == [True]
