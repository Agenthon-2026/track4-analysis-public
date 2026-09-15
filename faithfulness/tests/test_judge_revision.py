"""A verified production spec must control the actual model-loading arguments.

These use synthetic cache bytes and a recording loader, with no model downloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from faithfulness import judge as judge_module
from qfbench2_track_analysis import judge_factory


def _spec(tmp_path: Path) -> judge_factory.JudgeSpec:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "synthetic-weights").write_bytes(b"synthetic cache contents")
    return judge_factory.JudgeSpec.from_mapping(
        {
            "model_ids": ["synthetic/nli-a", "synthetic/nli-b"],
            "model_revisions": {
                "synthetic/nli-a": "a" * 40,
                "synthetic/nli-b": "b" * 40,
            },
            "tokenizer_digest": "sha256:" + "1" * 64,
            "cache_tree_digest": judge_factory.compute_cache_tree_digest(cache),
            "cache_dir": str(cache),
        },
        source="synthetic revision fixture",
    )


def _record_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    loads: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    class Pipeline:
        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"scores": [0.75]}

    def loader(**kwargs: Any) -> Pipeline:
        loads.append(kwargs)
        return Pipeline()

    monkeypatch.setattr(judge_module, "_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(judge_module, "pipeline", loader, raising=False)
    return loads, calls


def test_production_spec_reaches_each_member_model_and_tokenizer_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path)
    loads, calls = _record_loader(monkeypatch)
    ensemble, provenance = judge_factory.build_production_judge(spec)
    assert len(loads) == len(spec.model_ids)
    assert calls == []  # Loading the organizer assets evaluates no participant text.
    assert ensemble.entail("synthetic premise", "synthetic hypothesis") == 0.75
    assert loads == [
        {
            "task": "zero-shot-classification",
            "model": model,
            "revision": spec.model_revisions[model],
            "model_kwargs": {"cache_dir": spec.cache_dir},
            "device": -1,
        }
        for model in spec.model_ids
    ]
    assert provenance.model_revisions == spec.model_revisions
    assert (
        calls
        == [
            {
                "sequences": "synthetic premise",
                "candidate_labels": ["synthetic hypothesis"],
                "hypothesis_template": "{}",
                "multi_label": True,
            }
        ]
        * 2
    )


def test_standalone_judge_keeps_optional_revision_and_positional_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads, _ = _record_loader(monkeypatch)
    member = judge_module.DeBERTaNLIJudge("synthetic/nli-a", str(tmp_path), -1)
    assert member.entail("synthetic premise", "synthetic hypothesis") == 0.75
    assert loads[0]["revision"] is None
    assert loads[0]["model_kwargs"] == {"cache_dir": str(tmp_path)}


def test_pinned_member_loads_once_without_changing_entailment_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads, calls = _record_loader(monkeypatch)
    member = judge_module.DeBERTaNLIJudge("synthetic/nli-a", revision="a" * 40)
    assert member.entail("one premise", "one hypothesis") == 0.75
    assert member.entail("another premise", "another hypothesis") == 0.75
    assert len(loads) == 1 and loads[0]["revision"] == "a" * 40
    assert [call["candidate_labels"] for call in calls] == [
        ["one hypothesis"],
        ["another hypothesis"],
    ]
