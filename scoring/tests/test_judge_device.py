"""Scorer 5.3.0: the production judge's device, and its residency in the process.

* `QFBENCH2_T4_JUDGE_DEVICE` = ``auto`` (default) | ``cpu`` | ``cuda``. ``auto`` is the GPU when
  torch sees one, else the CPU; ``cuda`` with no GPU is an organizer fault, never a quiet CPU
  run; anything else is refused.
* Each member is loaded once per process and moved to the GPU once; every later
  `build_production_judge` / `build_verifier` call in the process reuses it (the platform driver
  calls `build_verifier` once per unit). The cache digest is checked at that first load; a later
  call for a different judge (spec, cache, device, revisions) is an organizer fault.

Everything here runs without a GPU and without model weights: the device query is a stub, the
pipelines are recording stand-ins, and `torch` is replaced by a stand-in module for the move,
so no tensor ever reaches a device. The one real-GPU parity check at the bottom runs only when
asked for (`QFBENCH2_T4_GPU_TESTS=1`) on a machine with the judge weights.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from faithfulness import judge as judge_module
from qfbench2_track_analysis import judge_factory
from qfbench2_track_analysis.codes import T4OrganizerFault


# --- device selection --------------------------------------------------------------------
def test_auto_without_a_gpu_is_the_cpu() -> None:
    assert (
        judge_factory.resolve_judge_device("auto", cuda_available=lambda: False)
        == "cpu"
    )


def test_auto_with_a_gpu_is_the_gpu() -> None:
    assert (
        judge_factory.resolve_judge_device("auto", cuda_available=lambda: True)
        == "cuda"
    )


def test_cuda_without_a_gpu_is_an_organizer_fault() -> None:
    with pytest.raises(T4OrganizerFault, match="no CUDA device"):
        judge_factory.resolve_judge_device("cuda", cuda_available=lambda: False)


def test_cpu_is_the_cpu_even_with_a_gpu() -> None:
    assert (
        judge_factory.resolve_judge_device("cpu", cuda_available=lambda: True) == "cpu"
    )


@pytest.mark.parametrize("value", ["gpu", "cuda:1", "0", "true"])
def test_an_unknown_device_is_refused(value: str) -> None:
    with pytest.raises(T4OrganizerFault):
        judge_factory.resolve_judge_device(value, cuda_available=lambda: True)


def test_the_device_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(judge_factory.ENV_JUDGE_DEVICE, "")
    assert judge_factory.resolve_judge_device(cuda_available=lambda: False) == "cpu"
    monkeypatch.setenv(judge_factory.ENV_JUDGE_DEVICE, " CUDA ")
    with pytest.raises(T4OrganizerFault):
        judge_factory.resolve_judge_device(cuda_available=lambda: False)


def test_the_device_is_not_a_judge_spec_key(tmp_path: Path) -> None:
    """A speed setting, not the judge's identity: the spec's closed key set still refuses it."""
    with pytest.raises(T4OrganizerFault, match="unknown judge spec keys"):
        judge_factory.JudgeSpec.from_mapping(
            {**_raw_spec(tmp_path), "device": "cuda"}, source="synthetic"
        )


# --- residency: loaded once, moved once, reused ----------------------------------------
def _raw_spec(tmp_path: Path) -> dict[str, Any]:
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    (cache / "synthetic-weights").write_bytes(b"synthetic cache contents")
    return {
        "model_ids": ["synthetic/nli-a", "synthetic/nli-b"],
        "model_revisions": {"synthetic/nli-a": "a" * 40, "synthetic/nli-b": "b" * 40},
        "tokenizer_digest": "sha256:" + "1" * 64,
        "cache_tree_digest": judge_factory.compute_cache_tree_digest(cache),
        "cache_dir": str(cache),
    }


class _Model:
    def __init__(self, moves: list[str]) -> None:
        self.moves = moves

    def to(self, device: Any) -> _Model:
        self.moves.append(str(device))
        return self

    def parameters(self) -> Any:
        yield types.SimpleNamespace(dtype="float16")


def _gpu_loader(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[str]]:
    """Recording pipelines, a GPU the device query reports, and a stand-in `torch`."""
    loads: list[str] = []
    moves: list[str] = []

    def loader(**kwargs: Any) -> Any:
        loads.append(kwargs["model"])
        return types.SimpleNamespace(model=_Model(moves), device=None)

    monkeypatch.setattr(judge_module, "_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(judge_module, "pipeline", loader, raising=False)
    monkeypatch.setattr(judge_factory, "_cuda_available", lambda: True)
    monkeypatch.setenv(judge_factory.ENV_JUDGE_DEVICE, "auto")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(device=lambda name: f"device:{name}"),
    )
    return loads, moves


def test_each_member_is_loaded_and_moved_once_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads, moves = _gpu_loader(monkeypatch)
    spec = judge_factory.JudgeSpec.from_mapping(_raw_spec(tmp_path), source="synthetic")
    first, _ = judge_factory.build_production_judge(spec)
    second, _ = judge_factory.build_production_judge(spec)
    third, _ = judge_factory.build_production_judge(spec)
    assert first is second is third
    assert loads == ["synthetic/nli-a", "synthetic/nli-b"]
    assert moves == ["device:cuda:0", "device:cuda:0"]  # one move per member, ever
    for member in first._judges:
        assert member.device == 0
        assert member._pipeline.device == "device:cuda:0"


def test_the_official_factory_reuses_the_judge_across_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver calls `build_verifier` once per unit; the weights load and move once."""
    from qfbench2_track_analysis.scoring import build_verifier

    loads, moves = _gpu_loader(monkeypatch)
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(_raw_spec(tmp_path)), encoding="utf-8")
    monkeypatch.setenv(judge_factory.ENV_JUDGE_SPEC, str(spec_file))
    monkeypatch.delenv(judge_factory.ENV_JUDGE_CACHE_DIR, raising=False)
    judges = []
    for unit in range(3):
        ctx: dict[str, Any] = {"unit_dir": tmp_path / f"unit-{unit}"}
        build_verifier(ctx)
        judges.append(ctx["judge"])
    assert judges[0] is judges[1] is judges[2]
    assert len(loads) == 2 and len(moves) == 2


def test_the_cache_digest_is_checked_at_the_first_load_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full digest is checked when the weights are loaded; re-hashing the files later in the
    process cannot protect weights already in memory, so a reuse does not re-hash."""
    _gpu_loader(monkeypatch)
    spec = judge_factory.JudgeSpec.from_mapping(_raw_spec(tmp_path), source="synthetic")
    hashed: list[str] = []
    real = judge_factory.compute_cache_tree_digest

    def counting(path: Any) -> str:
        hashed.append(str(path))
        return real(path)

    monkeypatch.setattr(judge_factory, "compute_cache_tree_digest", counting)
    first, _ = judge_factory.build_production_judge(spec)
    second, _ = judge_factory.build_production_judge(spec)
    assert first is second
    assert len(hashed) == 1


def test_a_bad_cache_is_refused_at_the_first_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads, _ = _gpu_loader(monkeypatch)
    spec = judge_factory.JudgeSpec.from_mapping(_raw_spec(tmp_path), source="synthetic")
    (tmp_path / "cache" / "synthetic-weights").write_bytes(b"swapped weights")
    with pytest.raises(T4OrganizerFault, match="does not match the digest"):
        judge_factory.build_production_judge(spec)
    assert loads == []


@pytest.mark.parametrize("change", ["cache_dir", "device", "revision"])
def test_a_different_judge_in_the_same_process_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """One process scores with one judge: a later call for another cache, device or revision is
    an organizer fault, never the loaded judge handed back and never a second load."""
    loads, _ = _gpu_loader(monkeypatch)
    raw = _raw_spec(tmp_path)
    spec = judge_factory.JudgeSpec.from_mapping(raw, source="synthetic")
    judge_factory.build_production_judge(spec)
    kwargs: dict[str, Any] = {}
    other = spec
    if change == "cache_dir":
        copy = tmp_path / "copy"
        copy.mkdir()
        (copy / "synthetic-weights").write_bytes(b"synthetic cache contents")
        kwargs["cache_dir"] = copy
    elif change == "device":
        monkeypatch.setenv(judge_factory.ENV_JUDGE_DEVICE, "cpu")
    else:
        revisions = {**raw["model_revisions"], "synthetic/nli-b": "c" * 40}
        other = judge_factory.JudgeSpec.from_mapping(
            {**raw, "model_revisions": revisions}, source="synthetic"
        )
    with pytest.raises(T4OrganizerFault, match="already loaded"):
        judge_factory.build_production_judge(other, **kwargs)
    assert len(loads) == 2  # the first judge's two members only


def test_a_cpu_judge_is_never_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads, moves = _gpu_loader(monkeypatch)
    monkeypatch.setenv(judge_factory.ENV_JUDGE_DEVICE, "cpu")
    spec = judge_factory.JudgeSpec.from_mapping(_raw_spec(tmp_path), source="synthetic")
    judge, _ = judge_factory.build_production_judge(spec)
    assert len(loads) == 2 and moves == []
    assert all(member.device == -1 for member in judge._judges)


def test_cpu_inputs_are_handed_to_the_pipeline_untouched() -> None:
    member = judge_module.DeBERTaNLIJudge("synthetic/nli-a")
    inputs = {"input_ids": object()}
    assert member._to_device(inputs) is inputs


# --- real GPU parity (opt-in) -------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("QFBENCH2_T4_GPU_TESTS") != "1",
    reason="real-GPU parity runs only when asked for (QFBENCH2_T4_GPU_TESTS=1, judge weights)",
)
def test_the_gpu_judge_matches_the_cpu_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """`auto` on a GPU machine picks the GPU, and the probabilities equal the CPU judge's within
    2e-4 (measured bound on the pinned pair about 1.3e-4: the fp16 member computes slightly
    differently on the two devices). A few synthetic pairs only; the GPU carries other load."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    pairs = [
        ("Revenue rose 12% to $4.1 billion.", "Revenue grew year over year."),
        ("Revenue rose 12% to $4.1 billion.", "Revenue fell 12%."),
        ("The board kept the dividend unchanged.", "The company raised its dividend."),
    ]
    monkeypatch.setenv(judge_factory.ENV_JUDGE_DEVICE, "cpu")
    cpu, _ = judge_factory.build_production_judge()
    # One judge per process: a second device needs a fresh process cache.
    monkeypatch.setattr(judge_factory, "_PROCESS_JUDGES", {})
    monkeypatch.setenv(judge_factory.ENV_JUDGE_DEVICE, "auto")
    gpu, _ = judge_factory.build_production_judge()
    assert cpu is not gpu
    assert all(member.device == 0 for member in gpu._judges)
    for premise, hypothesis in pairs:
        assert gpu.contradiction(premise, hypothesis) == pytest.approx(
            cpu.contradiction(premise, hypothesis), abs=2e-4
        )
        assert gpu.entail(premise, hypothesis) == pytest.approx(
            cpu.entail(premise, hypothesis), abs=2e-4
        )
