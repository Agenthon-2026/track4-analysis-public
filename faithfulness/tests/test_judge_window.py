"""The judged window: exactly the premise text the NLI pipeline reads (scorer 5.1.1).

The zero-shot pipeline tokenises ``[CLS] premise [SEP] hypothesis [SEP]`` and, when the pair is
longer than the tokenizer's ``model_max_length``, truncates the PREMISE only (``only_first``).
When even an empty premise cannot make room for the hypothesis the tokenizer refuses with a
"too short" error and the pipeline re-tokenises with no truncation at all. `judged_premise_text`
returns the premise prefix that survives, so the figure check can read the same text.

These tests use a tiny word-level fast tokenizer built in memory: no model weights, no download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from faithfulness import judge as judge_module
from faithfulness.judge import DeBERTaNLIJudge, judged_premise_text
from qfbench2_track_analysis import judge_factory
from qfbench2_track_analysis.codes import T4OrganizerFault

WINDOW = 16


def _tokenizer(window: int = WINDOW) -> PreTrainedTokenizerFast:
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    core.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        model_max_length=window,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def _words(n: int, stem: str = "w") -> str:
    return " ".join(f"{stem}{i}" for i in range(n))


HYPOTHESIS = (
    "revenue was 5.2 billion"  # 4 words -> the premise budget is 16 - 3 - 4 = 9
)


def test_a_pair_inside_the_window_returns_the_premise_unchanged() -> None:
    premise = _words(9)
    assert judged_premise_text(_tokenizer(), premise, HYPOTHESIS) == premise


def test_a_long_premise_is_cut_to_what_only_first_truncation_keeps() -> None:
    tok = _tokenizer()
    premise = _words(40) + " the figure 5.2 sits past the window"
    cut = judged_premise_text(tok, premise, HYPOTHESIS)
    assert cut == _words(9)
    assert premise.startswith(cut)
    # The cut, untruncated, is token for token the pair the pipeline builds by truncating.
    truncated = tok([[premise, HYPOTHESIS]], truncation="only_first")["input_ids"]
    untruncated = tok([[cut, HYPOTHESIS]], truncation=False)["input_ids"]
    assert untruncated == truncated
    assert len(truncated[0]) == WINDOW


def test_a_hypothesis_that_fills_the_window_leaves_the_premise_whole() -> None:
    """The pipeline's fallback: only_first cannot fit the pair, so nothing is truncated and the
    judge reads the whole premise. The cut mirrors that rather than inventing a shorter text."""
    tok = _tokenizer()
    premise = _words(5)
    hypothesis = _words(20, stem="h")
    with pytest.raises(Exception, match="too short"):
        tok([[premise, hypothesis]], truncation="only_first")
    assert judged_premise_text(tok, premise, hypothesis) == premise


def test_a_member_reads_its_own_loaded_tokenizer() -> None:
    class Pipeline:
        tokenizer = _tokenizer()

    member = DeBERTaNLIJudge("synthetic/nli-a")
    member._pipeline = cast(Any, Pipeline())
    assert member.judged_premise(_words(30), HYPOTHESIS) == _words(9)


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
        source="synthetic window fixture",
    )


def _loader(monkeypatch: pytest.MonkeyPatch, windows: dict[str, int]) -> None:
    def loader(**kwargs: Any) -> Any:
        class Pipeline:
            tokenizer = _tokenizer(windows[kwargs["model"]])

            def preprocess(self, *_: Any, **__: Any) -> Any:
                yield {}

            def forward(self, inputs: Any) -> Any:
                return inputs

            def postprocess(self, *_: Any, **__: Any) -> dict[str, Any]:
                return {"scores": [0.75]}

        return Pipeline()

    monkeypatch.setattr(judge_module, "_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(judge_module, "pipeline", loader, raising=False)


def test_the_production_ensemble_exposes_its_members_common_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _loader(monkeypatch, {"synthetic/nli-a": WINDOW, "synthetic/nli-b": WINDOW})
    ensemble, _ = judge_factory.build_production_judge(_spec(tmp_path))
    assert ensemble.judged_premise(_words(30), HYPOTHESIS) == _words(9)
    assert ensemble.judged_premise(_words(3), HYPOTHESIS) == _words(3)


def test_members_that_would_read_different_text_are_an_organizer_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is no single 'text the judge read' when the members' windows differ, so the figure
    check has nothing honest to read: refuse rather than pick one."""
    _loader(monkeypatch, {"synthetic/nli-a": WINDOW, "synthetic/nli-b": WINDOW + 4})
    ensemble, _ = judge_factory.build_production_judge(_spec(tmp_path))
    assert ensemble.judged_premise(_words(3), HYPOTHESIS) == _words(3)
    with pytest.raises(T4OrganizerFault, match="window"):
        ensemble.judged_premise(_words(30), HYPOTHESIS)
