"""Scorer 5.3.0: a long premise is cut before it is tokenized, and the judge reads the same tokens.

The zero-shot pipeline keeps at most its window (512 tokens with the hypothesis) of a premise and
truncates the rest away, but it tokenized the whole premise first: a cited whole document of
hundreds of thousands of characters was tokenized in full on every call (three times per
citation for the `prediction_relevance` diagnostic, twice more per member for the judged window).
`bounded_premise` cuts after the 2,048th word, which still overflows any window of fewer than
2,048 tokens, at a word end, so the tokens the pipeline keeps are unchanged.

A tiny word-level tokenizer (no weights) stands in for the judge's here; the same property holds
for the pinned judge tokenizers on long real filings and on synthetic dumped documents.
"""

from __future__ import annotations

import builtins
from typing import Any, cast

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from faithfulness.judge import (
    PREMISE_TOKENIZE_WORDS,
    DeBERTaNLIJudge,
    bounded_premise,
    judged_premise_text,
)


def _tokenizer(window: int = 512) -> PreTrainedTokenizerFast:
    words = [f"w{i}" for i in range(6000)] + [f"h{i}" for i in range(600)]
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    vocab.update({w: i + 4 for i, w in enumerate(words)})
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


def _words(n: int, stem: str = "w", sep: str = " ") -> str:
    return sep.join(f"{stem}{i}" for i in range(n))


HYPOTHESIS = "h1 h2 h3 h4"


def _pipeline_ids(tok: Any, premise: str, hypothesis: str) -> list[int]:
    """The ids the zero-shot pipeline builds (`_parse_and_tokenize`: only_first, padding)."""
    ids: list[int] = tok(
        [[premise, hypothesis]], truncation="only_first", padding=True
    )["input_ids"][0]
    return ids


@pytest.mark.parametrize("sep", [" ", "\n", " \t ", "\n\n"])
def test_a_long_premise_is_cut_and_the_kept_tokens_are_identical(sep: str) -> None:
    tok = _tokenizer()
    premise = _words(5000, sep=sep)
    cut = bounded_premise(tok, premise, HYPOTHESIS)
    assert len(cut) < len(premise)
    assert len(cut.split()) == PREMISE_TOKENIZE_WORDS
    assert _pipeline_ids(tok, cut, HYPOTHESIS) == _pipeline_ids(
        tok, premise, HYPOTHESIS
    )
    assert judged_premise_text(tok, premise, HYPOTHESIS) == judged_premise_text(
        tok, cut, HYPOTHESIS
    )


def test_words_without_a_letter_or_digit_are_not_counted() -> None:
    """Punctuation-only runs may vanish in normalisation, so they never count toward the cut."""
    tok = _tokenizer()
    premise = " ".join(f"-- w{i}" for i in range(3000))
    cut = bounded_premise(tok, premise, HYPOTHESIS)
    assert sum(1 for run in cut.split() if any(c.isalnum() for c in run)) == 2048


def test_a_premise_up_to_the_word_bound_is_untouched() -> None:
    tok = _tokenizer()
    premise = _words(PREMISE_TOKENIZE_WORDS)
    assert bounded_premise(tok, premise, HYPOTHESIS) is premise


def test_a_hypothesis_that_fills_the_window_leaves_the_premise_whole() -> None:
    """The pipeline's "too short" fallback reads the whole premise untruncated, so it is not cut."""
    tok = _tokenizer(window=16)
    premise = _words(5000)
    hypothesis = _words(14, stem="h")
    assert bounded_premise(tok, premise, hypothesis) is premise
    assert judged_premise_text(tok, premise, hypothesis) is premise


def test_a_window_as_wide_as_the_bound_is_never_cut() -> None:
    """The proof needs more words than window tokens; a wider window turns the cut off."""
    tok = _tokenizer(window=PREMISE_TOKENIZE_WORDS)
    premise = _words(5000)
    assert bounded_premise(tok, premise, HYPOTHESIS) is premise


def test_no_tokenizer_means_no_cut() -> None:
    premise = _words(5000)
    assert bounded_premise(None, premise, HYPOTHESIS) is premise


class _Logits:
    def float(self) -> "_Logits":
        return self

    def reshape(self, *_: Any) -> "_Logits":
        return self

    def tolist(self) -> list[builtins.float]:
        return [0.0, 0.0, 0.0]


def test_the_member_tokenizes_only_the_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    """`entail`, `three_way` and `judged_premise` hand the pipeline the cut, never the whole."""
    tok = _tokenizer()
    seen: list[int] = []

    class Pipeline:
        tokenizer = tok
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

        def preprocess(self, inputs: str, **_: Any) -> Any:
            seen.append(len(inputs))
            yield {}

        def forward(self, _: Any) -> Any:
            return {"logits": _Logits()}

        def postprocess(self, *_: Any, **__: Any) -> dict[str, Any]:
            return {"scores": [0.5]}

    member = DeBERTaNLIJudge("synthetic/nli-a")
    member._pipeline = cast(Any, Pipeline())
    premise = _words(50_000)
    # Independent of the function under test: the first 2,048 words, as written.
    cut = _words(PREMISE_TOKENIZE_WORDS)
    assert premise.startswith(cut + " ")
    member.entail(premise, HYPOTHESIS)
    member.three_way(premise, HYPOTHESIS)
    assert seen == [len(cut), len(cut)]
    assert member.judged_premise(premise, HYPOTHESIS) == judged_premise_text(
        tok, cut, HYPOTHESIS
    )
