"""`DeBERTaNLIJudge.three_way` / `contradiction` (scorer 5.2.0): the three-way softmax of
the SAME forward pass `entail` runs, read by label NAME from the model config.

The two pinned members order their labels differently (MoritzLaurer: entailment, neutral,
contradiction; cross-encoder: contradiction, entailment, neutral), so reading the contradiction
probability by position would silently read entailment on one of them. These tests use a stand-in
pipeline (no weights): it records the arguments `preprocess` was called with and returns fixed
logits from `forward`.
"""

from __future__ import annotations

import builtins
import math
from typing import Any

import pytest

from faithfulness.judge import DeBERTaNLIJudge

MORITZ = {0: "entailment", 1: "neutral", 2: "contradiction"}
CROSS = {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}


class _Logits:
    """Stands in for the logits tensor: `float()`, `reshape()`, `tolist()`."""

    def __init__(self, values: list[builtins.float]) -> None:
        self.values = values

    def float(self) -> _Logits:
        return self

    def reshape(self, *_: Any) -> _Logits:
        return self

    def tolist(self) -> list[builtins.float]:
        return list(self.values)


class _Pipeline:
    def __init__(self, id2label: dict[int, str], logits: list[float]) -> None:
        self.model = type(
            "M", (), {"config": type("C", (), {"id2label": id2label})()}
        )()
        self.logits = logits
        self.preprocess_calls: list[tuple[Any, dict[str, Any]]] = []
        self.forwarded: list[Any] = []

    def preprocess(self, sequences: Any, **kwargs: Any) -> Any:
        self.preprocess_calls.append((sequences, kwargs))
        yield {"pair": (sequences, kwargs["candidate_labels"][0])}

    def forward(self, inputs: Any) -> dict[str, Any]:
        self.forwarded.append(inputs)
        return {"logits": _Logits(self.logits)}


def _judge(
    id2label: dict[int, str], logits: list[float]
) -> tuple[DeBERTaNLIJudge, _Pipeline]:
    judge = DeBERTaNLIJudge("synthetic/nli")
    pipe = _Pipeline(id2label, logits)
    judge._pipeline = pipe
    return judge, pipe


def _softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    return [e / sum(exps) for e in exps]


@pytest.mark.parametrize(
    ("id2label", "positions"),
    [(MORITZ, (0, 1, 2)), (CROSS, (1, 2, 0))],
    ids=["entailment-first", "contradiction-first"],
)
def test_the_probabilities_are_read_by_label_name(
    id2label: dict[int, str], positions: tuple[int, int, int]
) -> None:
    logits = [2.0, -1.0, 0.5]
    judge, pipe = _judge(id2label, logits)
    probs = _softmax(logits)
    ent, neu, con = judge.three_way("the passage", "the claim")
    assert (ent, neu, con) == pytest.approx(
        (probs[positions[0]], probs[positions[1]], probs[positions[2]])
    )
    assert ent + neu + con == pytest.approx(1.0)
    assert judge.contradiction("the passage", "the claim") == con


def test_the_pair_is_built_exactly_as_entail_builds_it() -> None:
    """Premise = the passage, hypothesis = the claim as the one candidate label, verbatim."""
    judge, pipe = _judge(MORITZ, [0.0, 0.0, 0.0])
    judge.contradiction("the passage", "the claim")
    assert pipe.preprocess_calls == [
        (
            "the passage",
            {"candidate_labels": ["the claim"], "hypothesis_template": "{}"},
        )
    ]
    assert pipe.forwarded == [{"pair": ("the passage", "the claim")}]


def test_an_empty_side_contradicts_nothing_and_runs_no_model() -> None:
    judge, pipe = _judge(MORITZ, [0.0, 0.0, 9.0])
    assert judge.three_way("   ", "the claim") == (0.0, 1.0, 0.0)
    assert judge.contradiction("the passage", "") == 0.0
    assert pipe.forwarded == []


def test_a_model_without_the_three_nli_labels_is_refused() -> None:
    judge, _ = _judge({0: "entailment", 1: "not_entailment"}, [0.0, 0.0])
    with pytest.raises(
        RuntimeError, match="not exactly entailment / neutral / contradiction"
    ):
        judge.contradiction("the passage", "the claim")
