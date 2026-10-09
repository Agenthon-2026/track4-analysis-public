"""Regression tests for :meth:`DeBERTaNLIJudge.entail`.

These pin the NLI call convention rather than model behaviour, so they need no
weights and no network.  They exist because the judge previously passed
``candidate_labels=["entailment", "neutral", "contradiction"]``, which classifies
the premise against those three literal words and never uses the claim -- making
every claim against a given premise score identically and rendering the
citation-faithfulness component of the composite inert.
"""

from __future__ import annotations

import inspect
from typing import Any, cast

import pytest

from faithfulness.judge import DeBERTaNLIJudge, NLI_MODEL_IDS


class _RecordingPipeline:
    """Stand-in for the HF zero-shot pipeline that records how it was called.

    Scorer 5.3.0: `entail` runs the pipeline's own three stages (``preprocess`` -> ``forward``
    -> ``postprocess``, what its ``__call__`` runs for one input), so one call is recorded as the
    arguments of all three, under the names ``__call__`` takes."""

    def __init__(self, score_by_hypothesis: dict[str, float]) -> None:
        self.score_by_hypothesis = score_by_hypothesis
        self.calls: list[dict[str, Any]] = []

    def preprocess(
        self, inputs: Any, candidate_labels: Any = None, hypothesis_template: str = ""
    ) -> Any:
        yield {
            "sequences": inputs,
            "candidate_labels": list(candidate_labels),
            "hypothesis_template": hypothesis_template,
        }

    def forward(self, inputs: Any) -> Any:
        return inputs

    def postprocess(self, outputs: Any, multi_label: bool = False) -> dict[str, Any]:
        (call,) = outputs
        self.calls.append({**call, "multi_label": multi_label})
        labels: list[str] = call["candidate_labels"]
        return {
            "sequence": call["sequences"],
            "labels": labels,
            "scores": [self.score_by_hypothesis.get(lbl, 0.0) for lbl in labels],
        }


def _judge(pipeline: _RecordingPipeline) -> DeBERTaNLIJudge:
    judge = DeBERTaNLIJudge(model_id=NLI_MODEL_IDS[0])
    # The stand-in is duck-typed, not a ZeroShotClassificationPipeline subclass, so the
    # assignment is deliberately unchecked. Note this is also why these tests could not
    # have caught a genuine signature mismatch: _RecordingPipeline takes **kwargs and
    # would swallow any argument name. The real signature is pinned by the annotation on
    # DeBERTaNLIJudge._pipeline and by
    # test_call_kwargs_bind_against_the_real_pipeline_signature below.
    judge._pipeline = cast(Any, pipeline)
    return judge


#: Placeholder for the bound-method ``self`` when binding an unbound ``__call__``.
_UNBOUND_SELF: Any = None

PREMISE = "Revenue for the quarter rose 12% year over year to $4.1 billion."
SUPPORTED = "Revenue grew year over year."
CONTRADICTED = "Revenue declined year over year."


def test_claim_is_passed_as_the_candidate_label() -> None:
    """The claim must reach the model; that is the whole point of the judge."""
    pipeline = _RecordingPipeline({SUPPORTED: 0.97})
    _judge(pipeline).entail(PREMISE, SUPPORTED)

    (call,) = pipeline.calls
    assert call["sequences"] == PREMISE
    assert call["candidate_labels"] == [SUPPORTED]
    assert call["hypothesis_template"] == "{}"
    # Keep the shipped two-way call explicit; False with one candidate also takes
    # the pinned Transformers entailment-versus-contradiction path.
    assert call["multi_label"] is True


def test_entailment_score_is_the_single_candidate_score() -> None:
    pipeline = _RecordingPipeline({SUPPORTED: 0.97})
    assert _judge(pipeline).entail(PREMISE, SUPPORTED) == 0.97


def test_supported_and_contradicted_claims_score_differently() -> None:
    """The regression: the old call returned the same score for both."""
    pipeline = _RecordingPipeline({SUPPORTED: 0.97, CONTRADICTED: 0.01})
    judge = _judge(pipeline)

    supported = judge.entail(PREMISE, SUPPORTED)
    contradicted = judge.entail(PREMISE, CONTRADICTED)

    assert supported > contradicted
    # tau_citation defaults to 0.5: one citation is admissible, the other is not.
    assert supported > 0.5 > contradicted


def test_empty_premise_or_hypothesis_scores_zero() -> None:
    pipeline = _RecordingPipeline({SUPPORTED: 0.97})
    judge = _judge(pipeline)

    assert judge.entail("", SUPPORTED) == 0.0
    assert judge.entail(PREMISE, "   ") == 0.0
    assert pipeline.calls == []


def test_call_kwargs_bind_against_the_real_pipeline_signature() -> None:
    """The kwargs ``entail`` uses must bind to the REAL upstream ``__call__``.

    Scorer 5.3.0 runs the three stages ``__call__`` runs; the next test binds those too. This
    one keeps the meaning of each argument pinned to the documented entry point.

    Every other test here drives ``_RecordingPipeline``, which takes ``**kwargs`` and so
    accepts any argument name at all -- it cannot tell ``sequences=`` from a typo or from an
    upstream rename. That is exactly how a signature mismatch hides: the suite stays green
    while the real pipeline would raise ``TypeError`` on the first scored submission, in the
    one configuration (``transformers`` actually installed) that only the scoring fleet runs.

    ``ZeroShotClassificationPipeline.__call__`` names its first parameter ``sequences``,
    while the base ``Pipeline.__call__`` names it ``inputs``. Binding against the concrete
    subclass pins that distinction to the transformers version actually in use.
    """
    pytest.importorskip(
        "transformers", reason="signature check needs the real transformers package"
    )
    from transformers.pipelines.zero_shot_classification import (
        ZeroShotClassificationPipeline,
    )

    signature = inspect.signature(ZeroShotClassificationPipeline.__call__)

    # The exact kwargs faithfulness/judge.py::DeBERTaNLIJudge.entail passes.
    signature.bind(
        _UNBOUND_SELF,
        sequences=PREMISE,
        candidate_labels=[SUPPORTED],
        hypothesis_template="{}",
        multi_label=True,
    )


def test_stage_kwargs_bind_against_the_real_pipeline_stages() -> None:
    """Scorer 5.3.0: the arguments `entail` and `three_way` pass to ``preprocess`` and
    ``postprocess`` bind to the REAL zero-shot pipeline stages, so a stand-in that accepts
    anything cannot hide an upstream rename."""
    pytest.importorskip(
        "transformers", reason="signature check needs the real transformers package"
    )
    from transformers.pipelines.zero_shot_classification import (
        ZeroShotClassificationPipeline,
    )

    inspect.signature(ZeroShotClassificationPipeline.preprocess).bind(
        _UNBOUND_SELF, PREMISE, candidate_labels=[SUPPORTED], hypothesis_template="{}"
    )
    inspect.signature(ZeroShotClassificationPipeline.postprocess).bind(
        _UNBOUND_SELF, [], multi_label=True
    )
