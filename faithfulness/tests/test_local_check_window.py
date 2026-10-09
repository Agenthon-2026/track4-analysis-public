"""The local pre-check reads the judged window exactly as the gate does (scorer 5.1.2).

Before 5.1.2 `build_ensemble_judge` / `build_judge` (local and served) returned the hub's plain
`EnsembleNLIJudge`, which has no `judged_premise`, and `scoring.judged_passage` then read the
WHOLE cited passage. So on a long span whose only matching figure sits past the judge's window,
the local check anchored the claim and asked the judge (PASS) while the gate, with the windowed
production ensemble, marked it unanchored (FAIL): the pre-check said the opposite of the gate.

These tests use the in-memory word-level tokenizer of `test_judge_window.py`: no weights.
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
    EnsembleNLIJudge,
    ServedNLIJudge,
    build_ensemble_judge,
    build_judge,
    build_unit_context,
    check_answer,
)
from faithfulness.tests.test_judge_window import HYPOTHESIS, _spec, _tokenizer, _words
from qfbench2_track_analysis import judge_factory
from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.scoring import judged_passage, score_unit
from scoring.tests.synthetic import PRE_CUTOFF_DOC, StubJudge, answer_for, build_unit

ROSTER = ("SYN-A",)
CLAIM = "Revenue was 5.2 billion"  # 4 words: the premise budget is 16 - 3 - 4 = 9 words
#: The claim's only figure sits after 40 filler words, far past the 9-word premise budget.
LONG_TEXT = _words(40) + " Synthetic Issuer A reported revenue of 5.2 billion."


def _loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every member pipeline: the 16-token window, and an entailment score above tau."""

    def loader(**_: Any) -> Any:
        class Logits:
            def float(self) -> Any:  # noqa: A003 - mirrors torch.Tensor.float
                return self

            def reshape(self, *_: Any) -> Any:
                return self

            def tolist(self) -> list[Any]:
                return [0.0, 9.0, 0.0]  # neutral

        class Pipeline:
            tokenizer = _tokenizer()
            model = type(
                "M",
                (),
                {
                    "config": type(
                        "C",
                        (),
                        {
                            "id2label": {
                                0: "entailment",
                                1: "neutral",
                                2: "contradiction",
                            }
                        },
                    )()
                },
            )()

            def preprocess(self, *_: Any, **__: Any) -> Any:
                yield {}

            def forward(self, _: Any) -> dict[str, Any]:
                return {"logits": Logits()}

            def postprocess(self, *_: Any, **__: Any) -> dict[str, Any]:
                return {"scores": [0.75]}

        return Pipeline()

    monkeypatch.setattr(judge_module, "_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(judge_module, "pipeline", loader, raising=False)


def test_the_local_builders_return_the_windowed_ensemble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _loader(monkeypatch)
    monkeypatch.delenv(judge_module.ENV_JUDGE_BACKEND, raising=False)
    for ensemble in (build_ensemble_judge(), build_judge()):
        assert isinstance(ensemble, judge_factory.WindowedEnsembleNLIJudge)
        assert ensemble.judged_premise(_words(30), HYPOTHESIS) == _words(9)


def test_the_served_builder_returns_the_windowed_ensemble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[dict[str, Any]] = []

    def from_pretrained(model_id: str, **kwargs: Any) -> Any:
        loaded.append({"model_id": model_id, **kwargs})
        return _tokenizer()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", from_pretrained)
    monkeypatch.setenv(judge_module.ENV_JUDGE_BACKEND, "served")
    monkeypatch.setenv(judge_module.ENV_JUDGE_URL, "http://127.0.0.1:9")
    ensemble = build_judge(cache_dir="/synthetic-cache")
    assert isinstance(ensemble, judge_factory.WindowedEnsembleNLIJudge)
    assert all(isinstance(m, ServedNLIJudge) for m in ensemble._judges)
    # The cut is computed client-side with each member's own tokenizer; no request is made.
    assert ensemble.judged_premise(_words(30), HYPOTHESIS) == _words(9)
    assert [row["model_id"] for row in loaded] == judge_module.NLI_MODEL_IDS
    assert all(row["cache_dir"] == "/synthetic-cache" for row in loaded)


def _unit_and_answer(tmp_path: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    document = {
        "doc_id": PRE_CUTOFF_DOC,
        "doc_date": "2026-02-01",
        "source": "synthetic",
        "spans": [{"span_id": 0, "start": 0, "end": len(LONG_TEXT), "text": LONG_TEXT}],
        "text": LONG_TEXT,
    }
    unit = build_unit(
        tmp_path, entities=ROSTER, with_outcome=True, docs={PRE_CUTOFF_DOC: document}
    )
    answer = answer_for(entities=ROSTER, claim_text=CLAIM)
    answer["entity_predictions"][0]["claims"][0].update(
        span_start=0, span_end=len(LONG_TEXT)
    )
    return unit, answer


def test_check_answer_and_score_unit_agree_past_the_window(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parity pin: the CLI's judge and the production judge, one long span, one verdict."""
    _loader(monkeypatch)
    monkeypatch.delenv(judge_module.ENV_JUDGE_BACKEND, raising=False)
    unit, answer = _unit_and_answer(tmp_path)

    local = check_answer(answer, build_unit_context(unit), build_judge())

    judge, provenance = judge_factory.build_production_judge(_spec(tmp_path))
    assert provenance.rankable is True
    out = tmp_path / "res"
    out.mkdir()
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    outcome = score_unit(
        {"unit_dir": unit, "output_dir": out},
        judge=judge,
        judge_provenance=provenance,
    )

    # 5.2.0: the figure past the window is found in the WHOLE cited span, so the claim is
    # anchored in both places; both then ask the judge (neutral here) about the same window.
    assert [c.status for c in local.claims] == ["neutral"]
    assert local.penalty_factor == 1.0
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["unanchored_claim_count"] == 0
    assert outcome.diagnostics["faithfulness_factor"] == 1.0
    assert local.faithfulness == outcome.diagnostics["faithfulness"] == 1.0


def test_an_ensemble_that_drops_its_members_window_is_an_organizer_fault() -> None:
    """The plain hub ensemble over window-reading members: the judged-window guard would fail open, so refuse."""
    plain = EnsembleNLIJudge([DeBERTaNLIJudge("synthetic/nli-a")])
    with pytest.raises(T4OrganizerFault, match="judged_premise"):
        judged_passage(plain, LONG_TEXT, CLAIM)


def test_a_rankable_run_refuses_a_judge_without_a_window() -> None:
    with pytest.raises(T4OrganizerFault, match="judged_premise"):
        judged_passage(StubJudge(()), LONG_TEXT, CLAIM, require_window=True)
    # A non-rankable run keeps the documented identity for a windowless stand-in.
    assert judged_passage(StubJudge(()), LONG_TEXT, CLAIM) == LONG_TEXT


def test_score_unit_with_a_rankable_provenance_refuses_a_windowless_judge(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag reaches the gate from the judge provenance, not from the caller's goodwill."""
    _loader(monkeypatch)
    unit, answer = _unit_and_answer(tmp_path)
    _, provenance = judge_factory.build_production_judge(_spec(tmp_path))
    out = tmp_path / "res"
    out.mkdir()
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    # Refused for the first window-reading hook the gate needs (the claim token count).
    with pytest.raises(T4OrganizerFault, match="judged_premise|claim_tokens"):
        score_unit(
            {"unit_dir": unit, "output_dir": out},
            judge=StubJudge((LONG_TEXT,)),
            judge_provenance=provenance,
        )
