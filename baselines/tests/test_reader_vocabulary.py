"""Minimal-baseline labels must belong to each task's published vocabulary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baselines.baseline_agent.cli import run
from baselines.baseline_agent.reader import predict_entity


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["negative", "positive"], "negative"),
        (["positive", "inline", "negative"], "inline"),
        (["miss", "beat"], "miss"),
    ],
)
def test_fallback_uses_the_declared_vocabulary(
    labels: list[str], expected: str
) -> None:
    assert predict_entity({}, "", labels=labels)["label"] == expected


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        (1.5, {"label": "beat", "point_forecast": 1.5, "lo": 1.35, "hi": 1.65}),
        (0.5, {"label": "miss", "point_forecast": 0.5, "lo": 0.45, "hi": 0.55}),
        (1.0, {"label": "inline", "point_forecast": 1.0, "lo": 0.9, "hi": 1.1}),
    ],
)
def test_compatible_eps_vocabulary_preserves_every_prediction_field(
    reported: float,
    expected: dict[str, object],
) -> None:
    entity = {"consensus_eps": 1.0, "threshold_pct": 0.05}
    text = f"earnings per share of ${reported:.2f}"
    old = predict_entity(entity, text)
    assert old == expected
    assert predict_entity(entity, text, labels=["beat", "miss", "inline"]) == old


@pytest.mark.parametrize("target_type", ["classification", "regression", "ranking"])
def test_cli_forwards_vocabulary_and_preserves_target_type(
    tmp_path: Path,
    target_type: str,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.json").write_text(
        json.dumps(
            {
                "doc_id": "doc",
                "doc_date": "2024-01-01",
                "text": "Company A published its report before the cutoff.",
            }
        )
    )
    target: dict[str, object] = {"type": target_type}
    if target_type == "classification":
        target["labels"] = ["negative", "positive"]
    task_path = tmp_path / "task.json"
    task_path.write_text(
        json.dumps(
            {
                "task_id": "synthetic-label-contract",
                "cutoff_date": "2024-01-31",
                "target": target,
                "entities": [{"entity_id": "A", "name": "Company A"}],
            }
        )
    )
    answer = run(task_path, corpus, tmp_path / "answer.json")
    assert answer["target_type"] == target_type
    prediction = answer["entity_predictions"][0]
    assert prediction["label"] == (
        "negative" if target_type == "classification" else "inline"
    )
    assert prediction["point_forecast"] == 0.0
    assert prediction["claims"]


def test_every_public_practice_task_uses_its_own_labels(tmp_path: Path) -> None:
    units = Path(__file__).resolve().parents[2] / "units"
    tasks = sorted(units.glob("*/task.json"))
    assert tasks, "No public practice tasks found; this check cannot run."
    for task_path in tasks:
        task = json.loads(task_path.read_text())
        answer = run(
            task_path,
            task_path.parent / "corpus",
            tmp_path / task_path.parent.name / "answer.json",
        )
        labels = task.get("target", {}).get("labels")
        if labels:
            assert all(
                prediction["label"] in labels
                for prediction in answer["entity_predictions"]
            ), f"{task_path.parent.name}: baseline emitted an off-vocabulary label"
