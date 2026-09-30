"""The strong-RAG baseline's answers pass the scorer's schema gate and its roster alignment.

The baseline asks the model about one entity at a time, so ranks it collected would rarely form a
permutation of 1..n, and a fallback row next to ranked rows would make the ranking partial: the
scorer refuses the whole unit for either. The baseline therefore never writes `rank`. These tests
run it on every public unit and put each answer through `g1_schema` and `align_predictions` (the
roster and `rank` checks of the domain gate), including a ranking unit whose replies are mixed:
some carry a rank, one cannot be used, one carries a boolean rank.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from baselines.strong_rag_baseline import cli
from baselines.strong_rag_baseline.client import MockModelClient
from qfbench2_track_analysis.alignment import EntityRoster, align_predictions
from qfbench2_track_analysis.scoring import _g1_schema

_REPO = Path(__file__).resolve().parents[1]
_UNITS = sorted(p.parent for p in (_REPO / "units").glob("*/task.json"))
_ENTITY_ID_RE = re.compile(r"^  entity_id: (.+)$", re.MULTILINE)


def _task(unit: Path) -> dict[str, Any]:
    task: dict[str, Any] = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    return task


def _run(unit: Path, out: Path, reply: Any) -> dict[str, Any]:
    answer: dict[str, Any] = cli.run(
        unit / "task.json", unit / "corpus", out, MockModelClient(reply=reply), 5
    )
    return answer


def _assert_accepted(answer: dict[str, Any], unit: Path) -> None:
    task = _task(unit)
    _g1_schema({"_answer": answer})
    align_predictions(
        answer,
        EntityRoster.from_task(task),
        target_type=task["target"]["type"],
        interval_level=task.get("interval_level", 0.90),
    )


def test_there_are_eleven_public_units() -> None:
    assert len(_UNITS) == 11, [u.name for u in _UNITS]


@pytest.mark.parametrize("unit", _UNITS, ids=[u.name for u in _UNITS])
def test_the_mock_answer_passes_the_gates_on_every_public_unit(
    tmp_path: Path, unit: Path
) -> None:
    _assert_accepted(_run(unit, tmp_path / "answer.json", cli._mock_reply), unit)


def test_a_ranking_unit_with_mixed_replies_passes_the_gates(tmp_path: Path) -> None:
    ranking = [u for u in _UNITS if _task(u)["target"]["type"] == "ranking"]
    assert ranking, "no public ranking unit"
    unit = ranking[0]
    ids = [str(e["entity_id"]) for e in _task(unit)["entities"]]
    assert len(ids) >= 3

    def reply(system: str, user: str) -> str:
        match = _ENTITY_ID_RE.search(user)
        entity_id = match.group(1) if match else ""
        if entity_id == ids[1]:
            return "I cannot answer that."  # unusable: this entity gets a fallback row
        parsed = json.loads(cli._mock_reply(system, user))
        parsed["rank"] = True if entity_id == ids[2] else ids.index(entity_id) + 1
        return json.dumps(parsed)

    answer = _run(unit, tmp_path / "answer.json", reply)
    _assert_accepted(answer, unit)
    assert answer["notes"]["fallback_entities"] == [ids[1]]
    assert all("rank" not in row for row in answer["entity_predictions"])
