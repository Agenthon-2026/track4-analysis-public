"""The scorer must be self-sufficient from the driver's three-key context.

`score.py` builds exactly {unit_dir, output_dir, failure_map} and passes nothing else. Track 3's
scorer states this in its own docstring and Track 1 honours it; this module did not, and read
ctx["cutoff"], ctx["card"] and ctx["corpus_lookup"] as if a caller had supplied them -- true of
the private final scorer's harness, never true of the CodaBench driver.

Measured 2026-08-20 before that port: KeyError: 'cutoff' at the embargo gate.

**What changed in this file.** The three-key context is still the contract, and it is
still tested. What can no longer come from that context is the *judge*: `build_verifier` is the
rankable factory and it constructs the pinned production judge or refuses, so the driver-shaped
tests here exercise `build_smoke_verifier`, which is the explicitly non-rankable local factory.
The old assertions on `ctx["cutoff"]`/`ctx["corpus_lookup"]` are gone with the string cutoff and
the path-interpolating lookup they described; the trusted equivalents are `ctx["_cutoff"]` (a
`datetime.date`) and `ctx["_corpus"]` (a digest-verified index).
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import shutil
from typing import Any

import pytest

from qfbench2_common.contracts import OrganizerFault

from qfbench2_track_analysis.scoring import build_smoke_verifier

from .synthetic import answer_for, build_unit

_UNIT = pathlib.Path(__file__).resolve().parents[2] / "units/t4-EXAMPLE-eps-beat"

ANSWER = {
    "task_id": "t4-EXAMPLE-eps-beat",
    "schema_version": "3",
    "target_type": "classification",
    "entity_predictions": [
        {
            "entity_id": "AAPL",
            "label": "beat",
            "interval": {"level": 0.9, "lo": 0.4, "hi": 0.95},
            "claims": [
                {
                    "doc_id": "EDGAR_0000320193_10Q_20240202",
                    "span_start": 0,
                    "span_end": 20,
                    "claim": "Apple beat consensus.",
                }
            ],
        }
    ],
}


def _ctx(tmp_path: pathlib.Path, unit: pathlib.Path = _UNIT) -> dict[str, Any]:
    out = tmp_path / "res"
    out.mkdir(parents=True, exist_ok=True)
    (out / "answer.json").write_text(json.dumps(ANSWER))
    # EXACTLY the three keys score.py builds. Adding any more would defeat the point.
    return {"unit_dir": unit, "output_dir": out, "failure_map": tmp_path / "fmap.jsonl"}


def test_runs_on_the_drivers_three_key_context(tmp_path: pathlib.Path) -> None:
    """The regression test. Before the port this raised KeyError: 'cutoff'."""
    ctx = _ctx(tmp_path)
    verdict = build_smoke_verifier(ctx).run(ctx)
    assert verdict.admissible is True, f"labels={[lab.value for lab in verdict.labels]}"


def test_everything_trusted_is_loaded_from_unit_dir(tmp_path: pathlib.Path) -> None:
    ctx = _ctx(tmp_path)
    build_smoke_verifier(ctx).run(ctx)
    assert ctx["_cutoff"] == dt.date(2024, 3, 15)  # parsed from task.json, as a date
    assert ctx["card"]  # from card.toml
    assert ctx["_roster"].entity_ids == ("AAPL",)  # from task.json entities[]
    # from the digest-verified manifest, plus the reserved task-table document `task` (5.2.0)
    assert len(ctx["_corpus"]) == 3
    lookup = ctx["_corpus"].lookup()
    assert lookup("EDGAR_0000320193_10Q_20240202")["doc_date"] == "2024-02-02"
    with pytest.raises(KeyError):
        lookup("no-such-doc")  # the index resolves nothing it was not given


def test_no_resolved_outcome_means_no_score_rather_than_a_made_up_one(
    tmp_path: pathlib.Path,
) -> None:
    """Public practice units carry no reference/outcome.json. That must not invent a number."""
    ctx = _ctx(tmp_path)
    verdict = build_smoke_verifier(ctx).run(ctx)
    assert ctx["realized"] is None
    assert verdict.score is None
    assert verdict.detail["rankable"] is False


def test_a_resolved_outcome_produces_a_real_score(tmp_path: pathlib.Path) -> None:
    unit = tmp_path / "unit"
    shutil.copytree(_UNIT, unit)
    (unit / "reference").mkdir()
    (unit / "reference" / "outcome.json").write_text(
        json.dumps(
            {"outcomes": [{"entity_id": "AAPL", "y": 1.0, "true_label": "beat"}]}
        )
    )
    # 5.1.0: a unit with numeric truth needs its declared naive interval (interval leg).
    (unit / "reference" / "naive_answer.json").write_text(json.dumps(ANSWER))
    ctx = _ctx(tmp_path, unit=unit)
    verdict = build_smoke_verifier(ctx).run(ctx)
    assert verdict.admissible is True
    assert isinstance(verdict.score, float)


def test_a_unit_with_no_cutoff_is_an_organizer_fault_not_a_participant_failure(
    tmp_path: pathlib.Path,
) -> None:
    """A gate that cannot run is our fault. Charging it to a participant is the forbidden shape.

    The pre-fix public scorer refused the SUBMISSION here (`T4_UNFAITHFUL_CITATION`), the private
    in-repo copy PASSED it ("nothing to embargo against; treat as pass"), and the sealed final
    scorer raised `TypeError` into a handler that made the unit disappear. Three implementations,
    three polarities, on the same input.
    """
    unit = tmp_path / "nocutoff"
    shutil.copytree(_UNIT, unit)
    task = json.loads((unit / "task.json").read_text())
    task.pop("cutoff_date", None)
    (unit / "task.json").write_text(json.dumps(task))
    ctx = _ctx(tmp_path, unit=unit)
    with pytest.raises(OrganizerFault, match="cutoff_date"):
        build_smoke_verifier(ctx).run(ctx)


def test_a_missing_answer_is_a_participant_failure_with_a_bounded_code(
    tmp_path: pathlib.Path,
) -> None:
    unit = build_unit(tmp_path)
    out = tmp_path / "empty"
    out.mkdir()
    ctx = {"unit_dir": unit, "output_dir": out, "failure_map": tmp_path / "f.jsonl"}
    verdict = build_smoke_verifier(ctx).run(ctx)
    assert verdict.admissible is False
    assert verdict.detail == {"code": "no_output", "missing_count": 1}


def test_an_unparseable_answer_is_a_participant_failure_not_a_crash(
    tmp_path: pathlib.Path,
) -> None:
    unit = build_unit(tmp_path)
    out = tmp_path / "res"
    out.mkdir()
    (out / "answer.json").write_text("{not json", encoding="utf-8")
    ctx = {"unit_dir": unit, "output_dir": out, "failure_map": tmp_path / "f.jsonl"}
    verdict = build_smoke_verifier(ctx).run(ctx)
    assert verdict.admissible is False
    assert verdict.detail["code"] == "malformed_output"


def test_an_answer_for_a_different_task_is_refused(tmp_path: pathlib.Path) -> None:
    unit = build_unit(tmp_path)
    out = tmp_path / "res"
    out.mkdir()
    answer = answer_for()
    answer["task_id"] = "t4-SOME-OTHER-UNIT"
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    ctx = {"unit_dir": unit, "output_dir": out, "failure_map": tmp_path / "f.jsonl"}
    verdict = build_smoke_verifier(ctx).run(ctx)
    assert verdict.admissible is False
