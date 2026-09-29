"""A roster refusal reaches the participant with a label, not a fall-through.

Before the one-scoring-implementation change, a roster-violating submission returned
``score=None`` with every gate passed and ``labels=[]``; the driver then derived the failure code
from the empty label list and reported ``domain_gate_failed`` with nothing to say why. These tests
pin what a roster refusal carries now: the ``SCHEMA_INVALID_OUTPUT`` label (the sealed scorer's
own), the ``incomplete_output`` code for a missing entity, and -- for an unknown or
repeated entity -- the ``domain_gate_failed`` code, which is the deliberate mapping in
``codes.py``: the artifact DID match the published schema, so ``schema_invalid`` would send the
participant hunting for a schema defect that is not there. ``test_numeric_contract`` holds that
half; this file holds the label half.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from qfbench2_common.failure_labels import FailureLabel

from qfbench2_track_analysis.codes import (
    T4ParticipantFailure,
    T4Reason,
    public_code_for,
)
from qfbench2_track_analysis.scoring import _g3_domain_semantics, hydrate

from .synthetic import SUPPORTING_TEXT, StubJudge, answer_for, build_unit

ROSTER = ("SYN-A", "SYN-B")


def _ctx(tmp_path: pathlib.Path, answer: dict[str, Any]) -> dict[str, object]:
    unit = build_unit(tmp_path, entities=ROSTER)
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    ctx: dict[str, object] = {
        "unit_dir": unit,
        "output_dir": out,
        "judge": StubJudge((SUPPORTING_TEXT,)),
    }
    hydrate(ctx)
    ctx["_answer"] = answer
    return ctx


@pytest.mark.parametrize(
    "entities, reason, code",
    [
        (("SYN-A", "SYN-Z"), T4Reason.ENTITY_UNKNOWN, "domain_gate_failed"),
        (("SYN-A", "SYN-A"), T4Reason.ENTITY_DUPLICATE, "domain_gate_failed"),
        (("SYN-A",), T4Reason.ENTITY_MISSING, "incomplete_output"),
    ],
)
def test_a_roster_refusal_carries_the_schema_invalid_label(
    tmp_path: pathlib.Path, entities: tuple[str, ...], reason: T4Reason, code: str
) -> None:
    with pytest.raises(T4ParticipantFailure) as excinfo:
        _g3_domain_semantics(_ctx(tmp_path, answer_for(entities=entities)))
    assert excinfo.value.reason is reason
    assert excinfo.value.label is FailureLabel.SCHEMA_INVALID_OUTPUT
    detail = excinfo.value.public_detail()
    assert detail["code"] == code
    assert set(detail) <= {
        "code",
        "missing_count",
        "extra_count",
        "invalid_row_count",
        "expected_count",
        "observed_count",
        "violation_count",
    }


def test_the_new_entity_reason_is_a_domain_gate_code() -> None:
    assert public_code_for(T4Reason.CITATION_WRONG_ENTITY).value == "domain_gate_failed"
