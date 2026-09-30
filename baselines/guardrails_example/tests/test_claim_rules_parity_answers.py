"""Opt-in parity over stored test answers: `check_claim_rules` vs the scorer's own gate, claim by claim.

Organizer-side only: the answers and units are not in this repository. Set
``T4_RAIL_PARITY_ANSWERS`` to a directory of ``<name>/<unit_id>/answer.json`` (or
``<unit_id>/answer.json``) files and ``T4_RAIL_PARITY_UNITS`` to the directory holding those
unit ids; the test is skipped otherwise. The judge-token cap uses the pinned judge tokenizer
(`judge_token_counter`), which must be installed. Every claim's deterministic verdict from the
rail must equal the one `score_unit` (g0-g3, contradiction not asked) records.
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from baselines.guardrails_example.citation_rail import (
    check_claim_rules,
    claim_penalty_preview,
    judge_token_counter,
)

ANSWERS = os.environ.get("T4_RAIL_PARITY_ANSWERS")
UNITS = os.environ.get("T4_RAIL_PARITY_UNITS")
pytestmark = pytest.mark.skipif(not (ANSWERS and UNITS), reason="T4_RAIL_PARITY_ANSWERS / _UNITS not set")


def _answers() -> list[Path]:
    return sorted(Path(ANSWERS).glob("**/answer.json")) if ANSWERS else []


def test_rail_equals_scorer_on_every_claim() -> None:
    from qfbench2_track_analysis.judge_factory import build_smoke_judge
    from qfbench2_track_analysis.scoring import score_unit

    counter = judge_token_counter()
    assert counter is not None, "the parity run needs the judge tokenizer installed"

    class Judge:
        def entail(self, premise: str, hypothesis: str) -> float:
            return 0.0

        def claim_tokens(self, hypothesis: str) -> int:
            return counter(hypothesis)

    _, prov = build_smoke_judge()
    paths = [p for p in _answers() if (Path(UNITS) / p.parent.name).is_dir()]
    assert paths, "no answer matched a unit"
    compared = false_total = factors = 0
    for path in paths:
        unit = Path(UNITS) / path.parent.name
        try:
            ans = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue  # not an answer at all (an empty or broken file): the scorer refuses it at g0
        out = Path(tempfile.mkdtemp(prefix="rail-parity-"))
        try:
            (out / "answer.json").write_text(json.dumps(ans))
            outcome = score_unit({"unit_dir": str(unit), "output_dir": str(out)}, judge=Judge(),
                                 judge_provenance=prov,
                                 require_outcome=(Path(unit) / "reference" / "outcome.json").is_file())
        finally:
            shutil.rmtree(out, ignore_errors=True)
        findings = check_claim_rules(ans, unit, token_counter=counter)
        refused = [f for f in findings if f.code == "unit_refused"]
        if outcome.state == "participant_failure":
            assert refused, path
            continue
        assert not refused, path
        scorer = collections.Counter(
            (fc["entity_id"], str(fc["claim"]).strip(), tuple(sorted(r for r in fc["reasons"] if r != "contradicted")))
            for fc in outcome.diagnostics.get("false_claims") or []
        )
        scorer = collections.Counter({k: v for k, v in scorer.items() if k[2]})
        texts = {r["entity_id"]: [str(c.get("claim", "")).strip() for c in r.get("claims", [])]
                 for r in ans["entity_predictions"]}
        per: dict[tuple[str, int], set[str]] = collections.defaultdict(set)
        for f in findings:
            if f.claim_index >= 0:
                per[(f.entity_id, f.claim_index)].add(f.code.removeprefix("claim_"))
        rail = collections.Counter((e, texts[e][i], tuple(sorted(r))) for (e, i), r in per.items())
        assert rail == scorer, path
        if "faithfulness_factor" in outcome.diagnostics:  # units with a mounted outcome only
            preview = claim_penalty_preview(ans, unit, token_counter=counter)
            assert preview["factor"] == outcome.diagnostics["faithfulness_factor"], path
            factors += 1
        compared += 1
        false_total += sum(rail.values())
    print(f"compared {compared} answers, {false_total} false claims, {factors} factors")
    assert compared and false_total, (compared, false_total)  # the run compared something non-trivial
