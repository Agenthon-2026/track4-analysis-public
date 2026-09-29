"""Every documented number is executed against the scorer, not matched as a string.

`test_docs_do_not_invent_scoring_terms.py` asks whether a doc names a term no Python file
implements. That catches invented vocabulary and nothing else, and it is keyed to this repo's
tree, so it never opens the hub's Track-4 starter pack at all. Four published claims were wrong
in ways it structurally could not see — each named a real term and stated a false consequence:

* the starter pack said a constant `point_forecast` "scores full marks — silently" and is
  "indistinguishable" from a correct ranking. It scores 0.5 against a correct ranking's 1.0.
* the starter pack said an ineligible unit "costs more than the worst admissible answer you
  could have submitted instead". Both are exactly W.
* `SUBMISSION_CLI.md` said an image that will not accept the verb scores "zero on that unit".
  It scores W (-0.27 through scorer 5.0.0; 0.0, the bottom of the domain, from 5.1.0).
* `CONCEPTS.md` said omitted rows are "scored worst-case for that row". A participant cannot
  reach that path: roster exactness refuses the submission first.

So this module does the opposite of a string allowlist. Each entry below is a CLAIM: a
scenario the scorer actually runs, the numbers the measurement must produce, and the document
passage that states them. A claim fails if the scorer's behaviour moves (the measurement no
longer matches) OR if the prose drifts away from what was measured (the anchor is gone, or the
stated number is missing from the passage). Neither direction can rot silently, and adding a
new documented number means adding a scenario that produces it — not a string to a list.
"""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import UnitOutcome, score_unit

from .synthetic import (
    SUPPORTING_TEXT,
    StubJudge,
    answer_for,
    build_unit,
    outcome_for,
    write_naive_answer,
)

_REPO = pathlib.Path(__file__).resolve().parents[2]
_HUB_ENV = "QFBENCH2_HUB_ROOT"
_ENTITIES = ("SYN-A", "SYN-B", "SYN-C", "SYN-D")


def _hub_root() -> pathlib.Path:
    """A checkout of the published hub (`Agenthon-2026/Agenthon2026-public`), which carries the
    Track-4 starter pack.

    A missing hub is a FAILURE, not a skip. Half of the Track-4 claims a participant reads live
    in the hub's starter pack, and a test that quietly stops checking them is the silent-pass
    class this repo has closed everywhere else. The hub is looked for beside this repository
    (`../Agenthon2026-public`) unless the environment variable names another checkout.
    """
    override = os.environ.get(_HUB_ENV)
    candidate = (
        pathlib.Path(override) if override else _REPO.parent / "Agenthon2026-public"
    )
    if not (candidate / "starter-packs" / "track4" / "AGENTS.md").is_file():
        raise AssertionError(
            f"hub starter pack not found under {candidate}. Check out "
            f"Agenthon-2026/Agenthon2026-public beside this repo, or set {_HUB_ENV}. "
            "This test is not skipped when the hub is absent: the participant-facing claims it "
            "verifies would then go unchecked."
        )
    return candidate


def _doc_text(where: str, relative: str) -> str:
    root = _REPO if where == "track" else _hub_root()
    return (root / relative).read_text(encoding="utf-8")


def _passage(text: str, anchor: str, *, doc: str) -> str:
    """The anchor's own passage: from the anchor to the next blank line after its block."""
    index = text.find(anchor)
    if index < 0:
        raise AssertionError(
            f"{doc}: the passage this claim verifies is gone. Anchor not found:\n  {anchor!r}\n"
            "If the wording moved, update the anchor here in the same change; if the claim "
            "itself was withdrawn, delete this entry rather than loosening the anchor."
        )
    tail = text[index:]
    # Tables and their lead-in are one passage: keep consuming while lines are non-empty or
    # the next non-empty line is still a table row.
    end, cursor = len(tail), 0
    while cursor < len(tail):
        stop = tail.find("\n\n", cursor)
        if stop < 0:
            break
        following = tail[stop + 2 : stop + 3]
        if following != "|":
            end = stop
            break
        cursor = stop + 2
    return tail[:end]


def _score(
    tmp_path: pathlib.Path,
    *,
    target_type: str,
    answer: dict[str, Any] | None,
    ys: tuple[object, ...] | None = None,
    raw: str | None = None,
) -> UnitOutcome:
    unit = build_unit(tmp_path, target_type=target_type, entities=_ENTITIES)
    outcome = outcome_for(_ENTITIES)
    outcome["target_type"] = target_type
    if ys is not None:
        for row, y in zip(outcome["outcomes"], ys):
            row["y"] = y
    (unit / "reference").mkdir(exist_ok=True)
    (unit / "reference" / "outcome.json").write_text(
        json.dumps(outcome), encoding="utf-8"
    )
    # Every unit with numeric truth carries its declared naive rule: regression reads its point
    # forecasts, and from 5.1.0 every type reads its interval (band [0.5, 3.5], answer_for's default).
    write_naive_answer(unit, _ENTITIES, point_forecast=2.5, target_type=target_type)
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    if raw is not None:
        (out / "answer.json").write_text(raw, encoding="utf-8")
    elif answer is not None:
        (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out},
        judge=StubJudge((SUPPORTING_TEXT,)),
        judge_provenance=provenance,
    )


def _answer(
    target_type: str, forecasts: tuple[float, ...], **kwargs: Any
) -> dict[str, Any]:
    answer = answer_for(_ENTITIES, **kwargs)
    answer["target_type"] = target_type
    for index, (row, forecast) in enumerate(
        zip(answer["entity_predictions"], forecasts)
    ):
        row["point_forecast"] = float(forecast)
        row["rank"] = index + 1
    return answer


# --------------------------------------------------------------------------- measurements
_YS = (1.0, 2.0, 3.0, 4.0)


def _ranking_quality(tmp_path: pathlib.Path) -> dict[str, Any]:
    def pq(forecasts: tuple[float, ...]) -> float:
        outcome = _score(
            tmp_path / str(forecasts),
            target_type="ranking",
            answer=_answer("ranking", forecasts),
            ys=_YS,
        )
        return round(float(outcome.diagnostics["predictive_quality"]), 4)

    return {
        "correct": pq((1.0, 2.0, 3.0, 4.0)),
        "constant": pq((5.0, 5.0, 5.0, 5.0)),
        "reversed": pq((4.0, 3.0, 2.0, 1.0)),
    }


def _regression_composites(tmp_path: pathlib.Path) -> dict[str, Any]:
    def score(name: str, forecasts: tuple[float, ...], lo: float, hi: float) -> float:
        outcome = _score(
            tmp_path / name,
            target_type="regression",
            answer=_answer("regression", forecasts, lo=lo, hi=hi),
            ys=_YS,
        )
        return round(float(outcome.score), 4)

    perfect = score("perfect", _YS, 0.5, 4.5)
    honest = score("honest", (0.0, 0.0, 0.0, 0.0), -1000.0, 1000.0)
    precise = score("precise", (0.0, 0.0, 0.0, 0.0), 0.0, 0.0)
    short = _answer("regression", (0.0, 0.0, 0.0, 0.0), lo=0.0, hi=0.0)
    short["entity_predictions"] = short["entity_predictions"][:-1]
    omitted = _score(
        tmp_path / "omitted", target_type="regression", answer=short, ys=_YS
    )
    return {
        "perfect": perfect,
        "honest": honest,
        "worst_admissible": precise,
        "inadmissible": round(float(omitted.score), 4),
        "inadmissible_code": omitted.failure_code,
    }


def _no_output_and_malformed(tmp_path: pathlib.Path) -> dict[str, Any]:
    absent = _score(tmp_path / "absent", target_type="regression", answer=None, ys=_YS)
    empty = _score(
        tmp_path / "empty", target_type="regression", answer=None, raw="", ys=_YS
    )
    return {
        "no_output_code": absent.failure_code,
        "no_output_score": round(float(absent.score), 4),
        "malformed_code": empty.failure_code,
        "malformed_score": round(float(empty.score), 4),
    }


def _roster_exactness(tmp_path: pathlib.Path) -> dict[str, Any]:
    full = _score(
        tmp_path / "full",
        target_type="ranking",
        answer=_answer("ranking", _YS, lo=0.5, hi=4.5),
        ys=_YS,
    )
    short_answer = _answer("ranking", _YS, lo=0.5, hi=4.5)
    short_answer["entity_predictions"] = short_answer["entity_predictions"][:-1]
    short = _score(
        tmp_path / "short", target_type="ranking", answer=short_answer, ys=_YS
    )
    return {
        "full": round(float(full.score), 4),
        "omitted": round(float(short.score), 4),
        "omitted_code": short.failure_code,
    }


def _unresolvable_doc_id(tmp_path: pathlib.Path) -> dict[str, Any]:
    answer = _answer("regression", _YS)
    for row in answer["entity_predictions"]:
        row["claims"][0]["doc_id"] = "SYNTHDOC_NOT_DECLARED"
    outcome = _score(
        tmp_path / "unknown", target_type="regression", answer=answer, ys=_YS
    )
    return {
        "code": outcome.failure_code,
        "score": round(float(outcome.score), 4),
        # NOT `"faithfulness" in diagnostics`. The failure path carries the key through so an
        # operator can tell HOW FAR a refused submission was from grounding its predictions.
        # Presence therefore does not mean the judge ran; a non-None VALUE does.
        # `None` is precisely "the unit failed before g3, so faithfulness never existed", which
        # is the state this claim is about.
        "reached_judge": outcome.diagnostics.get("faithfulness") is not None,
    }


# --------------------------------------------------------------------------- the claims
@dataclass(frozen=True)
class DocClaim:
    """One documented statement, the scenario that produces it, and where it is written."""

    claim_id: str
    where: str  # "track" (this repo) or "hub" (the starter pack)
    document: str
    anchor: str
    measure: Callable[[pathlib.Path], dict[str, Any]]
    expected: dict[str, Any]
    #: Values that must appear verbatim in the anchored passage, so prose cannot drift from
    #: the measurement while both remain individually self-consistent.
    stated: tuple[str, ...]


CLAIMS: tuple[DocClaim, ...] = (
    DocClaim(
        claim_id="constant-point-forecast-scores-half",
        where="hub",
        document="starter-packs/track4/AGENTS.md",
        anchor="**A CONSTANT `point_forecast` scores 0.5",
        measure=_ranking_quality,
        expected={"correct": 1.0, "constant": 0.5, "reversed": 0.0},
        stated=("1.0000", "0.5000", "0.0000"),
    ),
    DocClaim(
        claim_id="inadmissible-equals-worst-admissible",
        where="hub",
        document="starter-packs/track4/AGENTS.md",
        anchor="An inadmissible answer scores W; every admissible answer scores above W",
        measure=_regression_composites,
        # 5.0.0 measured +0.67 / +0.17 / -0.07 / -0.27. Under 5.1.0 the zero-width "precise"
        # answer is no longer the lowest admissible one (the interval leg pays width), so the
        # starter pack's table needs rewording, not just new numbers.
        expected={
            "perfect": 0.8737,
            "honest": 0.2008,
            "worst_admissible": 0.2297,
            "inadmissible": 0.0,
            "inadmissible_code": "incomplete_output",
        },
        stated=("+0.8737", "+0.2008", "+0.2297", "0.0000"),
    ),
    DocClaim(
        claim_id="no-output-takes-W-not-zero",
        where="track",
        document="SUBMISSION_CLI.md",
        anchor="the unit takes the pre-committed worst",
        measure=_no_output_and_malformed,
        expected={  # 5.0.0: -0.27 for both
            "no_output_code": "no_output",
            "no_output_score": 0.0,
            "malformed_code": "malformed_output",
            "malformed_score": 0.0,
        },
        stated=("W = 0.0", "`no_output`\nat 0.0", "`malformed_output` at 0.0"),
    ),
    DocClaim(
        claim_id="omitted-rows-are-refused-not-scored",
        where="track",
        document="docs/CONCEPTS.md",
        anchor="**You cannot omit a row at all.**",
        measure=_roster_exactness,
        # 5.0.0: full +0.67, omitted -0.27
        expected={"full": 0.8737, "omitted": 0.0, "omitted_code": "incomplete_output"},
        stated=("incomplete_output", "+0.8737", "W = 0.0"),
    ),
    DocClaim(
        claim_id="unresolvable-doc-id-never-reaches-the-judge",
        where="hub",
        document="starter-packs/track4/AGENTS.md",
        anchor="an **unresolvable `doc_id`**",
        measure=_unresolvable_doc_id,
        expected={  # 5.0.0: score -0.27
            "code": "domain_gate_failed",
            "score": 0.0,
            "reached_judge": False,
        },
        stated=("domain_gate_failed", "W = 0.0"),
    ),
)


@pytest.mark.parametrize("claim", CLAIMS, ids=lambda c: c.claim_id)
def test_documented_number_matches_the_scorer(
    claim: DocClaim, tmp_path: pathlib.Path
) -> None:
    """Run the scenario the document describes and compare it to what the document says."""
    measured = claim.measure(tmp_path)
    for key, want in claim.expected.items():
        got = measured[key]
        if isinstance(want, float):
            assert got == pytest.approx(want, abs=5e-5), (
                f"{claim.document} claims {key} = {want}; the scorer produced {got}. "
                "Either the scorer changed and the document is now wrong, or this "
                "expectation was wrong. Fix whichever is untrue — do not relax the tolerance."
            )
        else:
            assert (
                got == want
            ), f"{claim.document} claims {key} = {want!r}; the scorer produced {got!r}."


@pytest.mark.parametrize("claim", CLAIMS, ids=lambda c: c.claim_id)
def test_document_still_states_what_was_measured(claim: DocClaim) -> None:
    """The prose must still carry the measured numbers, in the passage that was verified."""
    passage = _passage(
        _doc_text(claim.where, claim.document),
        claim.anchor,
        doc=claim.document,
    )
    for value in claim.stated:
        assert value in passage, (
            f"{claim.document}: the passage anchored at {claim.anchor!r} no longer states "
            f"{value!r}. The measurement in this module is the source of truth; update the "
            "prose to match it, or update both together."
        )
