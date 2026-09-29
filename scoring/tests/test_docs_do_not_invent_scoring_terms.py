"""No published document may promise a scoring term the scorer does not implement.

`docs/CONCEPTS.md` described a normalised **Winkler score** -- with a 20x miss penalty and a
baseline calibrated to 0.5 -- and told participants that sharpness "rewards narrow intervals".
None of it was ever implemented. `README.md` repeated the advice, so both files pushed
participants toward a strategy the scorer actually punishes.

Measured on a three-entity unit, everything identical except the interval:

    [1.4, 1.9]      -> score -0.037, coverage 0.0
    [-1e9, 1e9]     -> score +0.203, coverage 1.0

The entire calibration leg is `w_cal * |interval_coverage - interval_level|`, so on any single
unit widening strictly helps. Advice to narrow was not merely unenforced, it was backwards.

(That was scorer 5.0.0. From 5.1.0 the interval leg is a Gneiting-Raftery interval score against
the naive interval, `interval_score` in the code; the docs describe it under that name, so every
term below stays patrolled.)

This test is deliberately keyed on the CODE, not on a hardcoded list of banned words: a term is
forbidden in the docs exactly when no Python file implements it. If someone later ships a real
Winkler term, this test goes green on its own rather than having to be remembered.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

#: Scoring vocabulary that would change how a participant sizes an interval. Each is checked
#: against the code before it is held against the docs.
_TERMS = ("winkler", "sharpness", "brier", "interval_width", "pinball", "crps")

_DOCS = (
    "README.md",
    "docs/CONCEPTS.md",
    "docs/CATEGORIES.md",
    "docs/AUTHORING-GUIDE.md",
    "SUBMISSION_CLI.md",
    "baselines/README.md",
)


#: Implementation only. TEST files are excluded deliberately: this very file names every term
#: it checks, so sweeping tests made `_implemented` return True for all of them -- all six
#: assertions skipped and the negative control failed. The control is the only reason that was
#: caught rather than shipped as six silent skips.
_CODE_ROOTS = ("qfbench2_track_analysis", "scoring", "faithfulness", "baselines")


def _implemented(term: str) -> bool:
    for root in _CODE_ROOTS:
        for path in (REPO / root).rglob("*.py"):
            parts = set(path.parts)
            if parts & {".git", "build", "__pycache__"} or "tests" in parts:
                continue
            if path.name.startswith("test_"):
                continue
            if term in path.read_text(encoding="utf-8", errors="replace").lower():
                return True
    return False


def test_the_detector_finds_a_term_that_is_implemented() -> None:
    """Control. Without it, a broken sweep would report every doc clean."""
    assert _implemented("interval_coverage"), (
        "the code sweep cannot find `interval_coverage`, which the scorer definitely "
        "implements -- the detector is broken and the assertions below prove nothing"
    )
    assert not _implemented("a_term_no_file_contains_zzz")


@pytest.mark.parametrize("term", _TERMS)
def test_no_doc_promises_a_scoring_term_the_scorer_does_not_implement(
    term: str,
) -> None:
    if _implemented(term):
        pytest.skip(
            f"NAMED REASON: {term!r} IS implemented in code, so the docs may describe it"
        )
    missing = [name for name in _DOCS if not (REPO / name).exists()]
    assert not missing, (
        f"patrolled document(s) missing: {missing}. This guard used to `continue` past a missing "
        "file, which meant deleting a document turned its check green. Restore the file or remove "
        "it from _DOCS in a commit that says why (the fail-closed rule)."
    )
    offenders = []
    for name in _DOCS:
        path = REPO / name
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            # A term inside inline code is being NAMED (a grep pattern, a field), not promised.
            prose = re.sub(r"`[^`]*`", "", line).lower()
            if term not in prose:
                continue
            # Denials are the point of this file, and prose wraps: look at the sentence, which
            # in these documents spans two or three physical lines. Checking one line at a time
            # flagged this test's own explanation, whose "no" sat on the line above.
            window = " ".join(lines[max(0, number - 3) : number + 2]).lower()
            if re.search(
                r"\b(no|not|never|nothing|does not|did not|was ever)\b", window
            ):
                continue
            offenders.append(f"{name}:{number}: {line.strip()[:110]}")
    assert not offenders, (
        f"docs promise {term!r}, which no Python file implements:\n  "
        + "\n  ".join(offenders)
    )
