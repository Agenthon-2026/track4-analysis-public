"""Repo-local answer-material firewall, because the shared one exempts our whole split.

MEASURED 2026-08-28 against the installed toolkit, planting one file at a time at the root of
`units/t4-EXAMPLE-eps-beat/` and running `qfbench2 manifest assert-public-safe` on it:

    solution.py             -> rejected      outcome.json            -> ACCEPTED
    my_oracle.json          -> rejected      expected_labels.json    -> ACCEPTED
    answer_key_x.json       -> rejected      reference/x.json        -> ACCEPTED
    solution/x.json         -> rejected      adversarial_variants/   -> ACCEPTED
                                             adversarial_log.jsonl   -> ACCEPTED
                                             perturbation_log.jsonl  -> ACCEPTED
                                             solution_leak.py        -> ACCEPTED

Two independent reasons, both in `qfbench2_common.manifest.assert_public_safe`:

* Its rule 4 blocks answer-bearing material (`reference/`, `adversarial_variants/`, `outcome*.json`,
  `expected*`) only for units that are NOT `public-dev`. That exemption exists so a Track-1 practice
  unit can ship `checks/reference_data/` for self-grading. **Every published Track-4 unit is
  `public-dev`**, so for this repository the exemption switches the rule off entirely.
* Its oracle globs are `*oracle*`, `answer_key*`, `solve.sh`, `solution.py` — literal
  `solution.py`, so `solution_leak.py` matches nothing, and neither `adversarial_log.jsonl` nor
  `perturbation_log.jsonl` appears in any list.

AGENTS.md promises this repository rejects all of them. This test is what makes that true here.
Widening the shared toolkit's rules is a hub decision and is filed separately; until then a
track-local guard is the honest way to keep the promise the docs make.

Standard library only, so it runs in the secret-free ``firewall`` CI job alongside the stdlib half
of the unit validator — the half that still runs when the toolkit install fails.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UNITS = REPO / "units"

#: Directory names that may never appear anywhere under units/, at any depth.
FORBIDDEN_DIRS = frozenset(
    {
        "reference",
        "adversarial_variants",
        "solution",
        "solutions",
        "oracle",
        "oracle_logs",
        "oracle_output",
        "dev",
    }
)

#: File-name patterns that may never appear anywhere under units/, at any depth.
#: Anchored, case-insensitive, matched against the file NAME only.
FORBIDDEN_FILE_RES = tuple(
    re.compile(p, re.I)
    for p in (
        r"^outcome.*\.json$",
        r".*outcome.*\.json$",
        r"^expected.*",
        r"^answer_key.*",
        r"^solution.*",
        r".*oracle.*",
        r"^solve\.sh$",
        r"^checkpoints\.json$",
        r"^adversarial_log\.jsonl$",
        r"^perturbation_log\.jsonl$",
        r"^generation_log\.jsonl$",
        r"^review_log\.jsonl$",
        r"^canary_registry\.json$",
    )
)


def _offenders(root: Path) -> list[str]:
    """Every path under `root` that answer-material rules forbid."""
    out: list[str] = []
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        if path.is_dir():
            if path.name.lower() in FORBIDDEN_DIRS:
                out.append(f"{rel}/ (forbidden directory name)")
            continue
        if not path.is_file():
            out.append(f"{rel} (not a regular file)")
            continue
        for rx in FORBIDDEN_FILE_RES:
            if rx.match(path.name):
                out.append(f"{rel} (matches {rx.pattern})")
                break
    return out


def test_units_directory_exists() -> None:
    """Fail-closed: an absent units/ is a guard that stopped guarding."""
    assert UNITS.is_dir(), "units/ is missing; this guard has nothing to patrol"
    assert any(UNITS.iterdir()), "units/ is empty"


def test_no_answer_material_anywhere_under_units() -> None:
    offenders = _offenders(UNITS)
    assert not offenders, (
        "answer-bearing or oracle material found under units/. None of this belongs in a public "
        "repository, and the shared toolkit's firewall does not reject all of it for a public-dev "
        "unit (see this module's docstring):\n  " + "\n  ".join(offenders)
    )


def test_the_guard_catches_every_pattern_the_shared_firewall_lets_through(
    tmp_path: Path,
) -> None:
    """Positive controls: the exact plants measured as ACCEPTED by assert_public_safe."""
    missed_by_the_toolkit = [
        "outcome.json",
        "expected_labels.json",
        "adversarial_log.jsonl",
        "perturbation_log.jsonl",
        "solution_leak.py",
        "reference_outcome.json",
        "reference/outcome.json",
        "adversarial_variants/counterfactual/corpus/doc.json",
    ]
    for rel in missed_by_the_toolkit:
        unit = tmp_path / rel.replace("/", "_").replace(".", "_")
        target = unit / "t4-control" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
        assert _offenders(unit), (
            f"this guard does not catch {rel!r}, which the shared firewall also accepts — the "
            "gap it exists to close would still be open"
        )


def test_the_guard_does_not_flag_a_legitimate_unit(tmp_path: Path) -> None:
    """Negative control: the real layout of a published unit must stay clean."""
    unit = tmp_path / "t4-control"
    (unit / "corpus").mkdir(parents=True)
    (unit / "card.toml").write_text("schema_version = \"2.0\"\n", encoding="utf-8")
    (unit / "task.json").write_text("{}", encoding="utf-8")
    (unit / "manifest.json").write_text("{}", encoding="utf-8")
    (unit / "corpus" / "manifest.json").write_text("{}", encoding="utf-8")
    (unit / "corpus" / "EDGAR_0000320193_10Q_20240202.json").write_text("{}", encoding="utf-8")
    assert not _offenders(tmp_path), (
        "the guard flags an ordinary published unit; it would be turned off within a week"
    )
