#!/usr/bin/env python3
"""Install each unit's declared naive rule as ``units/<u>/reference/naive_answer.json``.

The scorer reads ``reference/naive_answer.json`` beside ``reference/outcome.json``: it is the
zero-skill anchor of regression scoring (``naive_mae / (naive_mae + mae)``), and a scored regression
unit without it is an organizer fault. The declared rules live at
``baselines/naive/<u>/<rule>.answer.json`` in the production root, one per unit.

Given the production root, this copies each unit's SINGLE naive answer to its reference directory,
byte for byte, and prints a table of (unit, source file, sha256, action). Nothing is written until
every unit has passed the checks: exactly one ``*.answer.json`` per naive directory, a matching
unit directory, no differing file already installed, and every regression unit covered.

    python scripts/install_naive_reference.py <production-root> --dry-run   # table only
    python scripts/install_naive_reference.py <production-root>             # copies (a human step)

Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import sys
import tomllib

TARGET = pathlib.Path("reference") / "naive_answer.json"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_type(unit_dir: pathlib.Path) -> str | None:
    card = unit_dir / "card.toml"
    if not card.is_file():
        return None
    params = (tomllib.loads(card.read_text(encoding="utf-8")).get("scoring") or {}).get(
        "params"
    ) or {}
    value = params.get("target_type")
    return value if isinstance(value, str) else None


def plan(
    root: pathlib.Path,
) -> tuple[list[tuple[str, pathlib.Path, str, str]], list[str]]:
    """Return (rows, errors). A row is (unit, source, sha256, action)."""
    naive_root = root / "baselines" / "naive"
    units_root = root / "units"
    rows: list[tuple[str, pathlib.Path, str, str]] = []
    errors: list[str] = []
    if not naive_root.is_dir() or not units_root.is_dir():
        return rows, [f"{root} has no baselines/naive/ or no units/ directory"]
    covered: set[str] = set()
    for naive_dir in sorted(p for p in naive_root.iterdir() if p.is_dir()):
        unit = naive_dir.name
        sources = sorted(naive_dir.glob("*.answer.json"))
        if len(sources) != 1:
            errors.append(f"{unit}: {len(sources)} naive answer files, need exactly 1")
            continue
        unit_dir = units_root / unit
        if not unit_dir.is_dir():
            errors.append(f"{unit}: naive rule has no unit directory")
            continue
        source = sources[0]
        digest = _sha256(source)
        dest = unit_dir / TARGET
        if dest.is_symlink():
            errors.append(f"{unit}: {TARGET} is a link")
            continue
        if dest.exists():
            if _sha256(dest) != digest:
                errors.append(f"{unit}: {TARGET} exists and differs from {source.name}")
                continue
            action = "present"
        else:
            action = "copy"
        covered.add(unit)
        rows.append((unit, source, digest, action))
    for unit_dir in sorted(p for p in units_root.iterdir() if p.is_dir()):
        if _target_type(unit_dir) == "regression" and unit_dir.name not in covered:
            errors.append(
                f"{unit_dir.name}: regression unit with no installable naive rule"
            )
    return rows, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("root", type=pathlib.Path, help="the production root")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the table, write nothing"
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    rows, errors = plan(root)
    for message in errors:
        print(f"ERROR {message}", file=sys.stderr)
    if errors:
        print(f"refused: {len(errors)} error(s); nothing written", file=sys.stderr)
        return 1

    print("| unit | source file | sha256 | action |")
    print("|:--|:--|:--|:--|")
    written = 0
    for unit, source, digest, action in rows:
        if action == "copy" and not args.dry_run:
            dest = root / "units" / unit / TARGET
            dest.parent.mkdir(exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            tmp.write_bytes(source.read_bytes())
            os.replace(tmp, dest)
            if _sha256(dest) != digest:
                print(
                    f"ERROR {unit}: installed file does not match its source",
                    file=sys.stderr,
                )
                return 1
            written += 1
            action = "copied"
        elif action == "copy":
            action = "would copy"
        print(f"| {unit} | {source.relative_to(root)} | {digest} | {action} |")
    mode = "DRY RUN, nothing written" if args.dry_run else f"{written} file(s) written"
    print(f"\n{len(rows)} unit(s); {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
