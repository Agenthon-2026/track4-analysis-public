"""Load evaluation rosters and stage only declared prediction inputs."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from baselines.strong_rag_baseline.indexer import calendar_date

REPO = Path(__file__).resolve().parents[2]
SPLITS = ("train", "calibration", "test")


@dataclass(frozen=True)
class Case:
    case_id: str
    unit_dir: Path
    split: str
    group: str
    truth_path: Path | None = None
    domain: str | None = None


def public_roots() -> list[Path]:
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        REPO,
        *[
            Path(line.removeprefix("worktree ")).resolve()
            for line in result.stdout.splitlines()
            if line.startswith("worktree ")
        ],
    ]


def require_external(path: Path, roots: list[Path]) -> None:
    if any(path.resolve().is_relative_to(root.resolve()) for root in roots):
        raise ValueError(
            "evaluation manifests, truth and outputs must stay outside public worktrees"
        )


def load_cases(*, units: Path | None, manifest: Path | None) -> list[Case]:
    roots = public_roots()
    if units is not None:
        cases = [
            Case(path.name, path.resolve(), "public-dev", path.name)
            for path in sorted(units.iterdir())
            if (path / "task.json").is_file()
        ]
    else:
        assert manifest is not None
        require_external(manifest, roots)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("evaluation manifest must be an object with version 1")
        if not isinstance(data.get("cases"), list):
            raise ValueError("evaluation manifest must contain a cases list")
        cases = []
        for row in data["cases"]:
            if not isinstance(row, dict) or set(row) - {
                "id",
                "unit_dir",
                "split",
                "group",
                "truth_path",
                "domain",
            }:
                raise ValueError("invalid evaluation case fields")
            if not all(
                isinstance(row.get(key), str) and row[key]
                for key in ("id", "unit_dir", "split", "group")
            ):
                raise ValueError(
                    "each case requires id, unit_dir, split and group strings"
                )
            if row["split"] not in SPLITS:
                raise ValueError(
                    "private evaluation split must be train, calibration or test"
                )
            domain = row.get("domain")
            if domain is not None and (
                not isinstance(domain, str) or not domain.strip()
            ):
                raise ValueError("domain must be a nonempty string when provided")
            truth = row.get("truth_path")
            if truth is not None and (not isinstance(truth, str) or not truth):
                raise ValueError("truth_path must be a nonempty string when provided")
            truth_path = (manifest.parent / truth).resolve() if truth else None
            if truth_path is not None:
                require_external(truth_path, roots)
            cases.append(
                Case(
                    row["id"],
                    (manifest.parent / row["unit_dir"]).resolve(),
                    row["split"],
                    row["group"],
                    truth_path,
                    domain,
                )
            )
    if not cases:
        raise ValueError("evaluation roster is empty")
    ids: set[str] = set()
    paths: set[Path] = set()
    groups: dict[str, str] = {}
    dates: dict[str, list[tuple[str, str]]] = {}
    for case in cases:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", case.case_id):
            raise ValueError("case id must be a safe directory name")
        if case.case_id in ids or case.unit_dir in paths:
            raise ValueError("duplicate case id or unit directory in evaluation roster")
        ids.add(case.case_id)
        paths.add(case.unit_dir)
        if case.group in groups and groups[case.group] != case.split:
            raise ValueError("an event group cannot cross evaluation splits")
        groups[case.group] = case.split
        task = json.loads((case.unit_dir / "task.json").read_text(encoding="utf-8"))
        cutoff = calendar_date(task["cutoff_date"])
        resolution = calendar_date(task["resolution_date"])
        if resolution < cutoff:
            raise ValueError("resolution_date precedes cutoff_date")
        dates.setdefault(case.split, []).append(
            (cutoff.isoformat(), resolution.isoformat())
        )
        if case.truth_path is not None:
            if not case.truth_path.is_file():
                raise ValueError("declared development truth file does not exist")
            if any(case.truth_path.is_relative_to(c.unit_dir) for c in cases):
                raise ValueError(
                    "truth must be stored separately from all prediction units"
                )
    for i, earlier in enumerate(SPLITS):
        for later in SPLITS[i + 1 :]:
            if earlier in dates and later in dates:
                if max(end for _, end in dates[earlier]) >= min(
                    start for start, _ in dates[later]
                ):
                    raise ValueError(
                        "evaluation splits overlap in time: earlier outcomes must precede later cutoffs"
                    )
    return cases


def stage_inputs(unit: Path, dest: Path) -> str:
    """Copy task and manifest-declared JSON corpus only; return their content digest."""
    declared = json.loads((unit / "manifest.json").read_text(encoding="utf-8"))
    names = ["task.json"]
    for item in declared["files"]:
        if item.get("role") != "corpus" or item["path"] == "corpus/manifest.json":
            continue
        path = Path(item["path"])
        if path.parts and path.parts[0] == "corpus" and path.suffix == ".json":
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe corpus path in manifest")
            names.append(path.as_posix())
    if len(names) == 1:
        raise ValueError("manifest declares no JSON evidence documents")
    digest = hashlib.sha256()
    (dest / "corpus").mkdir(parents=True)
    for name in sorted(set(names)):
        source = unit / name
        if not source.resolve().is_relative_to(unit.resolve()) or source.is_symlink():
            raise ValueError("prediction input may not link outside its unit")
        payload = source.read_bytes()
        digest.update(name.encode() + b"\0" + payload + b"\0")
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return digest.hexdigest()
