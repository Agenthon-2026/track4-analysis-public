"""Stable hashes and regular-file reads for evaluator-owned evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def read_json_inside(root: Path, relative: str) -> dict:
    path = root / relative
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
        or not path.is_file()
    ):
        raise ValueError(
            "evidence must be a regular file inside its artifact directory"
        )
    return json.loads(path.read_text())
