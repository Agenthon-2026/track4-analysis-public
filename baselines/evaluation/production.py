"""Bind local production-profile evaluation to an explicit, immutable judge.

This verifies configured artifacts, not organizer approval or runtime equivalence.
It never constructs a model, downloads weights, or authorizes paid inference.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .dataset import public_roots, require_external


def freeze_judge(path: Path) -> dict:
    from qfbench2_track_analysis.codes import T4OrganizerFault
    from qfbench2_track_analysis.judge_factory import (
        JudgeProvenance,
        compute_cache_tree_digest,
        load_judge_spec,
    )

    require_external(path, public_roots())
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("judge specification must be a regular file")
        original = path.read_bytes()
        spec = load_judge_spec(path)
        cache = Path(spec.cache_dir)
        if not cache.is_absolute():
            raise ValueError("registered judge cache must use an absolute path")
        require_external(cache, public_roots())
        if compute_cache_tree_digest(cache) != spec.cache_tree_digest:
            raise ValueError("judge cache differs from the pinned specification")
        if path.read_bytes() != original:
            raise ValueError("judge specification changed during verification")
    except T4OrganizerFault as exc:
        raise ValueError(f"invalid production judge configuration: {exc}") from exc
    return {
        "spec_path": str(path.absolute()),
        "spec_sha256": hashlib.sha256(original).hexdigest(),
        "cache_dir": str(cache),
        "provenance": JudgeProvenance(
            judge_mode="production",
            model_ids=spec.model_ids,
            model_revisions=dict(spec.model_revisions),
            tokenizer_digest=spec.tokenizer_digest,
            local_cache_tree_digest=spec.cache_tree_digest,
        ).to_mapping(),
        "execution": "local-cached-offline",
    }


def verify_judge(record: dict) -> dict | None:
    binding = record.get("production_judge")
    if record["profile"] == "smoke":
        if binding is not None:
            raise ValueError("smoke registration cannot carry a production judge")
        return None
    if record["profile"] != "production" or not isinstance(binding, dict):
        raise ValueError("production registration requires a pinned judge")
    if freeze_judge(Path(binding["spec_path"])) != binding:
        raise ValueError("production judge changed since registration")
    return binding


def evaluation_environment(record: dict) -> dict | None:
    binding = verify_judge(record)
    if binding is None:
        return None
    from qfbench2_track_analysis.judge_factory import (
        ENV_JUDGE_CACHE_DIR,
        ENV_JUDGE_SPEC,
    )

    # Override ambient configuration for the evaluator process only. Inference
    # still uses the official factory and must load the already verified cache.
    return dict(
        os.environ,
        **{
            ENV_JUDGE_SPEC: binding["spec_path"],
            ENV_JUDGE_CACHE_DIR: binding["cache_dir"],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "T4_JUDGE_BACKEND": "local",
        },
    )


def validate_judge_report(report: dict, record: dict) -> None:
    binding = record.get("production_judge")
    if record["profile"] != "production":
        return
    if not isinstance(binding, dict):
        raise ValueError("production report has no registered judge")
    for row in report["runs"]:
        assessment = row.get("assessment", {})
        if (
            row.get("profile") != "production"
            or assessment.get("profile") != "production"
            or assessment.get("judge") != binding["provenance"]
        ):
            raise ValueError(
                "report judge differs from the registered production judge"
            )
