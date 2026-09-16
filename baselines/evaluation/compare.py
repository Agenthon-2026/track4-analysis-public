"""Paired development comparisons, resampling independent event groups."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .dataset import public_roots, require_external
from .historical import write_json


def _rows(report: dict) -> dict:
    rows = {}
    for row in report["runs"]:
        key = (row["case_id"], row["seed"])
        value = row["assessment"].get("development_score")
        if key in rows or row["status"] != "completed":
            raise ValueError("duplicate or incomplete comparison roster")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("every planned run needs a finite development score")
        if not row.get("truth_digest"):
            raise ValueError("comparison requires recorded development truth digests")
        rows[key] = row
    if not rows:
        raise ValueError("empty comparison roster")
    return rows


def compare(
    left: dict, right: dict, *, samples: int = 10000, seed: int = 20260908
) -> dict:
    a, b = _rows(left), _rows(right)
    if set(a) != set(b):
        raise ValueError("comparison case/seed rosters differ")
    if (
        left["provenance"]["toolkit_source_digest"]
        != right["provenance"]["toolkit_source_digest"]
    ):
        raise ValueError(
            "comparison toolkit bytes differ; re-evaluate both with one scorer"
        )
    groups: dict[str, list[float]] = defaultdict(list)
    splits: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    kinds: dict[str, list[float]] = defaultdict(list)
    kind_groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    domains: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key, before in a.items():
        after = b[key]
        for field in (
            "input_digest",
            "truth_digest",
            "split",
            "group",
            "profile",
            "target_type",
        ):
            if before[field] != after[field]:
                raise ValueError(f"comparison {field} differs for {key}")
        if before.get("domain") != after.get("domain"):
            raise ValueError(f"comparison domain differs for {key}")
        # Judge/scorer identity includes local production-vs-smoke differences.
        for field in ("scorer", "judge"):
            if before["assessment"].get(field) != after["assessment"].get(field):
                raise ValueError(f"comparison {field} differs for {key}")
        delta = (
            after["assessment"]["development_score"]
            - before["assessment"]["development_score"]
        )
        groups[before["group"]].append(delta)
        splits[before["split"]][before["group"]].append(delta)
        kinds[before["target_type"]].append(delta)
        kind_groups[before["target_type"]][before["group"]].append(delta)
        if before.get("domain"):
            domains[before["domain"]][before["group"]].append(delta)
    if not 100 <= samples <= 100000:
        raise ValueError("bootstrap samples must be between 100 and 100000")

    def summary(values: dict) -> dict:
        deltas = [mean(values[group]) for group in sorted(values)]
        rng = random.Random(seed)
        draws = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(samples))
        return {
            "independent_groups": len(deltas),
            "event_mean_delta": mean(deltas),
            "bootstrap_95_percent": [
                draws[int(samples * 0.025)],
                draws[min(samples - 1, int(samples * 0.975))],
            ]
            if len(deltas) >= 2
            else None,
        }

    return {
        "version": 1,
        "official_score": None,
        "rankable": False,
        "runs": len(a),
        "overall": summary(groups),
        "by_split": {
            split: summary(values) for split, values in sorted(splits.items())
        },
        "by_target_type_mean_delta": {
            kind: mean(values) for kind, values in sorted(kinds.items())
        },
        "by_target_type": {
            key: summary(values) for key, values in sorted(kind_groups.items())
        },
        "by_domain": {key: summary(values) for key, values in sorted(domains.items())},
        "before_failures": sum(not r["assessment"]["admissible"] for r in a.values()),
        "after_failures": sum(not r["assessment"]["admissible"] for r in b.values()),
        "bootstrap": {
            "seed": seed,
            "samples": samples,
            "unit": "event group; views/seeds averaged within group",
        },
        "note": "Smoke scores omit production NLI. Small or single-domain samples do not establish generalization. Inspect test separately; overall includes all supplied splits.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        for path in (args.before, args.after, args.out):
            require_external(path, public_roots())
        if args.out.exists():
            raise ValueError("comparison output must be new")
        result = compare(
            json.loads(args.before.read_text()), json.loads(args.after.read_text())
        )
        write_json(args.out, result)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(f"Comparison aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
