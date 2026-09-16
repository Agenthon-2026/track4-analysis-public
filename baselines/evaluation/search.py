"""Budget development candidates before selecting one for fresh acceptance.

Uses the existing batch runner, comparator and lifecycle journal. Development
scores select a candidate only; they never establish G2 or production eligibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import batch, lifecycle
from .acceptance import load_policy
from .artifacts import digest
from .compare import compare
from .dataset import load_cases


def freeze(manifest: Path, seeds: list[int]) -> dict:
    expected = batch.freeze_roster(manifest, seeds, "calibration")
    cases = load_cases(units=None, manifest=manifest)
    return {
        "manifest": str(manifest.resolve()),
        "manifest_digest": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "seeds": seeds,
        "expected_runs": expected,
        "resolution_bound": max(
            json.loads((c.unit_dir / "task.json").read_text())["resolution_date"]
            for c in cases
        ),
        "candidate_limit": load_policy()["max_candidates_per_round"],
        "candidates": [],
        "abandoned": [],
        "selection": None,
    }


def _current(root: Path) -> tuple[dict, dict]:
    state, _ = lifecycle._read(root)
    current = state.get("round")
    if not current or not current.get("development"):
        raise ValueError("start a round with a frozen development manifest and seeds")
    if current["policy_digest"] != digest(load_policy()):
        raise ValueError("development policy changed since round start")
    if current["incumbent"] != state["incumbent"]["identity"]:
        raise ValueError("development incumbent changed; start another round")
    return current, current["development"]


def apply(state: dict, action: str, data: dict) -> None:
    current = state.get("round")
    if (
        not current
        or not current.get("development")
        or data["round_id"] != current["id"]
    ):
        raise ValueError("development action requires its frozen search round")
    search = current["development"]
    if search["selection"] is not None or current["candidates"]:
        raise ValueError("development search is already closed for selection")
    if action == "development_register":
        if len(search["candidates"]) >= search["candidate_limit"]:
            raise ValueError("development candidate limit exhausted")
        if any(
            c["version"]["identity"] == data["version"]["identity"]
            for c in search["candidates"]
        ):
            raise ValueError("development candidate already registered")
        if data["version"]["identity"] == current["incumbent"]:
            raise ValueError("development candidate must differ from incumbent")
        lifecycle._check_capacity(current, data["role_budget"], roles=2)
        search["candidates"].append(data)
    elif action == "development_reserve":
        candidate = next(
            (c for c in search["candidates"] if c["batch_id"] == data["batch_id"]), None
        )
        if not candidate or data["batch_id"] in search["abandoned"]:
            raise ValueError("development batch is unregistered or abandoned")
        if (
            data["role"] not in ("before", "after")
            or data["budget"] != candidate["role_budget"]
        ):
            raise ValueError("development role budget changed")
        if any(
            r["batch_id"] == data["batch_id"] and r["role"] == data["role"]
            for r in current["reservations"]
        ):
            raise ValueError("development run budget already consumed; do not retry")
        lifecycle._check_capacity(current, data["budget"])
        current["reservations"].append(data)
    elif action == "development_abandon":
        if (
            data["batch_id"] not in {c["batch_id"] for c in search["candidates"]}
            or data["batch_id"] in search["abandoned"]
        ):
            raise ValueError("development batch already abandoned or unregistered")
        search["abandoned"].append(data["batch_id"])
    elif action == "development_select":
        search["selection"] = data["result"]
    else:
        raise ValueError("unknown development action")


def register_candidate(root: Path, batch_id: str, record: dict) -> None:
    """Called by registration under the shared registry lock."""
    current, search = _current(root)
    if any(
        record[k] != search[k]
        for k in ("manifest", "manifest_digest", "seeds", "expected_runs")
    ):
        raise ValueError("development roster or seeds differ from the frozen round")
    if record["versions"]["before"]["identity"] != current["incumbent"]:
        raise ValueError("development comparison must retain the round incumbent")
    for frozen in record["versions"].values():
        if frozen["identity"].get("git_dirty") is not False:
            raise ValueError("development search requires clean frozen versions")
    lifecycle._append(
        root,
        "development_register",
        {
            "round_id": current["id"],
            "batch_id": batch_id,
            "version": record["versions"]["after"],
            "role_budget": batch.execution_budget(record),
        },
    )


def reserve_run(root: Path, directory: Path, role: str, record: dict) -> dict:
    current, search = _current(root)
    if (
        record["profile"] != "smoke"
        or record["mode"] != "grounded"
        or record["budget"]["model_request_attempts"] != 0
    ):
        raise ValueError("development search only permits offline grounded/smoke")
    if any(
        record[k] != search[k]
        for k in ("manifest", "manifest_digest", "seeds", "expected_runs")
    ):
        raise ValueError("development registration differs from frozen search")
    reservation = {
        "round_id": current["id"],
        "batch_id": directory.name,
        "role": role,
        "budget": batch.execution_budget(record),
    }
    lifecycle._append(root, "development_reserve", reservation)
    return reservation


def verify_reservation(directory: Path, role: str, started: dict, record: dict) -> None:
    root = directory.parent.parent
    lifecycle.verify_reservation(root, directory, role, started, record)
    state, _ = lifecycle._read(root)
    rounds = state["round_history"] + ([state["round"]] if state["round"] else [])
    reservation = started["round_reservation"]
    current = next(r for r in rounds if r["id"] == reservation["round_id"])
    candidate = next(
        (
            c
            for c in current.get("development", {}).get("candidates", [])
            if c["batch_id"] == directory.name
        ),
        None,
    )
    if not candidate or candidate["version"] != record["versions"]["after"]:
        raise ValueError("receipt is not from a registered development candidate")


def evaluate(root: Path, *, current: dict | None = None) -> dict:
    if current is None:
        current, _ = _current(root)
    search = current["development"]
    policy = load_policy()
    if current["policy_digest"] != digest(policy):
        raise ValueError("development policy changed since selection")
    if not search["candidates"]:
        raise ValueError("no registered development candidates")
    measurements = []
    for candidate in search["candidates"]:
        identifier = candidate["batch_id"]
        if identifier in search["abandoned"]:
            measurements.append(
                {"batch_id": identifier, "status": "ABANDONED", "eligible": False}
            )
            continue
        directory = root / "batches" / identifier
        before, after, record = batch.verified_reports(directory)
        for frozen in record["versions"].values():
            if lifecycle._frozen(frozen["spec"]) != frozen:
                raise ValueError("development version changed before selection")
        measured = compare(
            before,
            after,
            samples=policy["bootstrap_samples"],
            seed=policy["bootstrap_seed"],
        )
        eligible = (
            all(
                row["execution"].get("returncode") == 0
                and row["execution"].get("timed_out") is False
                for report in (before, after)
                for row in report["runs"]
            )
            and measured["before_failures"] == measured["after_failures"] == 0
            and measured["overall"]["event_mean_delta"] > 0
            and all(
                s["event_mean_delta"] >= policy["min_stratum_gain"]
                for field in ("by_domain", "by_target_type")
                for s in measured[field].values()
            )
        )
        measurements.append(
            {
                "batch_id": identifier,
                "status": "MEASURED",
                "eligible": eligible,
                "comparison": measured,
                "report_digests": [digest(before), digest(after)],
            }
        )
    eligible = [m for m in measurements if m["eligible"]]
    winner = (
        min(
            eligible,
            key=lambda m: (
                -m["comparison"]["overall"]["event_mean_delta"],
                m["batch_id"],
            ),
        )
        if eligible
        else None
    )
    return {
        "version": 1,
        "round_id": current["id"],
        "policy_digest": current["policy_digest"],
        "selected_batch": winner["batch_id"] if winner else None,
        "decision": "SELECT_FOR_FRESH_ACCEPTANCE" if winner else "KEEP_INCUMBENT",
        "measurements": measurements,
        "rankable": False,
        "scope": "development screening only; no G1/G2/G3 qualification or first-availability certification",
    }


def select(root: Path) -> dict:
    with batch.locked(root):
        current, _ = _current(root)
        result = evaluate(root)
        lifecycle._append(
            root, "development_select", {"round_id": current["id"], "result": result}
        )
        return result


def abandon(root: Path, directory: Path, reason: str) -> dict:
    if directory.resolve().parent.parent != root.resolve() or not reason.strip():
        raise ValueError(
            "development abandonment requires a matching registry and reason"
        )
    with batch.locked(root):
        current, _ = _current(root)
        if all(
            (directory / f"{role}.receipt.json").exists()
            for role in ("before", "after")
        ):
            raise ValueError(
                "completed development evidence must participate in selection"
            )
        return lifecycle._append(
            root,
            "development_abandon",
            {
                "round_id": current["id"],
                "batch_id": directory.name,
                "reason": reason,
            },
        )


def verify_selection(root: Path, record: dict, *, current: dict | None = None) -> dict:
    if current is None:
        current, _ = _current(root)
    search = current["development"]
    if search["selection"] is None:
        raise ValueError("select from measured development evidence before acceptance")
    result = evaluate(root, current=current)
    if result != search["selection"]:
        raise ValueError("development selection evidence changed")
    selected = next(
        (c for c in search["candidates"] if c["batch_id"] == result["selected_batch"]),
        None,
    )
    if not selected or selected["version"] != record["versions"]["after"]:
        raise ValueError(
            "acceptance candidate differs from the selected development version"
        )
    if record["seeds"] != search["seeds"]:
        raise ValueError("acceptance seeds differ from the frozen search")
    cases = load_cases(units=None, manifest=Path(record["manifest"]))
    earliest = min(
        json.loads((c.unit_dir / "task.json").read_text())["cutoff_date"] for c in cases
    )
    if earliest <= search["resolution_bound"]:
        raise ValueError("development outcomes must precede fresh acceptance cutoffs")
    return {"round_id": current["id"], "selection_digest": digest(result)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("select")
    abort = commands.add_parser("abandon")
    abort.add_argument("--batch", type=Path, required=True)
    abort.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    result = (
        select(args.registry.resolve())
        if args.command == "select"
        else abandon(args.registry.resolve(), args.batch.resolve(), args.reason)
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
