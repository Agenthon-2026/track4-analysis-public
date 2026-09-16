"""Apply verified G2 decisions to a local development incumbent and rollback chain.

The append-only journal lives beside the batch registry, outside public worktrees.
It does not grant production qualification, deploy images or submit to the contest.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import batch
from .acceptance import load_policy
from .artifacts import digest, read_json_inside
from .dataset import public_roots, require_external


def _journal(root: Path) -> Path:
    require_external(root, public_roots())
    directory = root / "lifecycle"
    require_external(directory, public_roots())
    if directory.is_symlink():
        raise ValueError("lifecycle journal cannot be a symlink")
    return directory


def _apply(state: dict, action: str, data: dict) -> None:
    if action == "initialize":
        if state:
            raise ValueError("lifecycle already initialized")
        state.update(
            incumbent={
                **data["version"],
                "qualification": "bootstrap",
                "batch_id": None,
            },
            rollback=[],
            pending=None,
            consumed=[],
            production_candidate=None,
            round=None,
            round_history=[],
        )
    elif not state:
        raise ValueError("initialize the lifecycle first")
    elif action.startswith("development_"):
        from .search import apply

        apply(state, action, data)
    elif action == "start_round":
        if state["pending"] is not None or state["round"] is not None:
            raise ValueError("close the current round before starting another")
        if data["incumbent"] != state["incumbent"]["identity"]:
            raise ValueError("round baseline differs from incumbent")
        state["round"] = {**data, "candidates": [], "reservations": []}
    elif action == "close_round":
        if state["pending"] is not None or state["round"] is None:
            raise ValueError("resolve pending work before closing an active round")
        state["round_history"].append({**state["round"], "reason": data["reason"]})
        state["round"] = None
    elif action == "attach":
        if state["pending"] is not None or data["batch_id"] in state["consumed"]:
            raise ValueError("a batch is pending or has already been resolved")
        before = data["versions"]["before"]
        if before["identity"] != state["incumbent"]["identity"]:
            raise ValueError("batch baseline is not the current incumbent")
        if before["identity"] == data["versions"]["after"]["identity"]:
            raise ValueError("candidate must differ from incumbent")
        # Old journal entries remain readable, but new selections always carry
        # a round identity and must pass the budget checks below.
        if "round_id" in data:
            current = state["round"]
            if current is None or data["round_id"] != current["id"]:
                raise ValueError("selection requires an active round")
            if before["identity"] != current["incumbent"]:
                raise ValueError("start a new round after incumbent changes")
            candidate = digest(data["versions"]["after"]["identity"])
            if current["candidates"]:
                raise ValueError("only one candidate may enter acceptance per round")
            _check_capacity(current, data["role_budget"], roles=2)
            current["candidates"].append(candidate)
        state["pending"] = data
    elif action == "reserve_run":
        pending, current = state["pending"], state["round"]
        if pending is None or current is None or data["round_id"] != current["id"]:
            raise ValueError("run requires a selected candidate in an active round")
        if data["batch_id"] != pending["batch_id"] or data["role"] not in (
            "before",
            "after",
        ):
            raise ValueError("run is not part of the selected pair")
        if data["budget"] != pending["role_budget"]:
            raise ValueError("run budget differs from selected batch")
        if any(
            r["batch_id"] == data["batch_id"] and r["role"] == data["role"]
            for r in current["reservations"]
        ):
            raise ValueError("run budget already consumed; do not retry")
        _check_capacity(current, data["budget"])
        current["reservations"].append(data)
    elif action == "resolve":
        pending = state["pending"]
        if pending is None or pending["batch_id"] != data["batch_id"]:
            raise ValueError("decision does not belong to the pending batch")
        promote = all(data["goals"][key] == "PASS" for key in ("G1", "G2"))
        if data["decision"] != ("PROMOTE_DEVELOPMENT" if promote else "KEEP_INCUMBENT"):
            raise ValueError("transition disagrees with acceptance gates")
        if promote:
            state["rollback"].append(state["incumbent"])
            state["incumbent"] = {
                **pending["versions"]["after"],
                "qualification": "G2",
                "batch_id": data["batch_id"],
            }
        state["consumed"].append(data["batch_id"])
        state["pending"] = None
    elif action == "rollback":
        if state["pending"] is not None:
            raise ValueError("resolve the pending batch before rollback")
        if not state["rollback"]:
            raise ValueError("no retained incumbent to restore")
        if state["rollback"][-1] != data["restore"]:
            raise ValueError("rollback must restore the previous retained incumbent")
        state["incumbent"] = state["rollback"].pop()
    elif action == "abandon":
        if state["pending"] is None or state["pending"]["batch_id"] != data["batch_id"]:
            raise ValueError("no matching pending batch")
        state["consumed"].append(data["batch_id"])
        state["pending"] = None
    else:
        raise ValueError("unknown lifecycle action")


def _read(root: Path) -> tuple[dict, list[dict]]:
    directory = _journal(root)
    state: dict = {}
    events = []
    previous = None
    for index, path in enumerate(sorted(directory.glob("*.json"))):
        if path.name != f"{index:08d}.json":
            raise ValueError("lifecycle journal has a missing or unexpected record")
        envelope = read_json_inside(directory, path.name)
        event = envelope["event"]
        if envelope["digest"] != digest(event) or event["previous"] != previous:
            raise ValueError("lifecycle journal hash chain changed")
        if event["version"] != 1 or event["sequence"] != index:
            raise ValueError("unsupported lifecycle record")
        _apply(state, event["action"], event["data"])
        previous = envelope["digest"]
        events.append(envelope)
    return state, events


def _append(root: Path, action: str, data: dict) -> dict:
    state, events = _read(root)
    _apply(state, action, data)
    event = {
        "version": 1,
        "sequence": len(events),
        "previous": events[-1]["digest"] if events else None,
        "action": action,
        "data": data,
    }
    directory = _journal(root)
    directory.mkdir(exist_ok=True)
    batch.write_new(
        directory / f"{len(events):08d}.json", {"event": event, "digest": digest(event)}
    )
    return state


def status(root: Path) -> dict:
    with batch.locked(root):
        state, events = _read(root)
        if not events:
            raise ValueError("initialize the lifecycle first")
        return {**state, "journal_head": events[-1]["digest"], "events": len(events)}


def _frozen(spec: dict) -> dict:
    identity = batch.snapshot(spec)
    if identity.get("git_dirty") is not False:
        raise ValueError("lifecycle versions require clean immutable checkouts")
    return {"spec": spec, "identity": identity}


def initialize(root: Path, spec: dict) -> dict:
    with batch.locked(root):
        return _append(root, "initialize", {"version": _frozen(spec)})


def _check_capacity(current: dict, budget: dict, roles: int = 1) -> None:
    for field, limit in (("runs", "max_runs"), ("timeout_s", "max_reserved_seconds")):
        used = sum(r["budget"][field] for r in current["reservations"])
        if used + roles * budget[field] > current["budget"][limit]:
            raise ValueError(f"round {limit} budget exhausted")


def start_round(
    root: Path,
    hypothesis: str,
    budget: dict,
    *,
    development_manifest: Path | None = None,
    seeds: list[int] | None = None,
) -> dict:
    policy = load_policy()
    if (development_manifest is None) != (seeds is None):
        raise ValueError("development manifest and seeds must be frozen together")
    development = None
    if development_manifest is not None:
        from .search import freeze

        development = freeze(development_manifest, seeds)
    if not hypothesis.strip():
        raise ValueError("record a falsifiable round hypothesis")
    batch.validate_budget(budget)
    with batch.locked(root):
        state, events = _read(root)
        if not state:
            raise ValueError("initialize the lifecycle first")
        data = {
            "hypothesis": hypothesis,
            "budget": budget,
            "incumbent": state["incumbent"]["identity"],
            "policy_digest": digest(policy),
            "mode": "grounded",
            "profile": "smoke",
            "paid_call_budget": 0,
            "previous": events[-1]["digest"],
        }
        if development is not None:
            data["development"] = development
        return _append(root, "start_round", {**data, "id": digest(data)})


def close_round(root: Path, reason: str) -> dict:
    if not reason.strip():
        raise ValueError("record the round closing reason")
    with batch.locked(root):
        return _append(root, "close_round", {"reason": reason})


def reserve_run(root: Path, directory: Path, role: str, record: dict) -> dict:
    """Called by the batch runner under its registry lock, before starting work."""
    state, _ = _read(root)
    current = state.get("round")
    if current is None or current["policy_digest"] != record["policy_digest"]:
        raise ValueError("run requires an active round with the frozen policy")
    if (
        record["mode"] != "grounded"
        or record["profile"] != "smoke"
        or record["budget"]["model_request_attempts"] != 0
    ):
        raise ValueError("round runner only authorizes offline grounded/smoke work")
    if current.get("development") is not None:
        from .search import verify_selection

        verify_selection(root, record)
    reservation = {
        "round_id": current["id"],
        "batch_id": directory.name,
        "role": role,
        "budget": batch.execution_budget(record),
    }
    _append(root, "reserve_run", reservation)
    return reservation


def verify_reservation(
    root: Path, directory: Path, role: str, started: dict, record: dict
) -> None:
    state, _ = _read(root)
    reservation = started.get("round_reservation")
    if not isinstance(reservation, dict):
        raise ValueError("run lacks a frozen round budget reservation")
    rounds = state["round_history"] + ([state["round"]] if state["round"] else [])
    matches = [
        r for current in rounds for r in current["reservations"] if r == reservation
    ]
    if (
        len(matches) != 1
        or reservation["batch_id"] != directory.name
        or reservation["role"] != role
        or reservation["budget"] != batch.execution_budget(record)
    ):
        raise ValueError("run reservation differs from the round journal")
    current = next(r for r in rounds if r["id"] == reservation["round_id"])
    if (
        current.get("development") is not None
        and record.get("purpose") != "development"
    ):
        from .search import verify_selection

        verify_selection(root, record, current=current)


def attach(root: Path, directory: Path) -> dict:
    """Bind selection before either side exposes acceptance results."""
    if directory.resolve().parent.parent != root.resolve():
        raise ValueError("batch must belong to this registry")
    with batch.locked(root):
        state, _ = _read(root)
        current = state.get("round")
        if current is None:
            raise ValueError("start a budgeted round before selecting a candidate")
        record = batch.read_registration(directory)
        if record.get("purpose") == "development":
            raise ValueError("development batch cannot be attached as fresh acceptance")
        if record["profile"] != "smoke" or (directory / "confirmation.json").exists():
            raise ValueError(
                "development selection requires smoke; seal production confirmations separately"
            )
        if current["policy_digest"] != record["policy_digest"]:
            raise ValueError("policy changed since the round was frozen")
        if any(
            (directory / name).exists()
            for name in ("before.started.json", "after.started.json", "decision.json")
        ):
            raise ValueError("select the candidate before either acceptance run starts")
        search_binding = None
        if current.get("development") is not None:
            from .search import verify_selection

            search_binding = verify_selection(root, record)
        for frozen in record["versions"].values():
            if _frozen(frozen["spec"]) != frozen:
                raise ValueError("version changed since registration")
        return _append(
            root,
            "attach",
            {
                "batch_id": directory.name,
                "versions": record["versions"],
                "policy_digest": record["policy_digest"],
                "round_id": current["id"],
                "role_budget": batch.execution_budget(record),
                **({"development_selection": search_binding} if search_binding else {}),
            },
        )


def resolve(root: Path) -> dict:
    with batch.locked(root):
        state, _ = _read(root)
        pending = state.get("pending")
        if pending is None:
            raise ValueError("no pending batch")
        if pending["policy_digest"] != digest(load_policy()):
            raise ValueError("policy changed since selection")
        directory = root / "batches" / pending["batch_id"]
        record = batch.read_registration(directory)
        if record["versions"] != pending["versions"]:
            raise ValueError("registered pair changed since selection")
        decision = batch.verified_decision(directory)
        for frozen in pending["versions"].values():
            if _frozen(frozen["spec"]) != frozen:
                raise ValueError("version changed before lifecycle transition")
        goals = {key: value["status"] for key, value in decision["goals"].items()}
        promote = all(goals[key] == "PASS" for key in ("G1", "G2"))
        return _append(
            root,
            "resolve",
            {
                "batch_id": directory.name,
                "decision_digest": digest(decision),
                "goals": goals,
                "decision": "PROMOTE_DEVELOPMENT" if promote else "KEEP_INCUMBENT",
            },
        )


def rollback(root: Path, reason: str) -> dict:
    if not reason.strip():
        raise ValueError("record the rollback reason")
    with batch.locked(root):
        state, _ = _read(root)
        if not state.get("rollback"):
            raise ValueError("no retained incumbent to restore")
        restore = state["rollback"][-1]
        if _frozen(restore["spec"])["identity"] != restore["identity"]:
            raise ValueError("retained rollback version is no longer reproducible")
        return _append(root, "rollback", {"restore": restore, "reason": reason})


def abandon(root: Path, reason: str) -> dict:
    """Close an interrupted batch without retrying it or releasing its events."""
    if not reason.strip():
        raise ValueError("record the abandonment reason")
    with batch.locked(root):
        state, _ = _read(root)
        if not state.get("pending"):
            raise ValueError("no pending batch")
        return _append(
            root,
            "abandon",
            {"batch_id": state["pending"]["batch_id"], "reason": reason},
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("initialize")
    init.add_argument("--version", type=Path, required=True)
    select = commands.add_parser("attach")
    select.add_argument("--batch", type=Path, required=True)
    commands.add_parser("resolve")
    commands.add_parser("status")
    start = commands.add_parser("start-round")
    start.add_argument("--hypothesis", required=True)
    start.add_argument("--budget", type=Path, required=True)
    start.add_argument("--development-manifest", type=Path)
    start.add_argument("--seeds", nargs="+", type=int)
    commands.add_parser("close-round").add_argument("--reason", required=True)
    for name in ("rollback", "abandon"):
        commands.add_parser(name).add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    try:
        root = args.registry.resolve()
        if args.command == "initialize":
            require_external(args.version, public_roots())
            result = initialize(root, json.loads(args.version.read_text()))
        elif args.command == "attach":
            result = attach(root, args.batch.resolve())
        elif args.command == "resolve":
            result = resolve(root)
        elif args.command == "rollback":
            result = rollback(root, args.reason)
        elif args.command == "abandon":
            result = abandon(root, args.reason)
        elif args.command == "start-round":
            require_external(args.budget, public_roots())
            result = start_round(
                root,
                args.hypothesis,
                json.loads(args.budget.read_text()),
                development_manifest=args.development_manifest,
                seeds=args.seeds,
            )
        elif args.command == "close-round":
            result = close_round(root, args.reason)
        else:
            result = status(root)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"Lifecycle refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
