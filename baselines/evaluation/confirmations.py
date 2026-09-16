"""Recheck a selected pair against fresh production-profile confirmation batches.

Configuration identity is not organizer approval. This audit never promotes a
production candidate without runtime-equivalence and artifact-eligibility evidence.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import batch
from .acceptance import check, load_policy, production_checks, stage
from .artifacts import digest, read_json_inside
from .dataset import public_roots, require_external


def _batch_paths(selection: Path, confirmations: list[Path]) -> list[Path]:
    paths = [selection, *confirmations]
    if len(confirmations) != load_policy()["confirmation_batches"]:
        raise ValueError("the policy requires exactly two confirmation batches")
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("selection and confirmations must be distinct batches")
    if len({p.resolve().parent for p in paths}) != 1:
        raise ValueError("all batches must share the same event-reservation registry")
    return paths


def _pair(selection: Path, confirmations: list[Path]) -> list[dict]:
    paths = _batch_paths(selection, confirmations)
    if (selection / "confirmation.json").exists():
        raise ValueError("a confirmation cannot become another selection")
    records = [batch.read_registration(path) for path in paths]
    selected = batch.verified_decision(selection)
    if any(selected["goals"][g]["status"] != "PASS" for g in ("G1", "G2")):
        raise ValueError("selection must have reverified G1 and G2 passes")
    _validate_records(records)
    return records


def _validate_records(records: list[dict]) -> None:
    groups, inputs = set(), set()
    for record in records:
        if any(
            record[k] != records[0][k]
            for k in ("versions", "seeds", "mode", "policy_digest")
        ):
            raise ValueError("confirmations changed the frozen pair, seeds or policy")
        new_groups = {r["group"] for r in record["expected_runs"]}
        new_inputs = {r["input_digest"] for r in record["expected_runs"]}
        if groups & new_groups or inputs & new_inputs:
            raise ValueError("selection and confirmation events/inputs overlap")
        groups.update(new_groups)
        inputs.update(new_inputs)
    judge = records[1].get("production_judge")
    if not judge or any(
        r["profile"] != "production" or r.get("production_judge") != judge
        for r in records[1:]
    ):
        raise ValueError("confirmations require the same pinned production judge")


def _incumbent(selection: Path, record: dict) -> None:
    root = selection.parent.parent
    if (root / "lifecycle").exists():
        from .lifecycle import _read

        state, _ = _read(root)  # Caller already holds the registry lock.
        incumbent = state.get("incumbent", {})
        if (
            state.get("pending") is not None
            or incumbent.get("batch_id") != selection.name
            or incumbent.get("qualification") != "G2"
            or incumbent.get("identity") != record["versions"]["after"]["identity"]
        ):
            raise ValueError(
                "confirmations require the selected G2 incumbent and no pending development batch"
            )


def seal(selection: Path, confirmations: list[Path], budget: dict) -> dict:
    """Freeze both batches and their combined local execution limit before either runs."""
    batch.validate_budget(budget)
    with batch.locked(selection.parent.parent):
        records = _pair(selection, confirmations)
        _incumbent(selection, records[0])
        for path in confirmations:
            if any(
                (path / name).exists()
                for name in (
                    "before.started.json",
                    "after.started.json",
                    "decision.json",
                    "confirmation.json",
                )
            ):
                raise ValueError("seal both confirmations before any run starts")
        roles = {
            p.name: batch.execution_budget(r)
            for p, r in zip(confirmations, records[1:])
        }
        for field, limit in (
            ("runs", "max_runs"),
            ("timeout_s", "max_reserved_seconds"),
        ):
            if 2 * sum(r[field] for r in roles.values()) > budget[limit]:
                raise ValueError(f"confirmation {limit} budget exhausted")
        plan = {
            "version": 1,
            "selection": selection.name,
            "selection_decision_digest": digest(batch.verified_decision(selection)),
            "confirmations": [p.name for p in confirmations],
            "role_budgets": roles,
            "budget": budget,
            "paid_call_budget": 0,
        }
        batch.write_new(selection / "confirmation-plan.json", plan)
        for path in confirmations:
            batch.write_new(
                path / "confirmation.json",
                {"selection": selection.name, "plan_digest": digest(plan)},
            )
        return plan


def verify_plan(directory: Path) -> tuple[dict, Path, list[dict]]:
    pointer = read_json_inside(directory, "confirmation.json")
    identifier = pointer.get("selection", "")
    if not re.fullmatch(r"[0-9a-f]{64}", identifier):
        raise ValueError("invalid confirmation selection identity")
    selection = directory.parent / identifier
    plan = read_json_inside(selection, "confirmation-plan.json")
    if (
        digest(plan) != pointer.get("plan_digest")
        or plan.get("selection") != identifier
    ):
        raise ValueError("confirmation plan changed after sealing")
    names = plan["confirmations"]
    if directory.name not in names or any(
        not re.fullmatch(r"[0-9a-f]{64}", name) for name in names
    ):
        raise ValueError("run is outside the sealed confirmation pair")
    paths = [directory.parent / name for name in names]
    for path in paths:
        if read_json_inside(path, "confirmation.json") != pointer:
            raise ValueError("both confirmations must be sealed before execution")
    records = _pair(selection, paths)
    if plan["selection_decision_digest"] != digest(batch.verified_decision(selection)):
        raise ValueError("selection evidence changed after confirmation sealing")
    roles = {p.name: batch.execution_budget(r) for p, r in zip(paths, records[1:])}
    if roles != plan["role_budgets"] or plan["paid_call_budget"] != 0:
        raise ValueError("confirmation execution budget changed")
    for field, limit in (("runs", "max_runs"), ("timeout_s", "max_reserved_seconds")):
        if 2 * sum(r[field] for r in roles.values()) > plan["budget"][limit]:
            raise ValueError("confirmation execution exceeds sealed budget")
    return plan, selection, records


def reserve_run(directory: Path, role: str) -> dict:
    """Called under the registry lock; the batch started marker burns this role."""
    plan, selection, records = verify_plan(directory)
    _incumbent(selection, records[0])
    if role not in ("before", "after") or (directory / f"{role}.started.json").exists():
        raise ValueError("confirmation role already consumed or invalid")
    return {
        "plan_digest": digest(plan),
        "batch_id": directory.name,
        "role": role,
        "budget": plan["role_budgets"][directory.name],
    }


def verify_reservation(directory: Path, role: str, started: dict) -> None:
    plan, _, _ = verify_plan(directory)
    expected = {
        "plan_digest": digest(plan),
        "batch_id": directory.name,
        "role": role,
        "budget": plan["role_budgets"][directory.name],
    }
    if started.get("confirmation_reservation") != expected:
        raise ValueError("run lacks its sealed confirmation budget reservation")


def audit_confirmations(selection: Path, confirmations: list[Path]) -> dict:
    policy = load_policy()
    paths = _batch_paths(selection, confirmations)
    plan, bound_selection, _ = verify_plan(confirmations[0])
    if bound_selection.resolve() != selection.resolve() or set(
        plan["confirmations"]
    ) != {p.name for p in confirmations}:
        raise ValueError("audit differs from the preregistered confirmation pair")

    records, decisions, reports = [], [], []
    for path in paths:
        # Saved PASS values alone are never authority. Reload the immutable
        # registration, individual results, receipts, faults and current policy.
        decision = batch.verified_decision(path)
        before, after, record = batch.verified_reports(path)
        records.append(record)
        decisions.append(decision)
        reports.append((before, after))

    _validate_records(records)

    def measured(goal: str) -> dict:
        return stage(
            [
                check(
                    path.name,
                    {"PASS": True, "FAIL": False, "UNMEASURED": None}[
                        decision["goals"][goal]["status"]
                    ],
                    decision["goals"][goal],
                )
                for path, decision in zip(paths, decisions)
            ]
        )

    engineering, quality = measured("G1"), measured("G2")
    faithfulness = []
    for path, pair in zip(confirmations, reports[1:]):
        for role, report in zip(("before", "after"), pair):
            for item in production_checks(report):
                faithfulness.append({**item, "name": f"{path.name}.{role}"})
    production = stage(
        faithfulness
        + [
            check("fixed_pair_and_judge", True, "registered identities reverified"),
            check(
                "independent_confirmations",
                {"PASS": True, "FAIL": False, "UNMEASURED": None}[
                    stage([engineering, quality])["status"]
                ],
                "disjoint registered event/input identities; each batch must pass G1/G2",
            ),
            check("production_equivalence", None, "organizer evidence still required"),
            check(
                "artifact_eligibility",
                None,
                "cutoff and licensing evidence still required",
            ),
        ]
    )
    return {
        "version": 1,
        "decision": "KEEP_INCUMBENT",
        "rankable": False,
        "policy_digest": digest(policy),
        "selection": selection.name,
        "confirmations": [path.name for path in confirmations],
        "versions": records[0]["versions"],
        "production_judge": records[1]["production_judge"],
        "decision_digests": {
            path.name: digest(decision) for path, decision in zip(paths, decisions)
        },
        "goals": {"G1": engineering, "G2": quality, "G3": production},
        "scope": "registered evidence audit; event independence and approval remain provenance obligations",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--confirmations", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seal-budget", type=Path)
    args = parser.parse_args(argv)
    if args.seal_budget is not None:
        if args.out is not None:
            parser.error("sealing does not produce an audit; omit --out")
        require_external(args.seal_budget, public_roots())
        seal(
            args.selection, args.confirmations, json.loads(args.seal_budget.read_text())
        )
        return 0
    if args.out is None:
        parser.error("audit requires --out")
    require_external(args.out, public_roots())
    result = audit_confirmations(args.selection, args.confirmations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    batch.write_new(args.out, result)
    print(json.dumps({key: value["status"] for key, value in result["goals"].items()}))
    return 1 if any(g["status"] == "FAIL" for g in result["goals"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
