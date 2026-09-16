"""Internal acceptance audit; absent measurements never become promotion evidence.

Consumes evaluator reports, not participant claims of success. Measures report
integrity, engineering diagnostics and paired quality without changing lifecycle
state. Registered G2 promotion belongs to the lifecycle; production qualification
still requires external evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from .artifacts import digest, read_json_inside
from .compare import compare
from .dataset import public_roots, require_external
from .engineering import resource_observations, verify_plan_file, verify_run_plan

POLICY_PATH = Path(__file__).with_name("acceptance-policy.json")


def finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def load_policy(path: Path = POLICY_PATH) -> dict:
    policy = json.loads(path.read_text())
    integer_keys = (
        "version",
        "min_event_groups",
        "min_domains",
        "min_groups_per_stratum",
        "min_repeats",
        "confirmation_batches",
        "max_candidates_per_round",
        "bootstrap_samples",
        "bootstrap_seed",
    )
    number_keys = ("min_mean_gain", "min_stratum_gain", "max_p95_timeout_fraction")
    if set(policy) != {*integer_keys, *number_keys, "name", "target_types"}:
        raise ValueError("unknown or missing acceptance policy fields")
    if any(
        (not isinstance(policy[key], int) or isinstance(policy[key], bool))
        or policy[key] < 1
        for key in integer_keys
    ):
        raise ValueError("acceptance counts and seed must be positive integers")
    if policy["version"] != 1 or any(not finite(policy[key]) for key in number_keys):
        raise ValueError("unsupported or nonfinite acceptance policy")
    if (
        not isinstance(policy["name"], str)
        or not policy["name"]
        or policy["min_mean_gain"] <= 0
        or not -1 <= policy["min_stratum_gain"] <= 0
        or not 0 < policy["max_p95_timeout_fraction"] <= 1
        or not 100 <= policy["bootstrap_samples"] <= 100000
        or policy["target_types"] != ["classification", "regression", "ranking"]
    ):
        raise ValueError("invalid acceptance policy limits")
    return policy


def check(name: str, passed: bool | None, detail: object) -> dict:
    return {
        "name": name,
        "status": "UNMEASURED" if passed is None else "PASS" if passed else "FAIL",
        "detail": detail,
    }


def stage(checks: list[dict]) -> dict:
    states = {c["status"] for c in checks}
    status = (
        "FAIL"
        if "FAIL" in states
        else "UNMEASURED"
        if "UNMEASURED" in states
        else "PASS"
    )
    return {"status": status, "checks": checks}


def report_rows(report: dict) -> list[dict]:
    rows = report.get("runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty or missing run roster")
    keys = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("run must be an object")
        key = row.get("case_id"), row.get("seed")
        if (
            not isinstance(key[0], str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", key[0])
            or (not isinstance(key[1], int) or isinstance(key[1], bool))
            or not 0 <= key[1] < 2**32
            or key in keys
        ):
            raise ValueError("invalid or duplicate case/seed roster")
        if not isinstance(row.get("group"), str) or not row["group"]:
            raise ValueError("missing independent event group")
        if row.get("target_type") not in ("classification", "regression", "ranking"):
            raise ValueError("invalid target type")
        if row.get("domain") is not None and (
            not isinstance(row["domain"], str) or not row["domain"].strip()
        ):
            raise ValueError("invalid domain")
        if not isinstance(row.get("assessment"), dict) or not isinstance(
            row.get("execution"), dict
        ):
            raise ValueError("missing assessment or execution")
        keys.add(key)
    return rows


def request_accounting(report: dict) -> dict:
    from baselines.strong_rag_baseline.config import (
        HOUSE_MAX_OUTPUT_TOKENS,
        HOUSE_MAX_REQUESTS,
    )

    missing = 0
    total = 0
    for row in report_rows(report):
        ledger = row.get("diagnostics", {}).get("request_ledger")
        if ledger is None:
            missing += 1
            continue
        if not isinstance(ledger, dict) or ledger.get("version") != 1:
            return check("request_attempt_accounting", False, "invalid ledger")
        attempts = ledger.get("attempts")
        if not isinstance(attempts, list) or ledger.get("attempts_used") != len(
            attempts
        ):
            return check(
                "request_attempt_accounting",
                False,
                "attempt count differs from ledger rows",
            )
        if ledger.get("source") == "offline":
            if report.get("mode") != "grounded" or attempts:
                return check(
                    "request_attempt_accounting",
                    False,
                    "offline ledger cannot account for model runs",
                )
        elif ledger.get("source") == "http-client":
            limit, output_limit = (
                ledger.get("attempt_limit"),
                ledger.get("output_token_limit"),
            )
            if (
                not finite(limit)
                or not isinstance(limit, int)
                or not 1 <= limit <= HOUSE_MAX_REQUESTS
                or not finite(output_limit)
                or not isinstance(output_limit, int)
                or not 1 <= output_limit <= HOUSE_MAX_OUTPUT_TOKENS
                or len(attempts) > limit
            ):
                return check(
                    "request_attempt_accounting",
                    False,
                    "request or output cap exceeded",
                )
            for index, attempt in enumerate(attempts, 1):
                if (
                    not isinstance(attempt, dict)
                    or attempt.get("sequence") != index
                    or attempt.get("status") not in ("success", "error")
                ):
                    return check(
                        "request_attempt_accounting",
                        False,
                        "missing, duplicated or unfinished attempt",
                    )
                completion = attempt.get("completion_tokens")
                if completion is not None and (
                    not finite(completion)
                    or not isinstance(completion, int)
                    or not 0 <= completion <= output_limit
                ):
                    return check(
                        "request_attempt_accounting",
                        False,
                        "reported output exceeds cap or is invalid",
                    )
        else:
            return check("request_attempt_accounting", None, "unknown ledger source")
        total += len(attempts)
    return check(
        "request_attempt_accounting",
        None if missing else True,
        {
            "attempts": total,
            "missing_runs": missing,
            "scope": "client diagnostics; not proxy billing verification",
        },
    )


def engineering_checks(
    report: dict, policy: dict | None = None, faults: Path | None = None
) -> list[dict]:
    rows = report_rows(report)
    checks = [
        check(
            "execution",
            all(
                row.get("status") == "completed"
                and isinstance(row["execution"].get("returncode"), int)
                and not isinstance(row["execution"].get("returncode"), bool)
                and row["execution"]["returncode"] == 0
                and row["execution"].get("timed_out") is False
                and not row["execution"].get("artifact_errors")
                for row in rows
            ),
            {"runs": len(rows)},
        )
    ]
    for gate in (
        "g0_integrity",
        "g1_schema",
        "g2_cutoff_resource",
        "g3_domain_semantics",
    ):
        values = [
            row["assessment"].get("gates", {}).get(gate, {}).get("passed")
            for row in rows
        ]
        checks.append(
            check(
                gate,
                False
                if False in values
                else True
                if all(x is True for x in values)
                else None,
                {
                    "observed": sum(isinstance(x, bool) for x in values),
                    "runs": len(rows),
                    "scope": "reported gates; smoke g3 omits production faithfulness",
                },
            )
        )
    records = [row["assessment"].get("hypothesis_records") for row in rows]
    citations = [
        citation
        for record in records or []
        if isinstance(record, list)
        for entity in record
        for citation in entity.get("citations", [])
    ]
    complete_citations = all(
        isinstance(record, list)
        and record
        and all(entity.get("citations") for entity in record)
        for record in records
    )
    checks.append(
        check(
            "citation_resolution",
            all(
                c.get("span_valid") is True and c.get("embargo_clean") is True
                for c in citations
            )
            if complete_citations
            else None,
            {"citations": len(citations)},
        )
    )
    image = report.get("provenance", {}).get("image_id")
    checks.append(
        check(
            "container_execution",
            True
            if isinstance(image, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", image)
            and all(
                row["execution"].get("isolation") == "docker-network-none"
                for row in rows
            )
            else None,
            "local process or an image name alone does not establish container execution",
        )
    )
    checks.append(request_accounting(report))
    if "run_plan" in report:
        verify_run_plan(report)
    checks.append(
        check(
            "planned_roster",
            True if "run_plan" in report else None,
            "complete case/seed roster frozen by the evaluator before prediction",
        )
    )
    passed, details = resource_observations(
        rows, (policy or load_policy())["max_p95_timeout_fraction"], image
    )
    checks.append(check("resource_envelope", passed, details))
    if faults is None:
        checks.append(
            check(
                "fault_recovery",
                None,
                "needs frozen-image cold-start and fault exercise results",
            )
        )
    else:
        from .faults import load_suite

        details = load_suite(faults, report.get("provenance", {}))
        checks.append(check("fault_recovery", True, details))
    return checks


def quality_checks(
    before: dict, after: dict, policy: dict
) -> tuple[list[dict], dict | None]:
    a, b = report_rows(before), report_rows(after)
    checks = []
    heldout = all(row.get("split") == "test" for row in a + b)
    checks.append(
        check(
            "heldout_split",
            True if heldout else None,
            "only test reports can establish acceptance quality",
        )
    )
    groups: dict[str, list[dict]] = defaultdict(list)
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in b:
        groups[row["group"]].append(row)
        by_case[row["case_id"]].append(row)
    domain_groups: dict[str, set] = defaultdict(set)
    kind_groups: dict[str, set] = defaultdict(set)
    for rows in by_case.values():
        for field in ("group", "domain", "target_type"):
            if len({row.get(field) for row in rows}) != 1:
                raise ValueError(
                    f"one case cannot have inconsistent {field} across seeds"
                )
    for group, rows in groups.items():
        for row in rows:
            # A shared shock can affect several domains. Count it once overall
            # and once within each affected stratum, never once per domain overall.
            if row.get("domain"):
                domain_groups[row["domain"]].add(group)
            kind_groups[row["target_type"]].add(group)
    dimensions = {
        "event_groups": len(groups),
        "domain_groups": {
            key: len(value) for key, value in sorted(domain_groups.items())
        },
        "target_type_groups": {
            key: len(value) for key, value in sorted(kind_groups.items())
        },
    }
    enough = (
        len(groups) >= policy["min_event_groups"]
        and len(domain_groups) >= policy["min_domains"]
        and all(
            len(value) >= policy["min_groups_per_stratum"]
            for value in domain_groups.values()
        )
        and all(
            len(kind_groups[key]) >= policy["min_groups_per_stratum"]
            for key in policy["target_types"]
        )
        and all(row.get("domain") for row in b)
    )
    checks.append(
        check("independent_sample_size", True if enough else None, dimensions)
    )
    seeds = [{row["seed"] for row in rows} for rows in by_case.values()]
    same_seeds = all(value == seeds[0] for value in seeds)
    checks.append(
        check(
            "repeats",
            True if same_seeds and len(seeds[0]) >= policy["min_repeats"] else None,
            {"smallest_repeat_count": min(map(len, seeds)), "same_seeds": same_seeds},
        )
    )
    balanced = len({len(rows) for rows in groups.values()}) == 1 and same_seeds
    checks.append(
        check(
            "balanced_event_weights",
            balanced,
            "equal units/views and repeats per independent event are required",
        )
    )
    scores = [row["assessment"].get("development_score") for row in a + b]
    if any(value is not None and not finite(value) for value in scores):
        raise ValueError("nonfinite or nonnumeric measured score")
    if not all(finite(value) for value in scores) or not all(
        row.get("truth_digest") for row in a + b
    ):
        checks.append(
            check(
                "paired_quality",
                None,
                "finite scored outcomes required for every planned run",
            )
        )
        return checks, None
    # The existing comparator owns pairing, scorer identity and bootstrap mathematics.
    result = compare(
        before,
        after,
        samples=policy["bootstrap_samples"],
        seed=policy["bootstrap_seed"],
    )
    interval = result["overall"]["bootstrap_95_percent"]
    gains_pass = (
        result["overall"]["event_mean_delta"] >= policy["min_mean_gain"]
        and interval is not None
        and interval[0] > 0
    )
    checks.append(
        check(
            "paired_quality",
            gains_pass if enough and heldout else None,
            result["overall"],
        )
    )
    for dimension in ("by_target_type", "by_domain"):
        for key, summary in result[dimension].items():
            measured = (
                summary["independent_groups"] >= policy["min_groups_per_stratum"]
                and heldout
            )
            checks.append(
                check(
                    f"{dimension}.{key}",
                    summary["event_mean_delta"] >= policy["min_stratum_gain"]
                    if measured
                    else None,
                    summary,
                )
            )
    checks.append(
        check(
            "no_admission_regressions",
            result["before_failures"] == result["after_failures"] == 0,
            {"before": result["before_failures"], "after": result["after_failures"]},
        )
    )
    return checks, result


def production_checks(report: dict) -> list[dict]:
    rows = report_rows(report)
    values = [row["assessment"].get("nli_faithfulness") for row in rows]
    applied = all(
        row.get("profile") == "production"
        and row["assessment"].get("faithfulness_gate_applied") is True
        for row in rows
    )
    measured = applied and all(finite(value) and 0 <= value <= 1 for value in values)
    return [
        check(
            "production_faithfulness",
            all(row["assessment"].get("admissible") is True for row in rows)
            if measured
            else None,
            {
                "measured_runs": sum(finite(value) for value in values),
                "runs": len(rows),
                "minimum": min(values) if measured else None,
                "scope": "official gates apply each unit's trusted threshold; local report remains non-rankable",
            },
        )
    ]


def audit(
    before: dict,
    after: dict,
    *,
    policy: dict,
    before_faults: Path | None = None,
    after_faults: Path | None = None,
) -> dict:
    """Report measured subchecks and outstanding prerequisites without self-certification."""
    try:
        engineering = engineering_checks(after, policy, after_faults)
        baseline = stage(engineering_checks(before, policy, before_faults))
        quality, comparison = quality_checks(before, after, policy)
        production = production_checks(after)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return {
            "version": 1,
            "policy_digest": digest(policy),
            "rankable": False,
            "decision": "KEEP_INCUMBENT",
            "input_error": str(exc),
            "goals": {
                key: stage([check("report_integrity", False, str(exc))])
                for key in ("G1", "G2", "G3")
            },
        }
    quality.extend(
        check(name, None, reason)
        for name, reason in (
            (
                "fresh_holdout",
                "needs preregistration and one-time consumption of an independent sealed batch",
            ),
        )
    )
    quality.append(
        check(
            "baseline_engineering",
            True
            if baseline["status"] == "PASS"
            else False
            if baseline["status"] == "FAIL"
            else None,
            baseline,
        )
    )
    production.extend(
        check(name, None, reason)
        for name, reason in (
            (
                "independent_confirmations",
                "needs two non-overlapping fresh batches with the same frozen pair and judge",
            ),
            (
                "production_equivalence",
                "needs approved judge/runtime and model-service provenance bound to the candidate",
            ),
            (
                "artifact_eligibility",
                "needs cutoff, model-artifact and reproducibility verification",
            ),
        )
    )
    g1, g2 = stage(engineering), stage(quality)
    g2["checks"].append(
        check(
            "candidate_engineering",
            True
            if g1["status"] == "PASS"
            else False
            if g1["status"] == "FAIL"
            else None,
            g1["status"],
        )
    )
    g2 = stage(g2["checks"])
    production.append(
        check(
            "quality_acceptance",
            True
            if g2["status"] == "PASS"
            else False
            if g2["status"] == "FAIL"
            else None,
            g2["status"],
        )
    )
    return {
        "version": 1,
        "policy_digest": digest(policy),
        "rankable": False,
        "decision": "KEEP_INCUMBENT",
        "goals": {"G1": g1, "G2": g2, "G3": stage(production)},
        "comparison": comparison,
        "report_digests": {"before": digest(before), "after": digest(after)},
        "note": "This audit cannot promote a candidate until the outstanding evaluator evidence paths are implemented and measured.",
    }


def load_report(path: Path) -> dict:
    """Bind diagnostics to persisted per-run results and answers, detecting edited reports."""
    report = json.loads(path.read_text())
    rows = report_rows(report)
    verify_plan_file(path, report)
    for row in rows:
        folder = path.parent / row["case_id"] / f"seed-{row['seed']}"
        result = folder / "result.json"
        if result.is_symlink() or not result.resolve().is_relative_to(
            path.parent.resolve()
        ):
            raise ValueError("run result must remain inside its report directory")
        if json.loads(result.read_text()) != row:
            raise ValueError("report row differs from its persisted result")
        answer = folder / "answer.json"
        if row.get("answer_sha256"):
            if answer.is_symlink() or not answer.resolve().is_relative_to(
                path.parent.resolve()
            ):
                raise ValueError("answer must remain inside its report directory")
            if hashlib.sha256(answer.read_bytes()).hexdigest() != row["answer_sha256"]:
                raise ValueError("answer bytes differ from recorded digest")
        if "diagnostics" in row:
            relative = f"{row['case_id']}/seed-{row['seed']}/diagnostics.json"
            if read_json_inside(path.parent, relative) != row["diagnostics"]:
                raise ValueError(
                    "diagnostics differ from persisted participant artifact"
                )
            if "run_plan" in report or row.get("diagnostics_sha256"):
                if hashlib.sha256(
                    (path.parent / relative).read_bytes()
                ).hexdigest() != row.get("diagnostics_sha256"):
                    raise ValueError("diagnostics bytes differ from recorded digest")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--before-faults", type=Path)
    parser.add_argument("--after-faults", type=Path)
    args = parser.parse_args(argv)
    try:
        for path in (args.before, args.after, args.out):
            require_external(path, public_roots())
        result = audit(
            load_report(args.before),
            load_report(args.after),
            policy=load_policy(),
            before_faults=args.before_faults,
            after_faults=args.after_faults,
        )
        with args.out.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write("\n")
        print(
            json.dumps({key: value["status"] for key, value in result["goals"].items()})
        )
        return (
            1
            if any(value["status"] == "FAIL" for value in result["goals"].values())
            else 2
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Acceptance aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
