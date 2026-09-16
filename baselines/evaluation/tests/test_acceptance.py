"""Acceptance must distinguish incomplete evidence from observed failure."""

import copy
import json

import pytest

from baselines.evaluation.acceptance import (
    audit,
    load_policy,
    load_report,
    quality_checks,
)
from baselines.evaluation.compare import compare


def reports(groups=60, repeats=3):
    rows = []
    for group in range(groups):
        for kind in ("classification", "regression", "ranking"):
            for seed in range(repeats):
                rows.append(
                    {
                        "case_id": f"event-{group}-{kind}",
                        "group": f"event-{group}",
                        "domain": f"domain-{group % 3}",
                        "seed": seed,
                        "split": "test",
                        "target_type": kind,
                        "profile": "smoke",
                        "status": "completed",
                        "input_digest": f"input-{group}-{kind}",
                        "truth_digest": f"truth-{group}-{kind}",
                        "elapsed_s": 1.0,
                        "execution": {
                            "returncode": 0,
                            "timed_out": False,
                            "isolation": "local-process",
                        },
                        "assessment": {
                            "development_score": 0.1,
                            "admissible": True,
                            "gates": {
                                name: {"passed": True}
                                for name in (
                                    "g0_integrity",
                                    "g1_schema",
                                    "g2_cutoff_resource",
                                    "g3_domain_semantics",
                                )
                            },
                            "hypothesis_records": [
                                {
                                    "citations": [
                                        {"span_valid": True, "embargo_clean": True}
                                    ]
                                }
                            ],
                            "scorer": {"version": "synthetic"},
                            "judge": {"mode": "smoke"},
                        },
                    }
                )
    before = {"provenance": {"toolkit_source_digest": "same"}, "runs": rows}
    after = copy.deepcopy(before)
    for row in after["runs"]:
        row["assessment"]["development_score"] = 0.12
    return before, after


def checks_by_name(checks):
    return {item["name"]: item for item in checks}


def test_real_improvement_can_pass_quality_subchecks_without_faking_g3():
    before, after = reports()
    checks, result = quality_checks(before, after, load_policy())
    assert all(item["status"] == "PASS" for item in checks)
    assert result["overall"]["event_mean_delta"] == pytest.approx(0.02)
    decision = audit(before, after, policy=load_policy())
    assert decision["decision"] == "KEEP_INCUMBENT"
    assert decision["goals"]["G3"]["status"] == "UNMEASURED"
    assert (
        checks_by_name(decision["goals"]["G3"]["checks"])["production_faithfulness"][
            "status"
        ]
        == "UNMEASURED"
    )


def test_repeating_seeds_and_target_views_cannot_make_independent_events():
    before, after = reports(groups=2, repeats=30)
    checks, result = quality_checks(before, after, load_policy())
    values = checks_by_name(checks)
    assert values["independent_sample_size"]["status"] == "UNMEASURED"
    assert values["paired_quality"]["status"] == "UNMEASURED"
    assert result["overall"]["independent_groups"] == 2


def test_mixed_splits_and_missing_domain_do_not_pass_acceptance():
    before, after = reports(groups=3)
    for report in (before, after):
        for row in report["runs"]:
            row["split"] = "calibration"
            row.pop("domain")
    checks, _ = quality_checks(before, after, load_policy())
    assert checks_by_name(checks)["heldout_split"]["status"] == "UNMEASURED"
    assert checks_by_name(checks)["independent_sample_size"]["status"] == "UNMEASURED"


@pytest.mark.parametrize("shared_events,expected", [(20, "UNMEASURED"), (60, "PASS")])
def test_common_shocks_count_once_overall_and_once_per_domain(shared_events, expected):
    before, after = reports(groups=shared_events * 3)
    for report in (before, after):
        for row in report["runs"]:
            original_event = int(row["group"].split("-")[1])
            row["group"] = f"shared-shock-{original_event // 3}"
    checks, result = quality_checks(before, after, load_policy())
    values = checks_by_name(checks)
    assert values["independent_sample_size"]["status"] == expected
    assert values["paired_quality"]["status"] == expected
    assert values["balanced_event_weights"]["status"] == "PASS"
    assert result["overall"]["independent_groups"] == shared_events
    assert all(
        s["independent_groups"] == shared_events for s in result["by_domain"].values()
    )
    assert all(
        s["independent_groups"] == shared_events
        for s in result["by_target_type"].values()
    )


@pytest.mark.parametrize("field", ["group", "domain", "target_type"])
def test_case_metadata_cannot_change_between_seeds(field):
    before, after = reports(groups=3)
    for report in (before, after):
        report["runs"][0][field] = (
            "ranking" if field == "target_type" else "inconsistent"
        )
    with pytest.raises(ValueError, match="across seeds"):
        quality_checks(before, after, load_policy())


def test_aggregate_improvement_cannot_hide_a_regressed_domain():
    before, after = reports()
    for row in after["runs"]:
        row["assessment"]["development_score"] = (
            0.08 if row["domain"] == "domain-0" else 0.3
        )
    checks, result = quality_checks(before, after, load_policy())
    assert result["overall"]["event_mean_delta"] > 0.01
    assert checks_by_name(checks)["by_domain.domain-0"]["status"] == "FAIL"


def test_mean_gain_with_interval_crossing_zero_does_not_pass():
    before, after = reports()
    for row in after["runs"]:
        row["assessment"]["development_score"] = (
            -0.27 if int(row["group"].split("-")[1]) % 2 else 0.5
        )
    checks, _ = quality_checks(before, after, load_policy())
    assert checks_by_name(checks)["paired_quality"]["status"] == "FAIL"


def test_no_truth_is_unmeasured_not_perfect():
    before, after = reports(groups=1)
    for report in (before, after):
        for row in report["runs"]:
            row["truth_digest"] = None
            row["assessment"]["development_score"] = None
    checks, result = quality_checks(before, after, load_policy())
    assert result is None
    assert checks_by_name(checks)["paired_quality"]["status"] == "UNMEASURED"


def test_failing_unit_is_not_hidden_by_missing_other_evidence():
    before, after = reports(groups=2)
    after["runs"][0]["execution"]["returncode"] = 124
    after["runs"][0]["assessment"].update(admissible=False, development_score=-0.27)
    decision = audit(before, after, policy=load_policy())
    assert decision["goals"]["G1"]["status"] == "FAIL"
    assert decision["comparison"]["after_failures"] == 1
    assert decision["comparison"]["overall"]["event_mean_delta"] < 0.02


@pytest.mark.parametrize(
    "field", ["toolkit", "judge", "domain", "missing_run", "duplicate_run"]
)
def test_incomparable_reports_cannot_yield_promotion(field):
    before, after = reports(groups=2)
    if field == "toolkit":
        after["provenance"]["toolkit_source_digest"] = "different"
    elif field == "judge":
        after["runs"][0]["assessment"]["judge"] = {"mode": "other"}
    elif field == "domain":
        after["runs"][0]["domain"] = "different"
    elif field == "missing_run":
        after["runs"].pop()
    else:
        after["runs"].append(after["runs"][0])
    decision = audit(before, after, policy=load_policy())
    assert decision["goals"]["G2"]["status"] == "FAIL"
    assert decision["decision"] == "KEEP_INCUMBENT"


def test_domains_use_event_uncertainty_not_entity_or_seed_counts():
    before, after = reports(groups=6)
    result = compare(before, after, samples=100)
    assert result["by_domain"]["domain-0"]["independent_groups"] == 2
    assert result["by_target_type"]["ranking"]["independent_groups"] == 6


def test_artifact_binding_detects_edited_report(tmp_path):
    before, _ = reports(groups=1, repeats=1)
    for row in before["runs"]:
        directory = tmp_path / row["case_id"] / f"seed-{row['seed']}"
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps(row))
    path = tmp_path / "report.json"
    path.write_text(json.dumps(before))
    assert load_report(path) == before
    before["runs"][0]["assessment"]["development_score"] = 1
    path.write_text(json.dumps(before))
    with pytest.raises(ValueError, match="persisted result"):
        load_report(path)


@pytest.mark.parametrize(
    "key,value", [("min_repeats", True), ("min_mean_gain", float("nan")), ("extra", 1)]
)
def test_invalid_policy_cannot_silently_relax_a_gate(tmp_path, key, value):
    policy = load_policy()
    policy[key] = value
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        load_policy(path)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, "0.1"])
def test_invalid_measured_score_is_failure_not_unmeasured(score):
    before, after = reports(groups=1)
    after["runs"][0]["assessment"]["development_score"] = score
    decision = audit(before, after, policy=load_policy())
    assert decision["goals"]["G2"]["status"] == "FAIL"
    assert "nonnumeric" in decision["input_error"]


def test_request_accounting_distinguishes_offline_missing_and_overbudget():
    from baselines.evaluation.acceptance import request_accounting

    before, _ = reports(groups=1, repeats=1)
    assert request_accounting(before)["status"] == "UNMEASURED"
    before["mode"] = "grounded"
    for row in before["runs"]:
        row["diagnostics"] = {
            "request_ledger": {
                "version": 1,
                "source": "offline",
                "attempts_used": 0,
                "attempts": [],
            }
        }
    assert request_accounting(before)["status"] == "PASS"
    before["mode"] = "model"
    assert request_accounting(before)["status"] == "FAIL"
    for row in before["runs"]:
        row["diagnostics"]["request_ledger"] = {
            "version": 1,
            "source": "http-client",
            "attempts_used": 26,
            "attempt_limit": 25,
            "output_token_limit": 1024,
            "attempts": [{"sequence": i + 1, "status": "error"} for i in range(26)],
        }
    assert request_accounting(before)["status"] == "FAIL"
