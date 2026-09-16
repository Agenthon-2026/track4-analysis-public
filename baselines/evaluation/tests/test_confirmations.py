"""Two confirmations cannot hide a failed batch or a changed comparison pair."""

import copy

import pytest

from baselines.evaluation import confirmations


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    paths = [tmp_path / "batches" / name for name in ("selection", "first", "second")]
    records, decisions, reports = {}, {}, {}
    for index, path in enumerate(paths):
        record = {
            "versions": {
                "before": {"identity": "original-baseline"},
                "after": {"identity": "selected-candidate"},
            },
            "seeds": [1, 2, 3],
            "mode": "grounded",
            "policy_digest": "policy",
            "profile": "smoke" if index == 0 else "production",
            "expected_runs": [
                {"group": f"event-{index}", "input_digest": f"input-{index}"}
            ],
        }
        if index:
            record["production_judge"] = {"spec_sha256": "same-test-judge"}
        records[path] = record
        # Test doubles for the already-verified evidence boundary. These tiny
        # fixtures are not a policy-sized measurement or a production model.
        decisions[path] = {
            "goals": {goal: {"status": "PASS", "checks": []} for goal in ("G1", "G2")}
        }
        report = {
            "runs": [
                {
                    "case_id": f"case-{index}",
                    "seed": 1,
                    "group": f"event-{index}",
                    "target_type": "classification",
                    "domain": "test-domain",
                    "execution": {},
                    "profile": record["profile"],
                    "assessment": {
                        "nli_faithfulness": 0.9,
                        "admissible": True,
                        "faithfulness_gate_applied": True,
                    },
                }
            ]
        }
        reports[path] = (copy.deepcopy(report), copy.deepcopy(report))
    verified = []

    def decision(path):
        verified.append(path)
        return decisions[path]

    monkeypatch.setattr(confirmations.batch, "verified_decision", decision)
    monkeypatch.setattr(
        confirmations.batch,
        "verified_reports",
        lambda path: (*reports[path], records[path]),
    )
    # Seal and receipt enforcement are exercised with the real registry below.
    monkeypatch.setattr(
        confirmations,
        "verify_plan",
        lambda _: ({"confirmations": [p.name for p in paths[1:]]}, paths[0], []),
    )
    return paths, records, decisions, reports, verified


def audit(evidence):
    paths = evidence[0]
    return confirmations.audit_confirmations(paths[0], paths[1:])


def test_confirmation_evidence_cannot_self_certify_organizer_approval(evidence):
    result = audit(evidence)
    assert evidence[4] == evidence[0]
    assert result["goals"]["G1"]["status"] == result["goals"]["G2"]["status"] == "PASS"
    assert result["goals"]["G3"]["status"] == "UNMEASURED"
    assert result["decision"] == "KEEP_INCUMBENT"
    assert result["rankable"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        (
            "versions",
            {
                "before": {"identity": "selected-candidate"},
                "after": {"identity": "selected-candidate"},
            },
        ),
        ("seeds", [3, 4, 5]),
        ("policy_digest", "new-policy"),
        ("production_judge", {"spec_sha256": "different-judge"}),
        ("profile", "smoke"),
    ],
)
def test_changed_pair_policy_or_judge_is_rejected(evidence, field, value):
    paths, records, *_ = evidence
    records[paths[2]][field] = value
    with pytest.raises(ValueError):
        audit(evidence)


@pytest.mark.parametrize("field", ["group", "input_digest"])
@pytest.mark.parametrize("source", [0, 1])
def test_confirmations_cannot_reuse_selection_or_confirmation_data(
    evidence, field, source
):
    paths, records, *_ = evidence
    records[paths[2]]["expected_runs"][0][field] = records[paths[source]][
        "expected_runs"
    ][0][field]
    with pytest.raises(ValueError, match="overlap"):
        audit(evidence)


@pytest.mark.parametrize("status", ["FAIL", "UNMEASURED"])
def test_one_batch_cannot_hide_the_other_batch_quality(evidence, status):
    paths, _, decisions, *_ = evidence
    decisions[paths[1]]["goals"]["G2"]["status"] = status
    result = audit(evidence)
    assert result["goals"]["G2"]["status"] == status
    assert result["goals"]["G3"]["status"] == status


def test_baseline_production_faithfulness_also_must_pass(evidence):
    paths, _, _, reports, _ = evidence
    reports[paths[1]][0]["runs"][0]["assessment"]["admissible"] = False
    assert audit(evidence)["goals"]["G3"]["status"] == "FAIL"


def test_failed_evidence_reverification_propagates(evidence, monkeypatch):
    def corrupt(path):
        raise ValueError("saved decision differs from registered evidence")

    monkeypatch.setattr(confirmations.batch, "verified_decision", corrupt)
    with pytest.raises(ValueError, match="registered evidence"):
        audit(evidence)


def test_missing_duplicate_or_cross_registry_batches_are_rejected(evidence):
    paths = evidence[0]
    for confirmations_ in (
        [paths[1]],
        [paths[1], paths[1]],
        [paths[1], paths[2].parent / "other" / "second"],
    ):
        with pytest.raises(ValueError):
            confirmations.audit_confirmations(paths[0], confirmations_)
