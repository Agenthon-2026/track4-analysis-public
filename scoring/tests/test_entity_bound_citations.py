"""The document-level entity check in ``g3_domain_semantics``.

An NLI model has no notion of which company a passage is about: read a claim about company A
against a passage from company B's filing and it returns entailment whenever the wording
matches, because an extracted span usually does not repeat the name of the company whose filing
it came from. The cited DOCUMENT does carry that identity, and the organizer labels every
document in the trusted manifest with the roster entities it is about (`entity_ids`) or as
market-wide (`shared`). These tests hold the gate to that: the right entity's claim is read by the
judge and costs nothing, a claim citing another entity's document is FALSE (5.2.0: a per-claim
penalty; it used to refuse the whole unit), a shared document admits anyone, a peer document
(`entity_ids: []`) admits nobody, and a manifest with no labels is an organizer fault rather than
a silent pass.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest


from qfbench2_track_analysis.codes import (
    T4OrganizerFault,
    T4ParticipantFailure,
    T4Reason,
)
from qfbench2_track_analysis.corpus import CorpusIndex
from qfbench2_track_analysis.scoring import ClaimReport, _g3_domain_semantics, hydrate

from .synthetic import (
    PRE_CUTOFF_DOC,
    SUPPORTING_TEXT,
    StubJudge,
    answer_for,
    build_unit,
)

ROSTER = ("SYN-A", "SYN-B")


def _ctx(
    tmp_path: pathlib.Path, answer: dict[str, Any], judge: object, **unit_kw: Any
) -> dict[str, object]:
    unit = build_unit(tmp_path, entities=ROSTER, **unit_kw)
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    ctx: dict[str, object] = {"unit_dir": unit, "output_dir": out, "judge": judge}
    hydrate(ctx)
    ctx["_answer"] = answer
    return ctx


def _report(ctx: dict[str, object]) -> ClaimReport:
    report = ctx["_claim_report"]
    assert isinstance(report, ClaimReport)
    return report


def _labels(*entity_ids: str) -> dict[str, dict[str, Any]]:
    return {PRE_CUTOFF_DOC: {"entity_ids": list(entity_ids)}}


def test_a_citation_to_the_entitys_own_document_reaches_the_judge(
    tmp_path: pathlib.Path,
) -> None:
    judge = StubJudge((SUPPORTING_TEXT,))
    ctx = _ctx(tmp_path, answer_for(entities=ROSTER), judge, labels=_labels(*ROSTER))
    assert _g3_domain_semantics(ctx).passed
    assert ctx["_faithfulness"] == 1.0
    assert (
        judge.calls
    )  # the judge was consulted: the entity check admitted every citation


def test_a_citation_to_another_entitys_document_makes_that_claim_false(
    tmp_path: pathlib.Path,
) -> None:
    """Both entities cite the one document; it is about SYN-B only. SYN-A's claim is false, and
    only it: the unit is admitted with half its claims false (factor 0.5 at k = 1)."""
    judge = StubJudge((SUPPORTING_TEXT,))
    ctx = _ctx(tmp_path, answer_for(entities=ROSTER), judge, labels=_labels("SYN-B"))
    assert _g3_domain_semantics(ctx).passed
    report = _report(ctx)
    assert [(v.entity_id, v.wrong_entity) for v in report.verdicts] == [
        ("SYN-A", True),
        ("SYN-B", False),
    ]
    assert report.wrong_entity_claim_count == 1
    assert ctx["_wrong_entity_citations"] == 1
    assert ctx["_faithfulness"] == 0.5
    assert ctx["_faithfulness_factor"] == 0.5


def test_a_shared_document_admits_every_entity(tmp_path: pathlib.Path) -> None:
    """The default label of the synthetic fixture is `shared`: this is the shared-corpus limit."""
    judge = StubJudge((SUPPORTING_TEXT,))
    ctx = _ctx(tmp_path, answer_for(entities=ROSTER), judge)
    assert _g3_domain_semantics(ctx).passed
    # Both entities reached the judge (each in the three scored phrasings).
    assert {"SYN-A", "SYN-B"} <= {
        eid for _premise, hyp in judge.calls for eid in ROSTER if eid in hyp
    }


def test_a_peer_document_admits_nobody(tmp_path: pathlib.Path) -> None:
    """`entity_ids: []` -- about someone off the roster, so it admits nobody on it."""
    judge = StubJudge((SUPPORTING_TEXT,))
    ctx = _ctx(tmp_path, answer_for(entities=ROSTER), judge, labels=_labels())
    assert _g3_domain_semantics(ctx).passed
    assert ctx["_wrong_entity_citations"] == 2
    assert _report(ctx).false_count == 2
    assert ctx["_faithfulness_factor"] == 0.0


def test_a_manifest_with_no_labels_is_an_organizer_fault_not_a_pass(
    tmp_path: pathlib.Path,
) -> None:
    judge = StubJudge((SUPPORTING_TEXT,))
    ctx = _ctx(tmp_path, answer_for(entities=ROSTER), judge, unlabelled=True)
    with pytest.raises(T4OrganizerFault, match="no entity labels"):
        _g3_domain_semantics(ctx)
    assert judge.calls == []


def test_the_entity_check_runs_after_the_embargo_and_before_the_judge(
    tmp_path: pathlib.Path,
) -> None:
    """Order is part of the contract: a post-cutoff citation is still reported as such."""
    from .synthetic import POST_CUTOFF_DOC

    judge = StubJudge()
    labels = {POST_CUTOFF_DOC: {"entity_ids": ["SYN-B"]}}
    ctx = _ctx(
        tmp_path,
        answer_for(entities=ROSTER, doc_id=POST_CUTOFF_DOC),
        judge,
        labels=labels,
    )
    with pytest.raises(T4ParticipantFailure) as excinfo:
        _g3_domain_semantics(ctx)
    assert excinfo.value.reason is T4Reason.CITATION_POST_CUTOFF


def test_the_smoke_path_still_applies_the_entity_check(tmp_path: pathlib.Path) -> None:
    """The check is deterministic organizer data, not a judge number: the lexical smoke path
    charges the same wrong-entity claim the production path charges (it skips only the judge's
    contradiction verdicts)."""
    judge = StubJudge((SUPPORTING_TEXT,), contradicted_premises=(SUPPORTING_TEXT,))
    ctx = _ctx(tmp_path, answer_for(entities=ROSTER), judge, labels=_labels("SYN-B"))
    ctx["_enforce_faithfulness"] = False
    assert _g3_domain_semantics(ctx).passed
    # Both claims are contradicted by the stub, SYN-A's is also wrong-entity: production would
    # charge both (factor 0), smoke charges only the deterministic one (factor 0.5).
    assert _report(ctx).false_count == 2
    assert ctx["_faithfulness_gate_applied"] is False
    assert ctx["_faithfulness_factor"] == 0.5


@pytest.mark.parametrize(
    "entry_patch, message",
    [
        ({"shared": False}, "must be true"),
        ({"entity_ids": "SYN-A"}, "list of non-empty strings"),
        ({"entity_ids": ["SYN-A", "SYN-A"]}, "repeats"),
        ({"entity_ids": ["SYN-A"], "shared": True}, "one or the other"),
    ],
)
def test_a_malformed_label_is_an_organizer_fault_at_index_build(
    tmp_path: pathlib.Path, entry_patch: dict[str, Any], message: str
) -> None:
    unit = build_unit(tmp_path, entities=ROSTER, unlabelled=True)
    manifest = json.loads((unit / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"][0].update(entry_patch)
    with pytest.raises(T4OrganizerFault, match=message):
        CorpusIndex.from_unit(unit, manifest=manifest)


def test_trusted_doc_carries_the_label_it_was_declared_with(
    tmp_path: pathlib.Path,
) -> None:
    unit = build_unit(tmp_path, entities=ROSTER, labels=_labels("SYN-B"))
    index = CorpusIndex.from_unit(unit)
    doc = index.resolve(PRE_CUTOFF_DOC)
    assert doc.entity_ids == ("SYN-B",) and not doc.shared
    assert doc.admits("SYN-B") and not doc.admits("SYN-A")
    assert index.labelled
    bare = CorpusIndex.from_unit(
        build_unit(tmp_path / "u2", entities=ROSTER, unlabelled=True)
    )
    assert not bare.labelled


def test_a_citation_to_an_unlabelled_document_in_a_labelled_manifest_is_an_organizer_fault(
    tmp_path: pathlib.Path,
) -> None:
    """One forgotten label must not become the participant's wrong document."""
    from .synthetic import POST_CUTOFF_DOC

    judge = StubJudge((SUPPORTING_TEXT,))
    unit = build_unit(tmp_path, entities=ROSTER, labels=_labels("SYN-A"))
    manifest = json.loads((unit / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["path"] == f"corpus/{POST_CUTOFF_DOC}.json":
            entry.pop(
                "shared", None
            )  # this one is now unlabelled; PRE_CUTOFF_DOC keeps SYN-A
    (unit / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    index = CorpusIndex.from_unit(unit)
    assert index.labelled and index.unlabelled_doc_ids == (POST_CUTOFF_DOC,)
    # A citation to the LABELLED document still behaves normally.
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    answer = answer_for(entities=ROSTER)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    ctx: dict[str, object] = {"unit_dir": unit, "output_dir": out, "judge": judge}
    hydrate(ctx)
    ctx["_answer"] = answer
    # SYN-B cites SYN-A's document: still the participant's fault, now a false claim.
    assert _g3_domain_semantics(ctx).passed
    assert [v.wrong_entity for v in _report(ctx).verdicts] == [False, True]
    judge.calls.clear()
    # A citation to the UNLABELLED document is the organizer's fault, not a refusal.
    cutoff_free = build_unit(
        tmp_path / "u2", entities=ROSTER, cutoff="2027-01-01", labels=_labels(*ROSTER)
    )
    manifest = json.loads((cutoff_free / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["path"] == f"corpus/{POST_CUTOFF_DOC}.json":
            entry.pop("shared", None)
    (cutoff_free / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    out2 = tmp_path / "res2"
    out2.mkdir(exist_ok=True)
    answer2 = answer_for(entities=ROSTER, doc_id=POST_CUTOFF_DOC)
    (out2 / "answer.json").write_text(json.dumps(answer2), encoding="utf-8")
    ctx2: dict[str, object] = {
        "unit_dir": cutoff_free,
        "output_dir": out2,
        "judge": judge,
    }
    hydrate(ctx2)
    ctx2["_answer"] = answer2
    with pytest.raises(
        T4OrganizerFault, match="unlabelled document in a labelled manifest"
    ):
        _g3_domain_semantics(ctx2)
    assert judge.calls == []
