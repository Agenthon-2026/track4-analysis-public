"""Citation resolution is a dictionary lookup, and the embargo gate fails closed.

Two measured defects, opposite in direction.

**Traversal.** The official scorer interpolated the participant's ``doc_id`` into a path:
``unit_dir / "corpus" / f"{doc_id}.json"``. A ``doc_id`` of ``../../secret_ref/oracle`` read a
sentinel planted outside the unit directory and returned it as a corpus document. The test below
plants exactly such a sentinel and requires that nothing reads it.

**Fail-open dates.** If the corpus could not supply a ``doc_date``, the shared helper fell back to
the citation's own self-reported date; the answer schema defines no such field, so the fallback is
always absent and **an unresolvable citation was embargo-clean by construction**. Dates were then
compared as raw strings under a comment that said "lexical == chronological", which is true of
exactly one spelling.

The corpus index refuses unresolved, undated, malformed-date and post-cutoff citations, and parses
every date through one policy. The positive control is a legitimate pre-cutoff citation, which must
still resolve and still pass.

**Two layers, both asserted.** Track 4's ``CorpusIndex.embargo_report`` runs first and is defence
in depth; the shared ``qfbench2_common.scoring.faithfulness.embargo_violations`` behind it now
fails closed too. The last section of this file exercises the shared layer directly rather than
only through ours, because "unreachable from here" is not "tested from here" — that is
precisely how the shared helper stayed fail-open for as long as it did.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib

import pytest

import qfbench2_common.scoring.faithfulness as F
from qfbench2_common.contracts import OrganizerFault

from qfbench2_track_analysis.codes import (
    T4OrganizerFault,
    T4ParticipantFailure,
    T4Reason,
)
from qfbench2_track_analysis.corpus import ISO_DATE_RE, CorpusIndex, parse_iso_date

from .synthetic import CUTOFF, POST_CUTOFF_DOC, PRE_CUTOFF_DOC, build_unit

CUTOFF_DATE = dt.date.fromisoformat(CUTOFF)


def _index(tmp_path: pathlib.Path) -> CorpusIndex:
    return CorpusIndex.from_unit(build_unit(tmp_path))


# --- positive control -------------------------------------------------------------------------
def test_a_declared_document_resolves_with_its_trusted_date(
    tmp_path: pathlib.Path,
) -> None:
    index = _index(tmp_path)
    doc = index.resolve(PRE_CUTOFF_DOC)
    assert doc.doc_id == PRE_CUTOFF_DOC
    assert doc.doc_date == dt.date(2026, 2, 1)
    assert index.embargo_report([{"doc_id": PRE_CUTOFF_DOC}], CUTOFF_DATE).clean is True


# --- traversal --------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "hostile",
    [
        "../../secret_ref/oracle",
        "../secret",
        "/etc/passwd",
        "corpus/../../secret",
        "..%2f..%2fsecret",
        "SYNTHDOC_PRE_20260201/../../secret",
        "",
        ".",
        "..",
    ],
)
def test_a_hostile_doc_id_resolves_to_nothing(
    tmp_path: pathlib.Path, hostile: str
) -> None:
    """The sentinel is real and readable; the point is that the lookup never reaches it."""
    sentinel = tmp_path / "secret_ref"
    sentinel.mkdir()
    (sentinel / "oracle.json").write_text(
        json.dumps({"doc_id": "oracle", "doc_date": "2999-01-01", "text": "SENTINEL"}),
        encoding="utf-8",
    )
    (tmp_path / "secret.json").write_text(
        json.dumps({"doc_id": "secret", "doc_date": "2999-01-01", "text": "SENTINEL"}),
        encoding="utf-8",
    )
    index = _index(tmp_path)
    with pytest.raises(T4ParticipantFailure) as excinfo:
        index.resolve(hostile)
    assert excinfo.value.reason is T4Reason.CITATION_UNRESOLVED


def test_a_non_string_doc_id_is_refused(tmp_path: pathlib.Path) -> None:
    index = _index(tmp_path)
    for hostile in (None, 42, ["a"], {"doc_id": "a"}):
        with pytest.raises(T4ParticipantFailure):
            index.resolve(hostile)


def test_a_unicode_denormalized_doc_id_does_not_alias_a_real_one(
    tmp_path: pathlib.Path,
) -> None:
    """NFD 'e' + combining acute must not resolve to the NFC document of the same name."""
    docs = {
        "CAFÉ_DOC": {
            "doc_id": "CAFÉ_DOC",
            "doc_date": "2026-01-05",
            "text": "synthetic",
        }
    }
    # The manifest path itself must be NFC; a denormalized organizer path is a staging fault.
    with pytest.raises(T4OrganizerFault, match="NFC"):
        CorpusIndex.from_unit(build_unit(tmp_path, docs=docs))


# --- links ------------------------------------------------------------------------------------
def test_a_symlinked_corpus_document_is_an_organizer_fault_not_a_redirect(
    tmp_path: pathlib.Path,
) -> None:
    unit = build_unit(tmp_path)
    target = tmp_path / "outside.json"
    target.write_text(
        json.dumps(
            {"doc_id": PRE_CUTOFF_DOC, "doc_date": "2026-02-01", "text": "SENTINEL"}
        ),
        encoding="utf-8",
    )
    doc_path = unit / "corpus" / f"{PRE_CUTOFF_DOC}.json"
    doc_path.unlink()
    os.symlink(target, doc_path)
    with pytest.raises(T4OrganizerFault):
        CorpusIndex.from_unit(unit)


def test_a_corpus_document_that_does_not_match_its_manifest_digest_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    unit = build_unit(tmp_path)
    doc_path = unit / "corpus" / f"{PRE_CUTOFF_DOC}.json"
    doc_path.write_text(
        json.dumps(
            {"doc_id": PRE_CUTOFF_DOC, "doc_date": "2026-02-01", "text": "tampered"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(T4OrganizerFault, match="manifest digest"):
        CorpusIndex.from_unit(unit)


def test_a_file_present_but_undeclared_is_not_resolvable(
    tmp_path: pathlib.Path,
) -> None:
    unit = build_unit(tmp_path)
    (unit / "corpus" / "UNDECLARED_DOC.json").write_text(
        json.dumps({"doc_id": "UNDECLARED_DOC", "doc_date": "2026-01-01", "text": "x"}),
        encoding="utf-8",
    )
    index = CorpusIndex.from_unit(unit)
    with pytest.raises(T4ParticipantFailure):
        index.resolve("UNDECLARED_DOC")


# --- embargo ------------------------------------------------------------------------------------
def test_post_cutoff_citation_is_a_violation(tmp_path: pathlib.Path) -> None:
    report = _index(tmp_path).embargo_report([{"doc_id": POST_CUTOFF_DOC}], CUTOFF_DATE)
    assert report.post_cutoff == 1
    assert report.clean is False


def test_unresolvable_citation_is_a_violation_not_embargo_clean(
    tmp_path: pathlib.Path,
) -> None:
    """Pre-fix: `embargo_violations` for an unresolvable doc_id returned [] — clean by default."""
    report = _index(tmp_path).embargo_report([{"doc_id": "NO_SUCH_DOC"}], CUTOFF_DATE)
    assert report.unresolved == 1
    assert report.clean is False


def test_malformed_citation_object_is_a_violation(tmp_path: pathlib.Path) -> None:
    report = _index(tmp_path).embargo_report(["not-an-object"], CUTOFF_DATE)  # type: ignore[list-item]
    assert report.malformed == 1
    assert report.clean is False


def test_a_participant_supplied_doc_date_cannot_override_the_trusted_one(
    tmp_path: pathlib.Path,
) -> None:
    report = _index(tmp_path).embargo_report(
        [{"doc_id": POST_CUTOFF_DOC, "doc_date": "1999-01-01"}], CUTOFF_DATE
    )
    assert report.post_cutoff == 1


# --- date policy ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad", ["20260201", "2026-2-1", "2026-02-31", "2026-W05-1", "", None, 20260201]
)
def test_only_one_date_spelling_is_accepted(bad: object) -> None:
    with pytest.raises(T4ParticipantFailure):
        parse_iso_date(bad, field="doc_date")


def test_a_malformed_organizer_cutoff_is_an_organizer_fault_not_a_participant_one() -> (
    None
):
    with pytest.raises(T4OrganizerFault):
        parse_iso_date("20260215", field="task.json.cutoff_date", fault="organizer")


def test_dates_compare_as_dates_not_as_strings() -> None:
    assert parse_iso_date("2026-02-09", field="d") < parse_iso_date(
        "2026-02-10", field="d"
    )


def test_an_undated_corpus_document_is_an_organizer_fault(
    tmp_path: pathlib.Path,
) -> None:
    docs = {"SYNTHDOC_NODATE": {"doc_id": "SYNTHDOC_NODATE", "text": "synthetic"}}
    with pytest.raises(T4OrganizerFault, match="doc_date"):
        CorpusIndex.from_unit(build_unit(tmp_path, docs=docs))


def test_the_corpus_index_file_is_not_itself_a_citable_document(
    tmp_path: pathlib.Path,
) -> None:
    """`corpus/manifest.json` is the index. It is manifested (every released file must be) and it
    carries no `doc_date`, so treating it as a document would make every unit an organizer fault."""
    unit = build_unit(tmp_path)
    index = unit / "corpus" / "manifest.json"
    index.write_text(
        json.dumps({"manifest_version": "2.0", "files": []}), encoding="utf-8"
    )
    manifest_path = unit / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    blob = index.read_bytes()
    manifest["files"].append(
        {
            "path": "corpus/manifest.json",
            "role": "corpus",
            "license": "CC-BY-4.0",
            "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
            "split": "public-dev",
            "redistributable": True,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    index_obj = CorpusIndex.from_unit(unit)
    assert PRE_CUTOFF_DOC in index_obj.doc_ids
    with pytest.raises(T4ParticipantFailure):
        index_obj.resolve("manifest")


# --- BOTH layers, not just ours -------------------------------------------------------------------
# Track 4's own gate above is defence in depth and it runs FIRST: `_g3_domain_semantics` calls
# `CorpusIndex.embargo_report`, which refuses an unresolvable, undated or post-cutoff citation
# before the shared helper is reached. That is deliberate and it stays.
#
# But "unreachable from here" is not "tested from here". The shared
# `qfbench2_common.scoring.faithfulness.embargo_violations` was fail-OPEN for exactly the cases our
# gate catches, and it stayed that way for as long as it did because every consumer that could have
# noticed had its own gate in front. It now fails closed. The tests below assert the shared layer
# directly, through the same trusted lookup Track 4 hands it, so a future refactor
# that drops our gate finds out whether the thing behind it actually holds — rather than inheriting
# the pre-2026-08-22 behaviour, in which citing a document that does not exist was the cheapest way
# past the embargo.
def _answer(*citations: object) -> dict[str, object]:
    return {"claims": [{"text": "synthetic claim", "citations": list(citations)}]}


def _reasons(violations: list[dict[str, object]]) -> list[object]:
    return [v["reason"] for v in violations]


def test_the_shared_helper_agrees_with_our_gate_on_a_legitimate_citation(
    tmp_path: pathlib.Path,
) -> None:
    index = _index(tmp_path)
    assert index.embargo_report([{"doc_id": PRE_CUTOFF_DOC}], CUTOFF_DATE).clean is True
    assert (
        F.embargo_violations(
            _answer({"doc_id": PRE_CUTOFF_DOC}), CUTOFF, index.lookup()
        )
        == []
    )


def test_the_shared_helper_also_refuses_an_unresolvable_citation(
    tmp_path: pathlib.Path,
) -> None:
    """The load-bearing one. Pre-fix this returned [] — indistinguishable from 'no violations'."""
    index = _index(tmp_path)
    assert index.embargo_report([{"doc_id": "NO_SUCH_DOC"}], CUTOFF_DATE).clean is False
    violations = F.embargo_violations(
        _answer({"doc_id": "NO_SUCH_DOC"}), CUTOFF, index.lookup()
    )
    assert _reasons(violations) == ["unresolved_doc"]


def test_the_shared_helper_also_refuses_a_post_cutoff_citation(
    tmp_path: pathlib.Path,
) -> None:
    index = _index(tmp_path)
    assert (
        index.embargo_report([{"doc_id": POST_CUTOFF_DOC}], CUTOFF_DATE).clean is False
    )
    violations = F.embargo_violations(
        _answer({"doc_id": POST_CUTOFF_DOC}), CUTOFF, index.lookup()
    )
    assert _reasons(violations) == ["post_cutoff"]


def test_the_shared_helper_also_refuses_a_malformed_citation(
    tmp_path: pathlib.Path,
) -> None:
    index = _index(tmp_path)
    assert index.embargo_report(["not-an-object"], CUTOFF_DATE).clean is False  # type: ignore[list-item]
    violations = F.embargo_violations(
        _answer("not-an-object", {"span_start": 0, "span_end": 1}),
        CUTOFF,
        index.lookup(),
    )
    assert _reasons(violations) == ["malformed_citation", "malformed_citation"]


def test_the_shared_helper_also_ignores_a_participant_supplied_doc_date(
    tmp_path: pathlib.Path,
) -> None:
    index = _index(tmp_path)
    assert (
        index.embargo_report(
            [{"doc_id": POST_CUTOFF_DOC, "doc_date": "1999-01-01"}], CUTOFF_DATE
        ).post_cutoff
        == 1
    )
    violations = F.embargo_violations(
        _answer({"doc_id": POST_CUTOFF_DOC, "doc_date": "1999-01-01"}),
        CUTOFF,
        index.lookup(),
    )
    assert _reasons(violations) == ["post_cutoff"]


def test_the_shared_helper_refuses_to_run_without_the_trusted_corpus() -> None:
    """No corpus_lookup is an ORGANIZER fault, not a clean verdict. This is the fail-closed edge.

    Our gate cannot express this case — `CorpusIndex.embargo_report` IS the corpus — so it is the
    one assertion here with no Track-4 counterpart, and the one most likely to be lost if the
    shared layer were only ever exercised through us.
    """
    with pytest.raises(OrganizerFault):
        F.embargo_violations(_answer({"doc_id": PRE_CUTOFF_DOC}), CUTOFF)


def test_the_shared_helper_refuses_a_malformed_organizer_cutoff(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(OrganizerFault):
        F.embargo_violations(
            _answer({"doc_id": PRE_CUTOFF_DOC}), "20260215", _index(tmp_path).lookup()
        )


def test_both_layers_parse_dates_under_one_grammar() -> None:
    """One date grammar, or the two halves of the gate can disagree about what day a doc is from."""
    assert F.ISO_DATE_RE.pattern == ISO_DATE_RE.pattern


def test_the_shared_helpers_reasons_are_countable_without_redaction() -> None:
    """Every record is `{doc_id, reason}` with no free-form text, so counts of it are publishable."""
    assert set(F.EMBARGO_REASONS) == {
        "malformed_citation",
        "unresolved_doc",
        "undated_doc",
        "post_cutoff",
    }
