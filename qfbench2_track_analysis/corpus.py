"""Trusted corpus resolution: dictionary lookup, no-follow reads, one date policy.

A citation names a ``doc_id``. The scorer used to turn that string into a path —
``unit_dir / "corpus" / f"{doc_id}.json"`` — which made the participant the author of a filesystem
path inside the organizer's tree. A measured ``doc_id`` of ``../../secret_ref/oracle`` read a
sentinel planted outside the unit directory. This module replaces path interpolation with a
**dictionary built from the unit's trusted manifest**, so an unknown id resolves to nothing rather
than to somewhere else.

Four properties, each of which was absent before:

1. **Grammar.** A ``doc_id`` is one path component matching a strict character class. Separators,
   ``..``, absolute paths, encoded separators and NFC-denormalized forms are refused before any
   lookup happens, and refused again by the fact that lookup is a dict.
2. **Membership.** Only ids declared ``role: corpus`` in the unit manifest exist. A file that
   happens to sit in ``corpus/`` but is not declared is not resolvable.
3. **No-follow, digest-verified reads.** Documents are opened relative to a directory descriptor
   with ``O_NOFOLLOW`` and must be regular files whose bytes match the manifest ``sha256``. A
   symlink at a declared name is an organizer fault, not a silent redirect.
4. **One date policy.** ``doc_date`` is an ISO-8601 *calendar* date, parsed to ``datetime.date``
   and compared as a date. The previous code compared raw strings and said so in a comment
   ("lexical == chronological"), which is true only for one exact spelling.

The embargo verdict **fails closed in every direction**: a citation that does not resolve, resolves
to a document with no ``doc_date``, or carries a malformed date is a participant failure. Before
this, an unresolvable citation was embargo-clean by construction, because the answer schema defines
no ``doc_date`` field for the fallback the shared helper reaches for.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import stat
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .codes import T4OrganizerFault, T4ParticipantFailure, T4Reason

__all__ = [
    "DOC_ID_RE",
    "ISO_DATE_RE",
    "MAX_DOC_BYTES",
    "TASK_DOC_ID",
    "CorpusIndex",
    "EmbargoReport",
    "TrustedDoc",
    "parse_iso_date",
    "task_table_text",
]

#: One path component, no separators, no leading dot. Deliberately the same shape as the hub's
#: `HANDLE_RE`: a doc_id becomes a filename, and two grammars for one identifier is how a name the
#: index accepts becomes a path the filesystem reads differently.
DOC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

#: The ONE accepted spelling of a date anywhere in Track 4. `date.fromisoformat` alone is too
#: permissive on 3.11+ — it accepts `20240201` and ISO week dates — and two spellings of one day
#: is how a "lexical == chronological" comparison stops being either.
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: The reserved doc_id of the unit's TASK TABLE (5.2.x). A claim may cite a figure
#: the task gives (a row of `task.json` `entities`) the same way it cites a corpus passage:
#: ``{"doc_id": "task", "span_start": s, "span_end": e}``. The text such a span indexes is
#: `task_table_text(task)`: one line per `entities` row, in task.json order, each row written as
#: ``json.dumps(row, ensure_ascii=False, separators=(", ", ": "))`` (keys in file order), lines
#: joined by a single "\n", no trailing newline. Offsets are Python string (code point) offsets
#: into that text, the convention of a corpus document's `text`. A task span must lie inside ONE
#: row, the row of the entity whose claim cites it (the entity check); a span that crosses a line
#: or names another entity's row is a wrong-entity citation.
TASK_DOC_ID = "task"

#: The corpus directory's own index file. Manifested like everything else, never a citable doc.
CORPUS_INDEX_FILENAME = "manifest.json"

#: Per-document byte cap. A corpus document is organizer material, so this is a tripwire on a
#: staging mistake rather than a defence against a participant, but an unbounded read into a judge
#: prompt is a resource path either way.
MAX_DOC_BYTES = 8 * 1024 * 1024


def parse_iso_date(value: Any, *, field: str, fault: str = "participant") -> _dt.date:
    """Parse an ISO-8601 calendar date under the single canonical policy.

    `fault="organizer"` is used for organizer-authored values (the unit cutoff): a malformed
    cutoff is our mistake and must abort, never charge a participant.
    """
    if not isinstance(value, str) or not ISO_DATE_RE.match(value):
        if fault == "organizer":
            raise T4OrganizerFault(
                f"{field} must be an ISO-8601 calendar date 'YYYY-MM-DD', got {value!r}"
            )
        raise T4ParticipantFailure(
            T4Reason.CITATION_UNDATED,
            f"{field} must be an ISO-8601 calendar date 'YYYY-MM-DD'",
            violation_count=1,
        )
    try:
        return _dt.date.fromisoformat(value)
    except ValueError as exc:  # e.g. 2024-02-31
        if fault == "organizer":
            raise T4OrganizerFault(
                f"{field} is not a real calendar date: {value!r}"
            ) from exc
        raise T4ParticipantFailure(
            T4Reason.CITATION_UNDATED,
            f"{field} is not a real calendar date",
            violation_count=1,
        ) from exc


@dataclass(frozen=True, slots=True)
class TrustedDoc:
    """One corpus document as the organizer committed it, not as the participant described it."""

    doc_id: str
    relative_path: str
    digest: str
    doc_date: _dt.date
    document: Mapping[str, Any]
    #: Which roster entities the organizer says this document is ABOUT (`entity_ids` on the
    #: manifest entry). ``()`` means "about someone off the roster" (a peer, a benchmark): no
    #: entity may cite it. ``None`` means the manifest carries no label for this document.
    entity_ids: tuple[str, ...] | None = None
    #: A market-wide document (`shared: true` on the manifest entry): any entity may cite it.
    shared: bool = False
    #: Only on the task table (`TASK_DOC_ID`): entity_id -> the [start, end) line of its row.
    row_ranges: Mapping[str, tuple[int, int]] | None = None

    def admits(self, entity_id: str) -> bool:
        """May a prediction for `entity_id` cite this document? Shared, or listed on it. The task
        table admits an entity that has a row; WHICH slice it may cite is `admits_citation`."""
        if self.row_ranges is not None:
            return entity_id in self.row_ranges
        return self.shared or (
            self.entity_ids is not None and entity_id in self.entity_ids
        )

    def admits_citation(self, entity_id: str, cite: Mapping[str, Any]) -> bool:
        """`admits`, and for the task table also: the cited slice lies inside the entity's own row.
        Offsets that are missing or not ints are left to the span readers (they name no passage)."""
        if not self.admits(entity_id):
            return False
        if self.row_ranges is None:
            return True
        start, end = cite.get("span_start"), cite.get("span_end")
        if not (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
        ):
            return True
        row_start, row_end = self.row_ranges[entity_id]
        return row_start <= start < end <= row_end


@dataclass(frozen=True, slots=True)
class EmbargoReport:
    """Counts only. Which documents a submission cited is a per-unit private diagnostic."""

    checked: int
    malformed: int
    unresolved: int
    undated: int
    post_cutoff: int

    @property
    def clean(self) -> bool:
        return not (
            self.malformed or self.unresolved or self.undated or self.post_cutoff
        )

    @property
    def violation_count(self) -> int:
        return self.malformed + self.unresolved + self.undated + self.post_cutoff


def _sha256_file(fd: int) -> tuple[str, bytes]:
    hasher = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_DOC_BYTES:
            raise T4OrganizerFault(
                f"a corpus document exceeds the {MAX_DOC_BYTES} byte cap; refusing to read further"
            )
        hasher.update(chunk)
        chunks.append(chunk)
    return "sha256:" + hasher.hexdigest(), b"".join(chunks)


def cls_labels(
    entry: Mapping[str, Any], path: str
) -> tuple[tuple[str, ...] | None, bool]:
    """``(entity_ids, shared)`` from one manifest entry, validated. Neither present = unlabelled."""
    raw_ids = entry.get("entity_ids")
    raw_shared = entry.get("shared")
    if raw_shared is not None and raw_shared is not True:
        raise T4OrganizerFault(
            f"manifest entry {path!r}: `shared` must be true when present, got {raw_shared!r}"
        )
    shared = raw_shared is True
    entity_ids: tuple[str, ...] | None = None
    if raw_ids is not None:
        if not isinstance(raw_ids, list) or not all(
            isinstance(x, str) and x for x in raw_ids
        ):
            raise T4OrganizerFault(
                f"manifest entry {path!r}: `entity_ids` must be a list of non-empty strings"
            )
        if len(set(raw_ids)) != len(raw_ids):
            raise T4OrganizerFault(
                f"manifest entry {path!r}: `entity_ids` repeats an id"
            )
        entity_ids = tuple(raw_ids)
    if shared and entity_ids:
        raise T4OrganizerFault(
            f"manifest entry {path!r} is both `shared` and entity-labelled; a document is one or "
            "the other"
        )
    return entity_ids, shared


def task_table_text(task: Mapping[str, Any]) -> tuple[str, dict[str, tuple[int, int]]]:
    """The text a ``doc_id: "task"`` span indexes, and each entity's row as [start, end).

    See `TASK_DOC_ID` for the published definition; participants rebuild the same text with the
    same one-line `json.dumps` (the local check prints it)."""
    rows = task.get("entities")
    if not isinstance(rows, list) or not rows:
        raise T4OrganizerFault("task.json carries no entities[] table")
    lines: list[str] = []
    ranges: dict[str, tuple[int, int]] = {}
    offset = 0
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("entity_id"), str):
            raise T4OrganizerFault(
                "task.json entities[] has a row without an entity_id"
            )
        line = json.dumps(row, ensure_ascii=False, separators=(", ", ": "))
        if row["entity_id"] in ranges:
            raise T4OrganizerFault("task.json entities[] names an entity twice")
        ranges[row["entity_id"]] = (offset, offset + len(line))
        lines.append(line)
        offset += len(line) + 1
    return "\n".join(lines), ranges


class CorpusIndex:
    """``doc_id -> TrustedDoc``, built from the unit's manifest and nothing else.

    Construction is eager and total: every declared corpus file is opened no-follow, digest-checked
    and parsed at build time, so a staging fault surfaces as an organizer fault before any
    participant bytes are looked at, rather than as a per-citation surprise in the middle of
    scoring.
    """

    def __init__(self, docs: Mapping[str, TrustedDoc]) -> None:
        self._docs = dict(docs)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_unit(
        cls, unit_dir: Path, *, manifest: Mapping[str, Any] | None = None
    ) -> CorpusIndex:
        unit_dir = Path(unit_dir)
        if manifest is None:
            manifest = cls._read_manifest(unit_dir)
        entries = cls._declared_corpus_entries(manifest)
        if not entries:
            raise T4OrganizerFault(
                f"unit {unit_dir.name!r} declares no corpus files in its manifest; a Track-4 "
                "analysis unit without a corpus cannot have its citations resolved"
            )
        corpus_dir = unit_dir / "corpus"
        try:
            dir_fd = os.open(
                corpus_dir, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
            )
        except OSError as exc:
            raise T4OrganizerFault(
                f"corpus directory for unit {unit_dir.name!r} is not an openable directory"
            ) from exc
        try:
            docs: dict[str, TrustedDoc] = {}
            for relative_path, declared_digest, entity_ids, shared in entries:
                doc_id = cls._doc_id_from_path(relative_path)
                if doc_id in docs:
                    raise T4OrganizerFault(
                        f"unit {unit_dir.name!r} declares two corpus files resolving to the same "
                        f"doc_id {doc_id!r}"
                    )
                docs[doc_id] = cls._load(
                    dir_fd,
                    relative_path,
                    doc_id,
                    declared_digest,
                    entity_ids=entity_ids,
                    shared=shared,
                )
            return cls(docs)
        finally:
            os.close(dir_fd)

    @staticmethod
    def _read_manifest(unit_dir: Path) -> Mapping[str, Any]:
        for candidate in (
            unit_dir / "manifest.json",
            unit_dir / "corpus" / "manifest.json",
        ):
            if candidate.is_file() and not candidate.is_symlink():
                try:
                    parsed: Mapping[str, Any] = json.loads(
                        candidate.read_text(encoding="utf-8")
                    )
                    return parsed
                except (OSError, json.JSONDecodeError) as exc:
                    raise T4OrganizerFault(
                        f"unit manifest {candidate.name} is unreadable or not JSON"
                    ) from exc
        raise T4OrganizerFault(
            f"unit {unit_dir.name!r} has no readable manifest.json; citation resolution requires "
            "a trusted manifest and never falls back to globbing the corpus directory"
        )

    @staticmethod
    def _declared_corpus_entries(
        manifest: Mapping[str, Any],
    ) -> list[tuple[str, str, tuple[str, ...] | None, bool]]:
        """``(path, digest, entity_ids, shared)`` per corpus entry.

        The entity label is organizer material read off the trusted manifest entry, never off the
        document: `entity_ids` (a list of roster ids; empty = about no roster entity) or
        `shared: true`. A malformed label, or both at once, is an organizer fault.
        """
        out: list[tuple[str, str, tuple[str, ...] | None, bool]] = []
        files = manifest.get("files")
        if not isinstance(files, list):
            raise T4OrganizerFault("unit manifest has no files[] array")
        for index, entry in enumerate(files):
            if not isinstance(entry, Mapping) or entry.get("role") != "corpus":
                continue
            # `corpus/manifest.json` is the corpus INDEX, not a corpus document. It is manifested
            # (every released file must be) and it carries no `doc_date`, so treating it as a
            # citable document would make every unit an organizer fault.
            if entry.get("path") == f"corpus/{CORPUS_INDEX_FILENAME}":
                continue
            path = entry.get("path")
            digest = entry.get("sha256")
            if not isinstance(path, str) or not path:
                raise T4OrganizerFault(f"manifest files[{index}] has no path")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise T4OrganizerFault(
                    f"manifest entry {path!r} has no lowercase sha256; an unverifiable corpus "
                    "document cannot be a trusted citation target"
                )
            out.append((path, "sha256:" + digest, *cls_labels(entry, path)))
        return out

    @property
    def labelled(self) -> bool:
        """Does the manifest carry entity labels at all? A unit without any is an organizer fault
        for the entity check -- an unlabelled corpus must refuse, never silently admit."""
        return any(
            d.shared or d.entity_ids is not None
            for d in self._docs.values()
            if d.row_ranges
            is None  # the task table labels itself; it says nothing of the corpus
        )

    @property
    def unlabelled_doc_ids(self) -> tuple[str, ...]:
        """Documents that carry neither `entity_ids` nor `shared`.

        In a labelled manifest these are an organizer gap, not a participant's wrong document:
        a citation to one must raise an organizer fault, never `CITATION_WRONG_ENTITY`:
        refusing the participant for a label the organizers forgot would blame them for it.
        """
        return tuple(
            sorted(
                d.doc_id
                for d in self._docs.values()
                if not d.shared and d.entity_ids is None
            )
        )

    @staticmethod
    def _doc_id_from_path(relative_path: str) -> str:
        normalized = unicodedata.normalize("NFC", relative_path)
        if normalized != relative_path:
            raise T4OrganizerFault(
                f"manifest path {relative_path!r} is not NFC-normalized"
            )
        parts = normalized.split("/")
        if len(parts) != 2 or parts[0] != "corpus" or not parts[1].endswith(".json"):
            raise T4OrganizerFault(
                f"manifest corpus path {relative_path!r} must be exactly 'corpus/<doc_id>.json'"
            )
        doc_id = parts[1][: -len(".json")]
        if not DOC_ID_RE.match(doc_id):
            raise T4OrganizerFault(
                f"corpus file {relative_path!r} yields an invalid doc_id"
            )
        return doc_id

    @classmethod
    def _load(
        cls,
        dir_fd: int,
        relative_path: str,
        doc_id: str,
        declared: str,
        *,
        entity_ids: tuple[str, ...] | None = None,
        shared: bool = False,
    ) -> TrustedDoc:
        name = relative_path.split("/", 1)[1]
        try:
            fd = os.open(
                name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=dir_fd
            )
        except OSError as exc:
            raise T4OrganizerFault(
                f"corpus document {relative_path!r} is missing or is a link; a declared corpus "
                "file must be a regular file opened without following links"
            ) from exc
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise T4OrganizerFault(
                    f"corpus document {relative_path!r} is not a regular file"
                )
            digest, payload = _sha256_file(fd)
        finally:
            os.close(fd)
        if digest != declared:
            raise T4OrganizerFault(
                f"corpus document {relative_path!r} does not match its manifest digest; the "
                "frozen corpus has been modified since the manifest was built"
            )
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise T4OrganizerFault(
                f"corpus document {relative_path!r} is not valid JSON"
            ) from exc
        if not isinstance(document, Mapping):
            raise T4OrganizerFault(
                f"corpus document {relative_path!r} is not a JSON object"
            )
        raw_date = document.get("doc_date")
        if raw_date is None:
            raise T4OrganizerFault(
                f"corpus document {relative_path!r} carries no doc_date; the embargo gate has no "
                "trusted date for it and an undated corpus document cannot be cited safely"
            )
        doc_date = parse_iso_date(
            raw_date, field=f"{relative_path}.doc_date", fault="organizer"
        )
        return TrustedDoc(
            doc_id=doc_id,
            relative_path=relative_path,
            digest=digest,
            doc_date=doc_date,
            document=document,
            entity_ids=entity_ids,
            shared=shared,
        )

    def with_task_table(self, task: Mapping[str, Any], cutoff: _dt.date) -> CorpusIndex:
        """This index plus the task table as the citable document `TASK_DOC_ID`, dated at the
        cutoff (organizer material the participant holds before it; never post-cutoff)."""
        if TASK_DOC_ID in self._docs:
            raise T4OrganizerFault(
                f"the corpus declares a document named {TASK_DOC_ID!r}, the reserved doc_id of "
                "the task table"
            )
        text, ranges = task_table_text(task)
        doc = TrustedDoc(
            doc_id=TASK_DOC_ID,
            relative_path="task.json",
            digest="sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
            doc_date=cutoff,
            document={
                "doc_id": TASK_DOC_ID,
                "doc_date": cutoff.isoformat(),
                "text": text,
            },
            entity_ids=tuple(ranges),
            shared=False,
            row_ranges=ranges,
        )
        return CorpusIndex({**self._docs, TASK_DOC_ID: doc})

    # -- lookup ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._docs)

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._docs))

    def resolve(self, doc_id: Any) -> TrustedDoc:
        """Dictionary lookup. An unknown or ill-formed id is a participant failure, never a path."""
        if not isinstance(doc_id, str) or not DOC_ID_RE.match(doc_id):
            raise T4ParticipantFailure(
                T4Reason.CITATION_UNRESOLVED,
                "doc_id is not a single well-formed corpus identifier",
                violation_count=1,
            )
        if unicodedata.normalize("NFC", doc_id) != doc_id:
            raise T4ParticipantFailure(
                T4Reason.CITATION_UNRESOLVED,
                "doc_id is not NFC-normalized",
                violation_count=1,
            )
        doc = self._docs.get(doc_id)
        if doc is None:
            raise T4ParticipantFailure(
                T4Reason.CITATION_UNRESOLVED,
                "doc_id is not a declared document of this unit's corpus",
                violation_count=1,
            )
        return doc

    def lookup(self) -> Callable[[str], Mapping[str, Any]]:
        """The ``doc_id -> doc`` callable the shared faithfulness primitives expect.

        Raises ``KeyError`` for an unknown id, which the shared helpers already treat as "this
        citation supports nothing". Track 4 refuses unresolved citations in its own gate *before*
        this is reached, so the tolerant behaviour downstream is unreachable rather than relied on.
        """
        docs = self._docs

        def _lookup(doc_id: str) -> Mapping[str, Any]:
            return docs[doc_id].document

        return _lookup

    # -- embargo -----------------------------------------------------------
    def embargo_report(
        self, citations: Iterable[Mapping[str, Any]], cutoff: _dt.date
    ) -> EmbargoReport:
        """Resolve every citation against the trusted index and date-compare against `cutoff`.

        Counts only: naming which documents a submission cited would put a per-unit private
        diagnostic into a participant-visible artifact.
        """
        checked = malformed = unresolved = undated = post_cutoff = 0
        for cite in citations:
            checked += 1
            if not isinstance(cite, Mapping):
                malformed += 1
                continue
            doc_id = cite.get("doc_id")
            try:
                doc = self.resolve(doc_id)
            except T4ParticipantFailure as failure:
                if failure.reason is T4Reason.CITATION_UNRESOLVED:
                    unresolved += 1
                else:  # pragma: no cover - resolve only raises the one reason
                    malformed += 1
                continue
            if (
                doc.doc_date is None
            ):  # pragma: no cover - construction refuses undated docs
                undated += 1
                continue
            if doc.doc_date > cutoff:
                post_cutoff += 1
        return EmbargoReport(
            checked=checked,
            malformed=malformed,
            unresolved=unresolved,
            undated=undated,
            post_cutoff=post_cutoff,
        )
