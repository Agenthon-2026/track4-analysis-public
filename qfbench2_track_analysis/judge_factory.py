"""The production judge is mandatory, pinned, and fails closed. Smoke is a different function.

The measured defect: the official factory never constructed a judge at all. ``ctx.get("judge")``
was ``None`` under the real driver, the whole faithfulness block was skipped, and ``_score`` read
``ctx.get("_faithfulness", 1.0)`` — so **a missing judge defaulted faithfulness to perfect** and
every submission cleared the admissibility gate the track exists to enforce. A separate
implementation degraded to a token-overlap "lexical" judge on an environment variable, on a card
that happened not to list an ensemble, and on *any exception* while constructing the real one, each
with nothing but a log line.

This module makes the three states distinguishable and makes only one of them rankable:

* :func:`build_production_judge` — constructs the pinned ensemble or raises
  :class:`~qfbench2_track_analysis.codes.T4OrganizerFault`. Missing weights, an unloadable runtime,
  an unpinned revision, an unverified cache, or *no configured artifact at all* are organizer
  faults with **no participant score**, never a per-participant zero.
* :func:`build_smoke_judge` — a separately named factory that always stamps
  ``judge_mode="smoke"`` and ``rankable=False``. It cannot be reached by mistake: nothing in the
  production path falls back to it, and no environment variable selects it.
* Everything else — an exception.

**The model identity is an escalated legal decision (D6), so it is configuration, not code.** What
is frozen here is the *interface* and the *provenance fields*: model ids, a revision per model, a
tokenizer digest and a local cache tree digest, in exactly the shape the hub's C4
``JudgeRecord`` parses. The default is fail-closed: with no judge artifact configured, the
production factory refuses. That converts "legal has not answered yet" from a silent perfect score
into a recorded, blocking organizer fault.

Nothing here logs a prompt. Provenance is ids and digests; the premise and hypothesis strings that
travel to the judge are never written to any sink from this module.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qfbench2_common.contracts import JudgeRecord, digest_json

from .codes import T4OrganizerFault

__all__ = [
    "ENV_JUDGE_CACHE_DIR",
    "ENV_JUDGE_SPEC",
    "JUDGE_MODES",
    "JudgeProvenance",
    "JudgeSpec",
    "build_production_judge",
    "build_smoke_judge",
    "compute_cache_tree_digest",
    "load_judge_spec",
]

#: Path to the organizer-authored judge artifact specification. There is NO default value: the
#: model choice is decision D6 (legal + model owner) and an unset variable must refuse, not guess.
ENV_JUDGE_SPEC = "QFBENCH2_T4_JUDGE_SPEC"

#: Root of the pre-staged, read-only model cache whose tree digest is recorded in provenance.
ENV_JUDGE_CACHE_DIR = "QFBENCH2_T4_MODEL_CACHE"

JUDGE_MODES = ("production", "smoke")

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class JudgeSpec:
    """The organizer's pinned judge artifact. Every field is required; none has a default."""

    model_ids: tuple[str, ...]
    model_revisions: Mapping[str, str]
    tokenizer_digest: str
    cache_tree_digest: str
    cache_dir: str

    @classmethod
    def from_mapping(cls, raw: Any, *, source: str) -> JudgeSpec:
        if not isinstance(raw, Mapping):
            raise T4OrganizerFault(f"{source}: the judge spec must be a JSON object")
        allowed = {
            "model_ids",
            "model_revisions",
            "tokenizer_digest",
            "cache_tree_digest",
            "cache_dir",
        }
        extra = sorted(set(raw) - allowed)
        if extra:
            raise T4OrganizerFault(f"{source}: unknown judge spec keys {extra}")
        model_ids = raw.get("model_ids")
        if (
            not isinstance(model_ids, list)
            or not model_ids
            or not all(isinstance(m, str) and m for m in model_ids)
        ):
            raise T4OrganizerFault(
                f"{source}: model_ids must be a non-empty array of strings"
            )
        if len(set(model_ids)) != len(model_ids):
            raise T4OrganizerFault(f"{source}: model_ids repeats a model")
        revisions = raw.get("model_revisions")
        if not isinstance(revisions, Mapping):
            raise T4OrganizerFault(f"{source}: model_revisions must be an object")
        resolved: dict[str, str] = {}
        for model_id in model_ids:
            revision = revisions.get(model_id)
            if not isinstance(revision, str) or not _REVISION_RE.match(revision):
                raise T4OrganizerFault(
                    f"{source}: model_revisions[{model_id!r}] must be a 40-hex commit sha. A model "
                    "id without a revision names a moving target, and the weights are the "
                    "ranking-critical artifact."
                )
            resolved[model_id] = revision
        for field in ("tokenizer_digest", "cache_tree_digest"):
            value = raw.get(field)
            if not isinstance(value, str) or not _DIGEST_RE.match(value):
                raise T4OrganizerFault(
                    f"{source}: {field} must be 'sha256:<64 lowercase hex>'"
                )
        cache_dir = raw.get("cache_dir")
        if not isinstance(cache_dir, str) or not cache_dir:
            raise T4OrganizerFault(f"{source}: cache_dir must be a non-empty path")
        return cls(
            model_ids=tuple(model_ids),
            model_revisions=resolved,
            tokenizer_digest=raw["tokenizer_digest"],
            cache_tree_digest=raw["cache_tree_digest"],
            cache_dir=cache_dir,
        )


@dataclass(frozen=True, slots=True)
class JudgeProvenance:
    """Exactly the C4 judge sub-record, plus the rankability it implies."""

    judge_mode: str
    model_ids: tuple[str, ...]
    model_revisions: Mapping[str, str]
    tokenizer_digest: str
    local_cache_tree_digest: str

    def __post_init__(self) -> None:
        if self.judge_mode not in JUDGE_MODES:
            raise T4OrganizerFault(f"judge_mode must be one of {list(JUDGE_MODES)}")

    @property
    def rankable(self) -> bool:
        """`smoke` is never rankable. This is the only place the question is answered."""
        return self.judge_mode == "production"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "judge_mode": self.judge_mode,
            "model_ids": list(self.model_ids),
            "model_revisions": dict(self.model_revisions),
            "tokenizer_digest": self.tokenizer_digest,
            "local_cache_tree_digest": self.local_cache_tree_digest,
        }

    def to_judge_record(self) -> JudgeRecord:
        """Parse through the hub's C4 type, so a shape drift here fails here and not downstream."""
        return JudgeRecord.from_mapping(self.to_mapping())


#: The smoke judge's provenance uses a digest of a fixed, well-known string rather than a real one.
#: It is a valid digest so the C4 record parses, and it is obviously not a model artifact so an
#: operator reading provenance cannot mistake a smoke run for a production run.
_SMOKE_SENTINEL_DIGEST = digest_json(
    {"qfbench2_track_analysis": "smoke-judge", "weights": None}
)


def compute_cache_tree_digest(cache_dir: str | os.PathLike[str]) -> str:
    """Digest of the model cache: JCS over ``{relative path: sha256 of bytes}``.

    Directory-relative and content-addressed, so a swapped weight file changes the digest and a
    re-staged identical cache does not. Symlinks are refused rather than followed: a cache whose
    contents depend on where a link points is not a cache anybody can attest to.
    """
    import hashlib

    root = Path(cache_dir)
    if not root.is_dir() or root.is_symlink():
        raise T4OrganizerFault(f"model cache {cache_dir!r} is not a directory")
    entries: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise T4OrganizerFault(
                f"model cache contains a symlink at {path.relative_to(root)}; a link makes the "
                "attested tree depend on something outside it"
            )
        if not path.is_file():
            continue
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
        entries[str(path.relative_to(root)).replace(os.sep, "/")] = (
            "sha256:" + hasher.hexdigest()
        )
    if not entries:
        raise T4OrganizerFault(f"model cache {cache_dir!r} is empty")
    digest: str = digest_json(entries)
    return digest


def load_judge_spec(path: str | os.PathLike[str] | None = None) -> JudgeSpec:
    """Read the pinned judge artifact spec. Unconfigured is an organizer fault, not a default."""
    resolved = str(path) if path is not None else os.environ.get(ENV_JUDGE_SPEC, "")
    if not resolved:
        raise T4OrganizerFault(
            f"no judge artifact is configured ({ENV_JUDGE_SPEC} is unset). The production judge is "
            "mandatory and its model identity is an escalated decision (judge model licence and "
            "redistribution); until that is answered the production factory refuses, which is a "
            "recorded organizer fault rather than a silent perfect faithfulness score."
        )
    spec_path = Path(resolved)
    if not spec_path.is_file() or spec_path.is_symlink():
        raise T4OrganizerFault(f"judge spec {resolved!r} is not a regular file")
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise T4OrganizerFault(
            f"judge spec {resolved!r} is unreadable or not JSON"
        ) from exc
    return JudgeSpec.from_mapping(raw, source=resolved)


def build_production_judge(
    spec: JudgeSpec | None = None,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    judge_builder: Any = None,
    verify_cache: bool = True,
) -> tuple[Any, JudgeProvenance]:
    """Construct the pinned production ensemble, or raise an organizer fault. No third outcome.

    `judge_builder` exists so a test can inject a constructed ensemble without model weights; it
    does **not** relax anything, because the spec, the revisions and the cache digest are still
    required and still verified. There is no environment variable that reaches it.
    """
    spec = spec or load_judge_spec()
    resolved_cache = (
        str(cache_dir)
        if cache_dir is not None
        else os.environ.get(ENV_JUDGE_CACHE_DIR, spec.cache_dir)
    )
    if verify_cache:
        observed = compute_cache_tree_digest(resolved_cache)
        if observed != spec.cache_tree_digest:
            raise T4OrganizerFault(
                "the local model cache does not match the digest the judge spec pins. The weights "
                "are the ranking-critical artifact and they live outside the scoring image, so an "
                "image digest does not cover them."
            )
    builder = (
        judge_builder if judge_builder is not None else _default_production_builder
    )
    try:
        judge = builder(spec, resolved_cache)
    except T4OrganizerFault:
        raise
    except Exception as exc:
        raise T4OrganizerFault(
            f"the production NLI judge could not be constructed ({type(exc).__name__}). A judge "
            "that fails to load is an organizer fault with no participant score; it is never a "
            "silent fallback to a lexical stub."
        ) from exc
    if judge is None or not hasattr(judge, "entail"):
        raise T4OrganizerFault(
            "the production judge builder returned no usable NLIJudge"
        )
    return judge, JudgeProvenance(
        judge_mode="production",
        model_ids=spec.model_ids,
        model_revisions=dict(spec.model_revisions),
        tokenizer_digest=spec.tokenizer_digest,
        local_cache_tree_digest=spec.cache_tree_digest,
    )


def _default_production_builder(spec: JudgeSpec, cache_dir: str) -> Any:
    """Load the pinned ensemble inside the organizer-fault construction boundary."""
    from qfbench2_common.scoring.faithfulness import EnsembleNLIJudge

    from faithfulness.judge import DeBERTaNLIJudge

    members = [
        DeBERTaNLIJudge(
            model_id=model_id,
            cache_dir=cache_dir,
            revision=spec.model_revisions[model_id],
        )
        for model_id in spec.model_ids
    ]
    # Loading uses only organizer configuration, before any participant text is evaluated.
    # Check every member inside the organizer-fault construction boundary above. Standalone
    # DeBERTa judges retain their lazy-loading behavior.
    for member in members:
        member._load()
    return EnsembleNLIJudge(members)


class LexicalSmokeJudge:
    """Token-overlap proxy. Never a production faithfulness signal; the type name says so."""

    def entail(self, premise: str, hypothesis: str) -> float:
        hyp = {t for t in hypothesis.lower().split() if len(t) > 2}
        if not hyp:
            return 0.0
        prem = {t for t in premise.lower().split() if len(t) > 2}
        return len(hyp & prem) / len(hyp)


def build_smoke_judge() -> tuple[Any, JudgeProvenance]:
    """The separately named non-rankable factory. Always `judge_mode="smoke"`."""
    from qfbench2_common.scoring.faithfulness import EnsembleNLIJudge

    judge = EnsembleNLIJudge([LexicalSmokeJudge()])
    provenance = JudgeProvenance(
        judge_mode="smoke",
        model_ids=("qfbench2_track_analysis.judge_factory.LexicalSmokeJudge",),
        model_revisions={
            "qfbench2_track_analysis.judge_factory.LexicalSmokeJudge": "0" * 40,
        },
        tokenizer_digest=_SMOKE_SENTINEL_DIGEST,
        local_cache_tree_digest=_SMOKE_SENTINEL_DIGEST,
    )
    return judge, provenance


def judge_model_ids(
    spec: JudgeSpec,
) -> Sequence[str]:  # pragma: no cover - trivial accessor
    return spec.model_ids
