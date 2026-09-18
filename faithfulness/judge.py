"""
Track 4 Faithfulness Judge — Explainability
=====================================================================
Wires DeBERTa-v3-based NLI models to the QFBench2 EnsembleNLIJudge protocol
for citation-faithfulness scoring.

In Track 4, agents predict a target (label or numeric value) per row in a table of
entities, grounded in a frozen evidence corpus. Each claim in the submission must be
entailed by the cited corpus passage. This judge checks that entailment relationship.

Models (Laurer et al., 2024):
  - cross-encoder/nli-deberta-v3-large  (HuggingFace)
  - MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli  (HuggingFace)

These models are run from locally cached weights only: the scoring environment
has no route to the HuggingFace hub (the eval network is restricted — egress is
limited to the audited model-API proxy and the organizer-hosted model endpoint;
hub domains are not on the allowlist).
Cache path: /model-cache/ (pre-staged in the evaluation Docker image).

NLI direction convention
------------------------
The *premise* is the cited passage from the corpus; the *hypothesis* is the
sentence supplied by the caller. The Track 4 prediction checker constructs that
sentence from the submitted prediction and trusted task schema, not claim prose.

Each member returns the two-way entailment-versus-contradiction softmax quantity
from the single-candidate zero-shot call. Neutral is omitted from that
normalization, so a high value does not establish a low neutral probability.
The ensemble averages these member values. This is the retained scoring
quantity, not a three-way entailment probability or a calibration guarantee.

References
----------
Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. (2024).
"Less Annotating, More Classifying: Addressing the Data Scarcity Issue of
Supervised Machine Learning with Deep Transfer Learning and BERT-NLI."
Political Analysis, 32(1), 84–100. https://doi.org/10.1017/pan.2022.34

HuggingFace model cards:
  https://huggingface.co/cross-encoder/nli-deberta-v3-large
  https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
"""

from __future__ import annotations

import json as _json
import logging
import os
import pathlib
import time as _time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

logger = logging.getLogger(__name__)

__all__ = [
    "TAU_CITATION",
    "FAITHFULNESS_THRESHOLD",
    "NLI_MODEL_IDS",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_ENTAILMENT_LABEL",
    "ENV_JUDGE_BACKEND",
    "ENV_JUDGE_URL",
    "ENV_JUDGE_TOKEN",
    "NLIJudge",
    "EnsembleNLIJudge",
    "DeBERTaNLIJudge",
    "ServedNLIJudge",
    "build_judge",
    "build_ensemble_judge",
    "score_claim",
    "AnswerCheck",
    "PredictionCheck",
    "build_unit_context",
    "check_answer",
]

#: Per-citation NLI entailment threshold (``card.toml [scoring].params.tau_citation``).
#: A claim counts as supported when its entailment score exceeds this value.
TAU_CITATION: float = 0.5

#: Admission threshold θ_f (``card.toml [scoring].params.faithfulness_threshold``).
#: The fraction of supported claims must be at least this for eligibility.
FAITHFULNESS_THRESHOLD: float = 0.80

# ---------------------------------------------------------------------------
# Guarded import of transformers / qfbench2_common
# ---------------------------------------------------------------------------
try:
    from transformers import pipeline

    # `transformers` imports fine WITHOUT torch, and then dies inside its own internals the
    # moment a pipeline is built -- observed in CI as a bare
    # `NameError: name 'torch' is not defined` raised from `hasattr(torch, dtype)`, which
    # escaped this module's "judge unavailable" path entirely and crashed the CLI after it had
    # already printed the roster banner. Importable is not the same as usable, so the backend
    # is only "available" when the tensor library it delegates to is present too.
    import torch as _torch  # noqa: F401  - probed for availability, never called here

    _TRANSFORMERS_AVAILABLE = True
except ImportError as exc:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "NLI judge will not function (%s). Install with: pip install transformers torch",
        exc,
    )

if TYPE_CHECKING:
    # The judge always builds a task="zero-shot-classification" pipeline, so annotate the
    # concrete subclass rather than the base `Pipeline`. This is load-bearing for the call
    # in `entail`: the zero-shot subclass names its first parameter `sequences`, while the
    # base class names it `inputs`, so annotating the base made mypy check the call against
    # the wrong signature and report a missing `inputs` argument for a call that is correct
    # at runtime. Naming the subclass means mypy validates against the signature actually
    # invoked -- and would catch a real upstream rename instead of being silenced.
    from transformers.pipelines.zero_shot_classification import (
        ZeroShotClassificationPipeline,
    )
else:  # at runtime the annotation is a string (PEP 563) and is never evaluated
    ZeroShotClassificationPipeline = Any

try:
    from qfbench2_common.scoring.faithfulness import NLIJudge, EnsembleNLIJudge

    _COMMON_AVAILABLE = True
except ImportError:
    _COMMON_AVAILABLE = False

    # Protocol stub for type-checking when qfbench2_common is absent.
    class NLIJudge(Protocol):  # type: ignore[no-redef]
        """Minimal NLI judge protocol compatible with qfbench2_common."""

        def entail(self, premise: str, hypothesis: str) -> float:
            """Return the judge's support score for the premise/hypothesis pair."""
            ...

    class EnsembleNLIJudge:  # type: ignore[no-redef]
        """Stub EnsembleNLIJudge matching the qfbench2_common interface."""

        def __init__(self, judges: list[NLIJudge]) -> None:
            self._judges = judges

        def entail(self, premise: str, hypothesis: str) -> float:
            """Average the support scores across all member judges."""
            if not self._judges:
                return 0.0
            return float(
                sum(j.entail(premise, hypothesis) for j in self._judges)
                / len(self._judges)
            )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: HuggingFace model IDs used for the NLI ensemble.
#:
#: Both models implement the Laurer et al. (2024) multi-dataset NLI training
#: regime, training on MNLI, FEVER-NLI, ANLI (R1–R3), Ling-NLI, and WANLI
#: with DeBERTa-v3-large as the backbone encoder. The ensemble averages their
#: two-way entailment scores; averaging does not establish calibration on
#: financial text domains.
NLI_MODEL_IDS: list[str] = [
    "cross-encoder/nli-deberta-v3-large",
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
]

#: Default directory where pre-staged model weights are expected inside the
#: evaluation Docker image.  Set ``TRANSFORMERS_CACHE`` or pass ``cache_dir``
#: explicitly if your environment differs.
DEFAULT_CACHE_DIR: str = "/model-cache"

#: The label string that the HuggingFace zero-shot-classification pipeline
#: returns for the entailment class.  Both DeBERTa models use this label.
DEFAULT_ENTAILMENT_LABEL: str = "entailment"


# ---------------------------------------------------------------------------
# DeBERTa NLI judge
# ---------------------------------------------------------------------------


@dataclass
class DeBERTaNLIJudge:
    """Single-model NLI judge backed by a DeBERTa-v3-large HuggingFace pipeline.

    Implements the :class:`NLIJudge` protocol (``score(premise, hypothesis) ->
    float``) so it can be used standalone or composed into an
    :class:`EnsembleNLIJudge`.

    The underlying HuggingFace pipeline is loaded **lazily** on the first call
    to :meth:`score`.  This avoids paying the GPU/CPU warmup cost for judge
    objects that are never actually used (e.g. in unit tests that mock the
    score method).

    NLI direction
    -------------
    The cited passage is the *premise* (``sequences``) and the hypothesis is the sole
    *candidate label*, with ``hypothesis_template="{}"`` so the pipeline's
    hypothesis is passed verbatim. ``multi_label=True`` requests the two-way
    entailment-versus-contradiction normalization; neutral is excluded. With a
    single candidate, ``multi_label=False`` takes the same normalization branch.
    Passing
    ``["entailment", "neutral", "contradiction"]`` as the candidate labels would
    classify the premise against those three literal words and ignore the hypothesis.

    Parameters
    ----------
    model_id : str
        HuggingFace model repository ID.  Must be one of the DeBERTa-v3 NLI
        models (see :data:`NLI_MODEL_IDS`).
    cache_dir : str
        Local filesystem path where cached model weights are stored.  Defaults
        to :data:`DEFAULT_CACHE_DIR` (``/model-cache``).
    device : int
        Torch device index.  ``-1`` selects CPU (default); ``0`` selects the
        first CUDA GPU.  The default is ``-1`` because it is the portable
        choice, not because no accelerator exists: this docstring used to cite
        ``gpu=false`` in ``card.toml`` as the reason, and both card files in
        this repo in fact declare ``gpu = true``.
    revision : str | None
        Model repository revision, shared by model and tokenizer loading. The
        production factory supplies the exact revision from its verified judge
        spec. Optional for standalone callers.

    Examples
    --------
    >>> judge = DeBERTaNLIJudge("cross-encoder/nli-deberta-v3-large")
    >>> score = judge.score(
    ...     premise="Apple's Q1 FY2024 Services revenue was $23.1B.",
    ...     hypothesis="Apple Services grew in Q1 FY2024.",
    ... )
    >>> assert 0.0 <= score <= 1.0
    """

    model_id: str
    cache_dir: str = DEFAULT_CACHE_DIR
    device: int = -1  # -1 = CPU
    revision: str | None = None
    _pipeline: ZeroShotClassificationPipeline | None = field(
        default=None, init=False, repr=False
    )

    def _load(self) -> ZeroShotClassificationPipeline:
        """Lazily initialise the zero-shot-classification pipeline, and return it.

        Returning the pipeline (rather than only assigning it) lets callers use the
        narrowed non-optional value directly, so the call in :meth:`entail` type-checks
        against the real signature instead of needing a blanket ``type: ignore`` that
        also hid a genuine signature mismatch.

        Called automatically on first use of :meth:`score`.  Loads the model
        weights from *cache_dir* (falling back to the HuggingFace hub if the
        cache is empty and hub access is available — note that the evaluation
        environment has no route to the hub: its restricted network only
        allows audited-proxy egress to model APIs, not hub downloads).

        Raises
        ------
        ImportError
            If the ``transformers`` package is not installed.
        RuntimeError
            If the model weights cannot be found in *cache_dir* and the
            HuggingFace hub is unreachable (as in the evaluation environment).
        """
        # Already-loaded first. A judge that HOLDS a pipeline does not need the import to
        # succeed a second time, and checking availability ahead of the short-circuit made four
        # hermetic call-convention tests -- which inject a stand-in pipeline and never touch a
        # model -- depend on `transformers` being installed. Nothing is weakened: a judge with no
        # pipeline still cannot be loaded without the package, which is the line below.
        if self._pipeline is not None:
            return self._pipeline
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The 'transformers' package is required to run DeBERTaNLIJudge. "
                "Install it with: pip install transformers torch"
            )

        logger.info(
            "DeBERTaNLIJudge: loading model '%s' from cache_dir='%s' device=%d",
            self.model_id,
            self.cache_dir,
            self.device,
        )

        self._pipeline = pipeline(
            task="zero-shot-classification",
            model=self.model_id,
            revision=self.revision,
            model_kwargs={"cache_dir": self.cache_dir},
            device=self.device,
        )

        logger.info("DeBERTaNLIJudge: model '%s' loaded successfully", self.model_id)
        return self._pipeline

    def entail(self, premise: str, hypothesis: str) -> float:
        """Return the retained two-way entailment-versus-contradiction quantity.

        The cited corpus passage is the *premise*. The caller supplies the
        *hypothesis*; the Track 4 prediction checker derives it from the submitted
        prediction and trusted task schema, rather than the participant's claim
        text. For entailment and contradiction logits e and c, this call returns
        exp(e) / (exp(e) + exp(c)). The neutral logit is excluded, so this is not
        the three-way entailment probability.

        The per-citation threshold and aggregate admission threshold are applied
        by the caller, using the unit's ``card.toml [scoring.params]``. This method
        does not establish that those thresholds are calibrated.

        Parameters
        ----------
        premise : str
            The full text of the cited corpus passage.  Should be the verbatim
            span resolved from the answer's citation offsets.
        hypothesis : str
            The hypothesis sentence to score against the passage.

        Returns
        -------
        float
            The two-way entailment score in [0.0, 1.0].

        Raises
        ------
        ImportError
            If ``transformers`` is not installed (raised on first call via
            :meth:`_load`).
        RuntimeError
            If the pipeline fails to produce a result (e.g. both strings are
            empty or the model weights are missing).
        """
        pipe = self._load()

        if not premise.strip() or not hypothesis.strip():
            logger.warning(
                "DeBERTaNLIJudge.score: received empty premise or hypothesis; "
                "returning 0.0"
            )
            return 0.0

        # The hypothesis is the candidate label: hypothesis_template "{}" passes it verbatim.
        # Passing ["entailment", "neutral", "contradiction"] as the labels instead classifies
        # the premise against those three literal words and never uses this hypothesis.
        # Keep the retained two-way normalization explicit. The pinned Transformers
        # single-candidate path also uses it when multi_label=False; that flag does not
        # turn this into a three-way probability or a constant 1.0.
        result = pipe(
            sequences=premise,
            candidate_labels=[hypothesis],
            hypothesis_template="{}",
            multi_label=True,
        )

        # result is a dict: {"labels": [...], "scores": [...], "sequence": ...}
        entailment_score = float(result["scores"][0])

        logger.debug(
            "DeBERTaNLIJudge[%s]: entailment=%.4f (hypothesis=%.80r)",
            self.model_id,
            entailment_score,
            hypothesis,
        )

        return entailment_score

    def __repr__(self) -> str:
        """Return a concise representation showing the model ID and device."""
        return (
            f"DeBERTaNLIJudge(model_id={self.model_id!r}, "
            f"cache_dir={self.cache_dir!r}, device={self.device!r})"
        )


# ---------------------------------------------------------------------------
# Served NLI judge (organizer-side serving backend)
# ---------------------------------------------------------------------------


#: Environment variables selecting the judge backend. ``local`` (the default)
#: preserves the historical behavior exactly; ``served`` routes ``entail()``
#: calls to an organizer-internal inference endpoint. This is a SERVING switch
#: only: the judge models, the ensemble mean, ``tau_citation`` and
#: ``faithfulness_threshold`` are identical on both backends.
ENV_JUDGE_BACKEND: str = "T4_JUDGE_BACKEND"
ENV_JUDGE_URL: str = "T4_JUDGE_URL"
ENV_JUDGE_TOKEN: str = "T4_JUDGE_TOKEN"


@dataclass
class ServedNLIJudge:
    """Single-model NLI judge that delegates inference to a served endpoint.

    Mirrors :class:`DeBERTaNLIJudge` one-to-one — same ``entail(premise,
    hypothesis) -> float`` protocol, one instance per ensemble member model —
    so an :class:`EnsembleNLIJudge` composed of ``ServedNLIJudge`` instances
    averages client-side exactly as the local ensemble does. The server only
    ever returns a single model's two-way entailment score; no scoring logic
    lives behind the endpoint.

    Wire protocol (JSON over HTTP)::

        POST {url}/entail
        {"model_id": "...", "premise": "...", "hypothesis": "..."}
        -> {"entailment": 0.9731}

    The endpoint is organizer-internal. It is never exposed to participants
    and its address is injected via environment variables on the scoring host.

    Parameters
    ----------
    model_id : str
        HuggingFace model repository ID of the ensemble member this judge
        represents (the server loads the same pinned weights).
    url : str
        Base URL of the serving endpoint (no trailing slash).
    token : str | None
        Optional bearer token for the endpoint.
    timeout_s : float
        Per-request timeout in seconds.
    max_retries : int
        Attempts per request before raising (with exponential backoff).
    """

    model_id: str
    url: str
    token: str | None = None
    timeout_s: float = 30.0
    max_retries: int = 3

    def entail(self, premise: str, hypothesis: str) -> float:
        """Return the served model's two-way entailment score.

        Empty premise or hypothesis short-circuits to 0.0 CLIENT-side, exactly
        as :meth:`DeBERTaNLIJudge.entail` does — the two backends must agree on
        degenerate inputs without relying on server behavior.
        """
        if not premise.strip() or not hypothesis.strip():
            logger.warning(
                "ServedNLIJudge.entail: empty premise or hypothesis; returning 0.0"
            )
            return 0.0

        payload = _json.dumps(
            {
                "model_id": self.model_id,
                "premise": premise,
                "hypothesis": hypothesis,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            request = urllib.request.Request(
                f"{self.url}/entail", data=payload, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_s
                ) as response:
                    body = _json.loads(response.read().decode("utf-8"))
                score = float(body["entailment"])
                if not 0.0 <= score <= 1.0:
                    raise ValueError(f"entailment {score} outside [0, 1]")
                return score
            except (
                urllib.error.URLError,
                KeyError,
                ValueError,
                _json.JSONDecodeError,
            ) as exc:
                last_error = exc
                logger.warning(
                    "ServedNLIJudge[%s]: attempt %d/%d failed: %s",
                    self.model_id,
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                if attempt + 1 < self.max_retries:
                    _time.sleep(min(2**attempt, 8))
        raise RuntimeError(
            f"ServedNLIJudge[{self.model_id}]: all {self.max_retries} attempts "
            f"to {self.url}/entail failed"
        ) from last_error


def build_judge(
    model_ids: list[str] | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    device: int = -1,
) -> EnsembleNLIJudge:
    """Build the ensemble judge for the backend selected by the environment.

    Reads ``T4_JUDGE_BACKEND``: ``"local"`` (default, or unset) returns exactly
    what :func:`build_ensemble_judge` returns today; ``"served"`` returns an
    ensemble of :class:`ServedNLIJudge` members pointed at ``T4_JUDGE_URL``
    (required) with optional ``T4_JUDGE_TOKEN``. Any other value raises.

    Scoring code should call this instead of :func:`build_ensemble_judge` to
    become backend-agnostic; existing callers of ``build_ensemble_judge`` are
    unaffected.
    """
    backend = os.environ.get(ENV_JUDGE_BACKEND, "local").strip().lower()
    if backend == "local":
        return build_ensemble_judge(
            model_ids=model_ids, cache_dir=cache_dir, device=device
        )
    if backend == "served":
        url = os.environ.get(ENV_JUDGE_URL, "").rstrip("/")
        if not url:
            raise RuntimeError(
                f"{ENV_JUDGE_BACKEND}=served requires {ENV_JUDGE_URL} to be set"
            )
        effective_model_ids = model_ids if model_ids is not None else NLI_MODEL_IDS
        token = os.environ.get(ENV_JUDGE_TOKEN) or None
        judges = [
            ServedNLIJudge(model_id=mid, url=url, token=token)
            for mid in effective_model_ids
        ]
        logger.info(
            "build_judge: served backend at %s with %d model(s): %r",
            url,
            len(judges),
            effective_model_ids,
        )
        return EnsembleNLIJudge(judges=judges)
    raise RuntimeError(
        f"{ENV_JUDGE_BACKEND}={backend!r} is not a valid backend "
        "(expected 'local' or 'served')"
    )


# ---------------------------------------------------------------------------
# Ensemble factory
# ---------------------------------------------------------------------------


def build_ensemble_judge(
    model_ids: list[str] | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    device: int = -1,
) -> EnsembleNLIJudge:
    """Instantiate an :class:`EnsembleNLIJudge` from one or more DeBERTa models.

    Creates one :class:`DeBERTaNLIJudge` per entry in *model_ids*, wraps them
    in :class:`EnsembleNLIJudge`, and returns the ensemble.  This is the
    recommended way to obtain a judge object for use with
    :func:`~scoring.scoring.build_verifier` or for pre-submission checks via
    :func:`score_claim`.

    The underlying models are loaded lazily on first call to
    :meth:`~EnsembleNLIJudge.score`; constructing the ensemble is cheap.

    Parameters
    ----------
    model_ids : list[str] | None
        HuggingFace model IDs to include in the ensemble.  Defaults to
        :data:`NLI_MODEL_IDS` (both DeBERTa-v3-large models).
    cache_dir : str
        Local path to cached model weights.  Passed to each
        :class:`DeBERTaNLIJudge`.  Defaults to :data:`DEFAULT_CACHE_DIR`.
    device : int
        Torch device index (``-1`` = CPU, ``0`` = first CUDA GPU).

    Returns
    -------
    EnsembleNLIJudge
        An ensemble judge whose :meth:`score` method returns the mean
        two-way entailment score across all member models.

    Raises
    ------
    ImportError
        If ``qfbench2-common`` is not installed.  The message includes the
        install command.

    Examples
    --------
    >>> judge = build_ensemble_judge()
    >>> score = judge.score(
    ...     premise="Revenue increased 12% year-over-year.",
    ...     hypothesis="Revenue grew year-over-year.",
    ... )
    >>> assert 0.0 <= score <= 1.0
    """
    if not _COMMON_AVAILABLE:
        raise ImportError(
            "qfbench2-common is required to build EnsembleNLIJudge. "
            "Install it with: "
            'pip install "qfbench2-common @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.4.3#subdirectory=common"'
        )

    effective_model_ids: list[str] = (
        model_ids if model_ids is not None else NLI_MODEL_IDS
    )

    judges: list[DeBERTaNLIJudge] = [
        DeBERTaNLIJudge(model_id=mid, cache_dir=cache_dir, device=device)
        for mid in effective_model_ids
    ]

    logger.info(
        "build_ensemble_judge: created ensemble with %d model(s): %r",
        len(judges),
        effective_model_ids,
    )

    return EnsembleNLIJudge(judges=judges)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def score_claim(
    claim_text: str,
    cited_span_text: str,
    judge: NLIJudge | None = None,
) -> float:
    """Score a single claim against its cited passage.

    Convenience wrapper around :meth:`~EnsembleNLIJudge.score` that follows
    the Track 4 NLI direction convention:

    * The **premise** is *cited_span_text* — the verbatim passage from the
      corpus document that the participant cited.
    * The **hypothesis** is *claim_text* — the participant's claim that
      purports to be grounded in that passage.

    With the DeBERTa ensemble, a high score means the models favor entailment
    over contradiction. The neutral logit is excluded from each member's
    normalization, so a high value alone does not establish that the claim
    follows from the passage.

    Parameters
    ----------
    claim_text : str
        The participant's claim, taken verbatim from
        ``answer["claims"][i]["text"]``.
    cited_span_text : str
        The resolved text of the cited corpus span (resolved from
        ``citation["doc_id"]`` + ``citation["span"]`` in the answer JSON).
    judge : NLIJudge | None
        A pre-built judge (e.g. :class:`DeBERTaNLIJudge` or
        :class:`EnsembleNLIJudge`).  If ``None``, :func:`build_ensemble_judge`
        is called with default arguments to construct a fresh ensemble.
        Passing an already-constructed judge is preferred in batch contexts
        to avoid repeated model initialisation.

    Returns
    -------
    float
        The judge's support score in [0.0, 1.0]. The Track 4 prediction checker
        applies the unit's thresholds to hypotheses built from roster predictions.
        Its denominator is the entity roster, not the number of claim strings
        scored by this utility.

    Raises
    ------
    ImportError
        If *judge* is ``None`` and ``qfbench2-common`` is not installed
        (raised inside :func:`build_ensemble_judge`).

    Examples
    --------
    >>> s = score_claim(
    ...     claim_text="Apple Services revenue was $23.1B in Q1 FY2024.",
    ...     cited_span_text="Services net sales were $23,117 million for the first quarter.",
    ... )
    >>> assert 0.0 <= s <= 1.0
    """
    effective_judge: NLIJudge
    if judge is None:
        effective_judge = build_ensemble_judge()
    else:
        effective_judge = judge

    return float(
        effective_judge.entail(
            premise=cited_span_text,
            hypothesis=claim_text,
        )
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _collect_claims(answer_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Every claim in an answer, from wherever the schema puts them.

    The schema nests claims per entity — ``entity_predictions[i].claims`` — and this script
    used to read a top-level ``answer["claims"]`` that the schema does not define. Measured
    2026-08-24 against this repository's own shipped baseline: the answer carried one claim
    under ``entity_predictions[0].claims`` and the script reported "Scoring 0 claim(s)", then
    printed "GATE: FAIL (judge not available)". A participant following the README's step 3
    got a passing-looking run that had examined nothing.

    The top-level form is still accepted so a hand-written legacy file keeps working.
    """
    claims: list[dict[str, Any]] = []
    for entity in answer_data.get("entity_predictions") or []:
        if isinstance(entity, dict):
            claims.extend(
                c for c in (entity.get("claims") or []) if isinstance(c, dict)
            )
    claims.extend(c for c in (answer_data.get("claims") or []) if isinstance(c, dict))
    return claims


def build_unit_context(unit_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """The trusted half of a scoring context for *unit_dir*, built by the scorer's own hydrator.

    Delegates to :func:`qfbench2_track_analysis.scoring.hydrate`, so the corpus index, the entity
    roster, the hypothesis spec, the cutoff date and the scoring parameters this check uses are the
    SAME objects the official gate uses, read the same way from the same files.

    In particular ``ctx["_corpus"]`` is a :class:`~qfbench2_track_analysis.corpus.CorpusIndex`
    built by ``CorpusIndex.from_unit`` — a dictionary keyed by the doc_ids the unit's manifest
    declares, whose files are opened ``O_NOFOLLOW`` and digest-checked. The local check used to
    build a path instead, ``corpus_dir / f"{doc_id}.json"``, which made the participant the author
    of a filesystem path inside the unit tree. Measured against a synthetic unit: an undeclared
    file dropped into ``corpus/`` resolved and scored 1.0 where the real path raises
    ``T4ParticipantFailure(t4.citation_unresolved)``, and ``doc_id="../../oracle"`` read a JSON
    file sitting one level ABOVE the unit directory. A dictionary lookup cannot traverse.
    """
    from qfbench2_track_analysis.scoring import hydrate

    ctx: dict[str, Any] = {"unit_dir": pathlib.Path(unit_dir)}
    hydrate(ctx)
    return ctx


class _MemoizingJudge:
    """Answers each ``(premise, hypothesis)`` pair exactly once.

    :func:`check_answer` consults the judge twice over the same pairs: once per roster entity for
    the rows it prints, and once over the whole list inside the shared ``citation_faithfulness``,
    which owns the aggregate. Memoizing makes the second pass free, so the check can reuse the
    shared primitive rather than re-implement it, and the printed rows cannot drift from the number
    the gate computes. NLI inference is the expensive part of this script; each pair is one call.
    """

    def __init__(self, inner: NLIJudge) -> None:
        self._inner = inner
        self.cache: dict[tuple[str, str], float] = {}

    def entail(self, premise: str, hypothesis: str) -> float:
        key = (premise, hypothesis)
        cached = self.cache.get(key)
        if cached is None:
            cached = float(self._inner.entail(premise, hypothesis))
            self.cache[key] = cached
        return cached

    def best_for(self, hypothesis: str) -> float:
        """The highest score any premise scored against *hypothesis*, over the calls made."""
        return max(
            (v for (_, h), v in self.cache.items() if h == hypothesis), default=0.0
        )


@dataclass(frozen=True)
class PredictionCheck:
    """One roster entity's result: the question the judge was asked, and its answer."""

    entity_id: str
    #: The canonical hypothesis — derived from the SUBMITTED prediction, never authored by it.
    hypothesis: str
    score: float
    supported: bool


@dataclass(frozen=True)
class AnswerCheck:
    """Aggregate result of :func:`check_answer` over one ``answer.json``."""

    predictions: tuple[PredictionCheck, ...]
    faithfulness: float
    gate_pass: bool
    #: The denominator. Always the trusted roster size, never the number of claims submitted.
    roster_count: int


def check_answer(
    answer_data: Mapping[str, Any],
    ctx: Mapping[str, Any],
    judge: NLIJudge,
) -> AnswerCheck:
    """Run the unit's evidence semantics over *answer_data* locally, as the real gate runs them.

    *ctx* is a hydrated context from :func:`build_unit_context`. The steps mirror
    ``qfbench2_track_analysis.scoring._g3_domain_semantics`` and are assembled from the same public
    parts, so there is no second implementation of the gate to drift:

    1. ``align_predictions`` against the **trusted roster**, which fixes the denominator. The
       previous local check counted ``_collect_claims(answer)``: padding one entity with seven
       copies of a claim moved the local number from 1 to 7 while the gate's stayed at 1.
    2. ``CorpusIndex.embargo_report`` over every citation. Unresolved, undated and post-cutoff are
       all violations (fail-closed since 2026-08-22), and each raises here as it raises there.
    3. ``prediction_claims(aligned, ctx["_hypothesis_spec"])``: the hypothesis handed to the judge
       is the **canonical prediction sentence** built from the submitted label / point forecast /
       rank / interval plus the trusted task schema. It is never ``claim["text"]``. That rule was
       deleted from the scorer in #28, and it is the reason this rewrite exists: measured on a
       synthetic unit, an answer with ``label="miss"``, ``point_forecast=-99.0`` and interval
       ``[-100, -98]`` whose prose accurately described a real corpus passage scored
       **faithfulness 1.0 / gate_pass True** locally while the real gate scored **0.0**. A
       pre-check that is wrong in that direction is worse than no pre-check.
    4. ``qfbench2_common.scoring.faithfulness.citation_faithfulness`` for the aggregate — the same
       shared primitive the gate calls, given the same claim list and the same trusted lookup.

    Raises :class:`~qfbench2_track_analysis.codes.T4ParticipantFailure` for anything the gate
    refuses before faithfulness is reached, so a local run fails on the same submissions and for
    the same stated reason.
    """
    from qfbench2_common.scoring import faithfulness as F
    from qfbench2_track_analysis.alignment import align_predictions
    from qfbench2_track_analysis.codes import T4ParticipantFailure, T4Reason
    from qfbench2_track_analysis.hypothesis import prediction_claims

    params = ctx["_params"]
    roster = ctx["_roster"]
    corpus = ctx["_corpus"]

    aligned = align_predictions(
        answer_data,
        roster,
        target_type=params.target_type,
        interval_level=params.interval_level,
    )

    report = corpus.embargo_report(aligned.all_citations(), ctx["_cutoff"])
    if not report.clean:
        reason = (
            T4Reason.CITATION_POST_CUTOFF
            if report.post_cutoff
            else T4Reason.CITATION_UNRESOLVED
            if (report.unresolved or report.malformed)
            else T4Reason.CITATION_UNDATED
        )
        raise T4ParticipantFailure(
            reason,
            "one or more citations are unresolved, undated or post-cutoff",
            violation_count=report.violation_count,
            observed_count=report.checked,
        )

    claims = prediction_claims(aligned, ctx["_hypothesis_spec"])
    lookup = corpus.lookup()
    memo = _MemoizingJudge(judge)
    tau = float(params.tau_citation)

    # Per-entity rows, each computed by the SHARED primitive over a one-element list, so the row
    # and the aggregate cannot be computed by different code. The judge calls are memoized, so
    # asking twice costs one inference.
    checks: list[PredictionCheck] = []
    for entity_id, claim in zip(aligned.entity_ids, claims, strict=True):
        hypothesis = str(claim["text"])
        supported = F.citation_faithfulness([claim], lookup, memo, tau=tau) >= 1.0
        checks.append(
            PredictionCheck(
                entity_id=entity_id,
                hypothesis=hypothesis,
                score=memo.best_for(hypothesis),
                supported=supported,
            )
        )

    phi = float(F.citation_faithfulness(claims, lookup, memo, tau=tau))
    return AnswerCheck(
        predictions=tuple(checks),
        faithfulness=phi,
        gate_pass=phi >= float(params.faithfulness_threshold),
        roster_count=roster.count,
    )


if __name__ == "__main__":
    import argparse
    import json
    import sys

    if __package__ in (None, ""):
        # `python faithfulness/judge.py` puts faithfulness/ on sys.path -- NOT the repo root --
        # so this repo's own `qfbench2_track_analysis` package would not import, and the check
        # needs it for the roster, the corpus index and the hypothesis spec. The repo is not
        # pip-installable from a checkout (see the README's harness step, which sets PYTHONPATH
        # for the same reason), so the script form has to say where it lives.
        # `python -m faithfulness.judge` needs none of this.
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the Track 4 faithfulness judge. "
            "Pass --answer to score a full answer.json, or run without arguments "
            "for a built-in synthetic test."
        )
    )
    parser.add_argument(
        "--answer",
        type=str,
        default=None,
        help="Path to an answer.json file to score.",
    )
    parser.add_argument(
        "--unit",
        type=str,
        default=None,
        help=(
            "Path to the unit directory (required with --answer): the one carrying task.json, "
            "card.toml, manifest.json and corpus/. The trusted roster, the cutoff, the scoring "
            "parameters and the manifest-declared corpus all come from it, because all four are "
            "inputs to the gate this check is previewing."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f"Model cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    args = parser.parse_args()

    if args.answer is not None and args.unit is None:
        # Fail closed, and BEFORE the judge is built, so a machine with no model weights still
        # gets this error rather than the judge-unavailable one. Without the unit there is no
        # roster to be the denominator, no manifest to resolve a doc_id against and no task
        # schema to build the hypothesis from; the only premise left would be the claim's own
        # text, which entails trivially and PASSes answers the real gate fails.
        parser.error(
            "--unit is required with --answer (the unit directory holding task.json, card.toml, "
            "manifest.json and corpus/)"
        )

    # ------------------------------------------------------------------
    # Attempt to build the judge; fall back gracefully when dependencies
    # are absent (e.g. in CI without model weights).
    # ------------------------------------------------------------------
    judge_obj: EnsembleNLIJudge | None = None

    if not _COMMON_AVAILABLE:
        print(
            "WARNING: qfbench2-common is not installed. "
            "Cannot build EnsembleNLIJudge. "
            "Install with: "
            'pip install "qfbench2-common @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.4.3#subdirectory=common"',
            file=sys.stderr,
        )
    elif not _TRANSFORMERS_AVAILABLE:
        print(
            "WARNING: transformers is not installed. "
            "Cannot run real NLI inference. "
            "Install with: pip install transformers torch",
            file=sys.stderr,
        )
    else:
        try:
            judge_obj = build_ensemble_judge(cache_dir=args.cache_dir)
        except Exception as exc:
            print(f"ERROR building judge: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Define test premise/hypothesis pairs.
    # ------------------------------------------------------------------
    test_pairs: list[tuple[str, str]] = [
        (
            # (hypothesis, premise) — faithful example
            "Apple's Services segment revenue grew year-over-year in Q1 FY2024.",
            "Services net sales were $23,117 million for the first quarter of fiscal 2024, "
            "compared to $20,766 million for the same period in fiscal 2023.",
        ),
        (
            # (hypothesis, premise) — unfaithful example (claim not in text)
            "Apple's iPhone revenue declined 20% in Q1 FY2024.",
            "Services net sales were $23,117 million for the first quarter of fiscal 2024.",
        ),
    ]

    # ------------------------------------------------------------------
    # Run scoring (or emit dummy 0.0 if unavailable).
    # ------------------------------------------------------------------
    if args.answer is not None:
        # Score a real answer file
        with open(args.answer) as fh:
            answer_data: dict[str, Any] = json.load(fh)

        from qfbench2_track_analysis.codes import T4OrganizerFault, T4ParticipantFailure

        try:
            unit_ctx = build_unit_context(args.unit)
        except T4OrganizerFault as fault:
            # The unit tree, not the submission, is unusable. Say which, and do not print a
            # gate verdict for a check that never ran.
            print(
                f"ERROR: {args.unit!r} is not a usable Track 4 unit: {fault}",
                file=sys.stderr,
            )
            raise SystemExit(2) from fault

        roster_count = unit_ctx["_roster"].count
        parsed_claims = len(_collect_claims(answer_data))
        print(
            f"\nScoring {roster_count} prediction(s) from {args.answer!r} "
            f"against unit {unit_ctx['unit_dir'].name!r}"
        )
        # The denominator is the roster, not the prose. Printed side by side because the two
        # numbers differing is normal and used to be invisible: the old check counted claims,
        # so padding an entity with copies of one claim raised the local figure and moved the
        # gate's not at all.
        print(
            f"({parsed_claims} participant claim(s) parsed; the faithfulness denominator is "
            f"the trusted roster: {roster_count})"
        )
        print("-" * 60)

        if judge_obj is None:
            # The unit hydrated and the banner above is real, but faithfulness is the whole
            # point of this check and it cannot be computed without the judge. A check that
            # prints FAIL and exits 0 is not a check: callers -- CI, a Makefile, an agent
            # following the README -- see only the status.
            print("\nJudge unavailable; no prediction was scored.")
            print("GATE: FAIL (judge not available — install dependencies)")
            raise SystemExit(1)

        try:
            result = check_answer(answer_data, unit_ctx, judge_obj)
        except T4ParticipantFailure as failure:
            # Everything the gate refuses before faithfulness is reached: an entity missing from
            # the roster, a non-finite number, an unresolved or post-cutoff citation. The real
            # gate stops here too, and the unit takes W.
            print(f"\nGATE: FAIL ({failure.code.value}) — {failure}")
            raise SystemExit(1) from failure

        for check in result.predictions:
            gate_status = "PASS" if check.supported else "FAIL"
            print(f"  {check.entity_id}: {check.score:.4f} [{gate_status}]")
            # Printed in full, not truncated: the interval clause is the half a participant
            # most often has not realised the judge is being asked about.
            print(f"    hypothesis: {check.hypothesis!r}")
        print(
            f"\nFaithfulness (fraction of supported predictions): "
            f"{result.faithfulness:.4f}"
        )
        print(f"GATE: {'PASS' if result.gate_pass else 'FAIL'}")
        if not result.gate_pass:
            raise SystemExit(1)

    else:
        # Built-in synthetic test
        print("\nRunning built-in synthetic faithfulness tests")
        print("-" * 60)

        for i, (hypothesis, premise) in enumerate(test_pairs):
            if judge_obj is None:
                s = 0.0
                note = " (dummy — judge unavailable)"
            else:
                s = score_claim(
                    claim_text=hypothesis,
                    cited_span_text=premise,
                    judge=judge_obj,
                )
                note = ""

            label = "faithful" if i == 0 else "unfaithful"
            print(f"  Test {i} [{label:>10}]: score={s:.4f}{note}")
            print(f"    hypothesis: {hypothesis[:80]!r}")
            print(f"    premise:    {premise[:80]!r}")

        print("\nSmoke test complete.")
