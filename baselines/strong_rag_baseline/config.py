"""Harness model settings and bounded per-unit inference budgets."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from .local_server import (
    DEFAULT_ALIAS,
    DEFAULT_CTX,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_STARTUP_S,
)


# Selected House allowance in the organiser SUBMISSION_CLI contract, 2026-09-15.
# Also a conservative local/BYO ceiling; it does not claim a BYO entitlement.
HOUSE_MAX_REQUESTS = 25
HOUSE_MAX_OUTPUT_TOKENS = 4000


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _bounded_env(name: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return min(hi, max(lo, value)) if math.isfinite(value) else default


@dataclass(frozen=True)
class Config:
    model_endpoint: str
    model_token: str | None
    model_id: str
    seed: int  # forwarded to the model AND used for any local tie-breaking
    top_k: int  # retrieved chunks per entity
    timeout_s: float  # per model call
    max_retries: int
    temperature: float  # fixed at 0 for determinism; env override for experiments
    max_tokens: int
    unit_timeout_s: float
    local_llama: bool  # developer-machine opt-in; official analyze leaves this false
    local_host: str
    local_port: int
    local_ctx: int
    local_startup_s: float
    max_requests: int = HOUSE_MAX_REQUESTS

    @property
    def local_base_url(self) -> str:
        return f"http://{self.local_host}:{self.local_port}/v1"

    @staticmethod
    def from_env() -> "Config":
        host = os.environ.get("T4_LOCAL_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
        if host not in {"127.0.0.1", "localhost", "::1"}:
            host = DEFAULT_HOST
        return Config(
            model_endpoint=os.environ.get("MODEL_ENDPOINT", "").strip().rstrip("/"),
            model_token=os.environ.get("MODEL_TOKEN") or None,
            model_id=os.environ.get("MODEL_NAME")
            or os.environ.get("MODEL_ID", "")
            or DEFAULT_ALIAS,
            seed=int(
                _bounded_env(
                    "QFBENCH_SEED",
                    _bounded_env("T4_SEED", 20260731, 0, 2**32 - 1),
                    0,
                    2**32 - 1,
                )
            ),
            top_k=int(_bounded_env("T4_TOP_K", 10, 1, 30)),
            timeout_s=_bounded_env("T4_MODEL_TIMEOUT_S", 30, 0.1, 60),
            max_retries=int(_bounded_env("T4_MODEL_RETRIES", 2, 1, 3)),
            temperature=_bounded_env("T4_TEMPERATURE", 0, 0, 2),
            max_tokens=int(
                _bounded_env("T4_MAX_TOKENS", 1024, 64, HOUSE_MAX_OUTPUT_TOKENS)
            ),
            unit_timeout_s=_bounded_env("T4_UNIT_TIMEOUT_S", 480, 1, 480),
            local_llama=_env_on("T4_LOCAL_LLAMA"),
            local_host=host,
            local_port=int(_bounded_env("T4_LOCAL_PORT", DEFAULT_PORT, 1, 65535)),
            local_ctx=int(_bounded_env("T4_LOCAL_CTX", DEFAULT_CTX, 512, 32768)),
            local_startup_s=_bounded_env(
                "T4_LOCAL_STARTUP_S", DEFAULT_STARTUP_S, 1, 90
            ),
        )
