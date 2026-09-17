"""Runtime configuration for the strong RAG baseline (env-driven, no files).

Everything is resolved from environment variables so the same agent runs
unchanged in the eval sandbox (where the organizer injects ``$MODEL_ENDPOINT``)
and locally (where you can point it at a mock or an ollama/llama.cpp server
speaking the same OpenAI-compatible protocol).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    model_endpoint: str  # House route origin as injected (http://host:port); a local http://host/v1 also works
    model_id: str
    model_token: str | None
    seed: int  # forwarded to the model AND used for any local tie-breaking
    top_k: int  # retrieved chunks per entity
    timeout_s: float  # per model call
    max_retries: int
    temperature: float  # fixed at 0 for determinism; env override for experiments

    @staticmethod
    def from_env() -> "Config":
        return Config(
            model_endpoint=os.environ.get("MODEL_ENDPOINT", "").rstrip("/"),
            # MODEL_NAME is what the harness injects (SUBMISSION_CLI.md container
            # contract); MODEL_ID is a local-dev fallback only.
            model_id=os.environ.get("MODEL_NAME") or os.environ.get("MODEL_ID", ""),
            model_token=os.environ.get("MODEL_TOKEN") or None,
            seed=int(os.environ.get("T4_SEED", "20260731")),
            top_k=int(os.environ.get("T4_TOP_K", "10")),
            timeout_s=float(os.environ.get("T4_MODEL_TIMEOUT_S", "60")),
            max_retries=int(os.environ.get("T4_MODEL_RETRIES", "3")),
            temperature=float(os.environ.get("T4_TEMPERATURE", "0")),
        )
