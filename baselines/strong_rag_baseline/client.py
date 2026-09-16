"""OpenAI-compatible house client with a deadline shared by every entity."""

from __future__ import annotations

import json
import copy
import hashlib
import time
import urllib.request
from dataclasses import dataclass, field
from http.client import HTTPException
from typing import Callable, Protocol
from urllib.parse import urlsplit

from .config import Config, HOUSE_MAX_OUTPUT_TOKENS, HOUSE_MAX_REQUESTS


class ModelBudgetExceeded(RuntimeError):
    """No further HTTP request may be made within this unit."""


class ModelClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the assistant message text for one chat exchange."""
        ...


@dataclass
class HTTPModelClient:
    config: Config
    base_url: str = ""
    deadline: float | None = None
    _attempts: list[dict] = field(default_factory=list, init=False)
    _budget_blocks: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        for value, limit in (
            (self.config.max_requests, HOUSE_MAX_REQUESTS),
            (self.config.max_tokens, HOUSE_MAX_OUTPUT_TOKENS),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= limit
            ):
                raise ValueError(
                    "model budget must stay within the configured House ceiling"
                )
        if self.deadline is None:
            self.deadline = time.monotonic() + self.config.unit_timeout_s

    def request_ledger(self) -> dict:
        return {
            "version": 1,
            "source": "http-client",
            "attempt_limit": self.config.max_requests,
            "output_token_limit": self.config.max_tokens,
            "attempts_used": len(self._attempts),
            "budget_blocks": self._budget_blocks,
            "attempts": copy.deepcopy(self._attempts),
            "scope": "client attempts including failures; does not assert organizer billing or input-token accounting",
        }

    def complete(self, system: str, user: str) -> str:
        return self._complete(system, user)

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        return self._complete(system, user, schema)

    def _complete(self, system: str, user: str, schema: dict | None = None) -> str:
        endpoint = (self.base_url or self.config.model_endpoint).rstrip("/")
        url = urlsplit(endpoint)
        official = (
            bool(self.config.model_endpoint) and endpoint == self.config.model_endpoint
        )
        loopback = url.hostname in {"127.0.0.1", "localhost", "::1"}
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or not (official or loopback)
        ):
            raise RuntimeError(
                "model URL must be MODEL_ENDPOINT or an explicit loopback URL (127.0.0.1)"
            )
        payload = {
            "model": self.config.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_tokens": self.config.max_tokens,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "entity_prediction",
                    "strict": True,
                    "schema": schema,
                },
            }
        headers = {"Content-Type": "application/json"}
        if official and self.config.model_token:
            headers["Authorization"] = f"Bearer {self.config.model_token}"
        request = urllib.request.Request(
            f"{endpoint}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            remaining = float(self.deadline) - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("unit model deadline exhausted") from last_error
            if len(self._attempts) >= self.config.max_requests:
                self._budget_blocks += 1
                raise ModelBudgetExceeded("unit request budget exhausted")
            started = time.monotonic()
            record = {
                "sequence": len(self._attempts) + 1,
                "status": "pending",
                "response_bytes": 0,
                "request_sha256": hashlib.sha256(request.data).hexdigest(),
                "prompt_tokens": None,
                "completion_tokens": None,
            }
            # Count before I/O: failed, truncated and retried requests all consume an attempt.
            self._attempts.append(record)
            try:
                with urllib.request.urlopen(
                    request, timeout=min(self.config.timeout_s, remaining)
                ) as response:
                    parts: list[bytes] = []
                    size = 0
                    while True:
                        if time.monotonic() >= float(self.deadline):
                            raise TimeoutError(
                                "unit model deadline exhausted during response"
                            )
                        part = response.read1(min(65536, 1_048_577 - size))
                        size += len(part)
                        record["response_bytes"] = size
                        if size > 1_048_576:
                            raise ValueError("model response exceeds 1 MiB")
                        if not part:
                            break
                        parts.append(part)
                    data = b"".join(parts)
                body = json.loads(data.decode("utf-8"))
                usage = body.get("usage", {}) if isinstance(body, dict) else {}
                if isinstance(usage, dict):
                    for key in ("prompt_tokens", "completion_tokens"):
                        value = usage.get(key)
                        if (
                            isinstance(value, int)
                            and not isinstance(value, bool)
                            and value >= 0
                        ):
                            record[key] = value
                if (
                    record["completion_tokens"] is not None
                    and record["completion_tokens"] > self.config.max_tokens
                ):
                    raise ValueError(
                        "model response reports output beyond requested limit"
                    )
                if body["choices"][0].get("finish_reason") in {
                    "length",
                    "content_filter",
                }:
                    raise ValueError("model response did not complete")
                content = body["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("model response has no text content")
                record["status"] = "success"
                return content
            except (
                OSError,
                HTTPException,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                RecursionError,
            ) as exc:
                record["status"] = "error"
                record["error_type"] = type(exc).__name__
                last_error = exc
                remaining = float(self.deadline) - time.monotonic()
                if (
                    attempt + 1 < self.config.max_retries
                    and remaining > 0
                    and len(self._attempts) < self.config.max_requests
                ):
                    time.sleep(min(2**attempt, remaining))
            finally:
                record["elapsed_s"] = max(0.0, time.monotonic() - started)
        raise RuntimeError(
            f"model call failed after {self.config.max_retries} attempts"
        ) from last_error


@dataclass
class MockModelClient:
    """Test double: returns canned text, or delegates to a callable."""

    reply: str | Callable[[str, str], str]

    def complete(self, system: str, user: str) -> str:
        if callable(self.reply):
            return self.reply(system, user)
        return self.reply
