"""OpenAI-compatible chat client for ``$MODEL_ENDPOINT`` (stdlib only).

The eval sandbox's only egress is the organizer-hosted House route. The harness
injects ``MODEL_ENDPOINT`` as the route **origin** (``scheme://host:port``) and the
OpenAI-compatible API is served under ``/v1``, so the request goes to
``$MODEL_ENDPOINT/v1/chat/completions`` with ``Authorization: Bearer $MODEL_TOKEN``
(see the hub's ``docs/HOUSE-MODEL.md``, "Calling the House route"). Locally, any
server speaking that protocol works (ollama, llama.cpp, vLLM) whether its URL is
given with or without the ``/v1`` suffix, and tests inject :class:`MockModelClient`
— same interface, canned replies, no network.

Determinism: temperature 0 and a fixed ``seed`` are sent on every request.
"""
from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol

from .config import Config


def chat_completions_url(model_endpoint: str) -> str:
    """The chat-completions URL for an endpoint given with or without ``/v1``.

    The harness injects the route origin (no path); local servers are often
    configured as ``http://host:port/v1``. Both resolve to ``.../v1/chat/completions``.
    """
    base = model_endpoint.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


class ModelCallError(RuntimeError):
    """The model call failed after every retry (network, timeout, server error, bad envelope)."""


class ModelConfigError(RuntimeError):
    """The endpoint refused the request as misconfigured; retrying cannot help."""


class ModelBudgetExhausted(ModelCallError):
    """The House refused the request because the unit's request allowance is used up.

    Past the per-unit request cap (and for an expired or unknown grant) the House answers
    HTTP 403 with the JSON body ``{"error": {"message": "request admission refused",
    "type": "invalid_request_error", "code": "grant_denied"}}``; only that code with that
    message is read as this refusal (see :func:`is_budget_refusal`).
    It is not a configuration error: before or after a reply, the remaining entities get
    fallback rows so the unit still finishes with an answer, and ``cli.run`` records in
    ``notes`` whether any reply (or any usable reply) came before the refusal.
    """


#: HTTP statuses that mean a wrong endpoint or credential rather than a transient failure.
_CONFIG_HTTP_STATUSES = frozenset({401, 403, 404})

#: The ``error.code`` and ``error.message`` of the House's 403 once the request allowance is
#: used up. The same code with another message (for example "request model does not match
#: credential") is a configuration error, not a used-up allowance.
_BUDGET_DENIED_CODE = "grant_denied"
_BUDGET_DENIED_MESSAGE = "request admission refused"


def is_budget_refusal(body: str) -> bool:
    """True only for a JSON body whose ``error.code`` is ``grant_denied`` and whose
    ``error.message`` is ``request admission refused``; any other or unparsable body is not."""
    try:
        parsed = json.loads(body)
    except ValueError:
        return False
    error = parsed.get("error") if isinstance(parsed, dict) else None
    return (
        isinstance(error, dict)
        and error.get("code") == _BUDGET_DENIED_CODE
        and error.get("message") == _BUDGET_DENIED_MESSAGE
    )


def _error_body(exc: urllib.error.HTTPError) -> str:
    """The text of an HTTP error response, or "" when it cannot be read."""
    try:
        return exc.read().decode("utf-8", errors="replace")
    except (OSError, http.client.HTTPException, ValueError, AttributeError):
        return ""


class ModelClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the assistant message text for one chat exchange."""
        ...


@dataclass
class HTTPModelClient:
    config: Config
    #: Set once the House has answered 403 grant_denied; later calls send nothing.
    budget_exhausted: bool = False

    def complete(self, system: str, user: str) -> str:
        if self.budget_exhausted:
            raise ModelBudgetExhausted(
                "the unit's model request allowance is used up (403 grant_denied); "
                "no request sent"
            )
        if not self.config.model_endpoint:
            raise RuntimeError(
                "MODEL_ENDPOINT is not set. In the eval sandbox it is injected "
                "by the harness; locally, point it at an OpenAI-compatible "
                "server or use --mock."
            )
        payload = {
            "model": self.config.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "seed": self.config.seed,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.model_token:
            headers["Authorization"] = f"Bearer {self.config.model_token}"
        request = urllib.request.Request(
            chat_completions_url(self.config.model_endpoint),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_s
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
            except (
                OSError,  # URLError, HTTPError, TimeoutError, ConnectionError
                http.client.HTTPException,
                ValueError,  # JSONDecodeError, UnicodeDecodeError
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                if (
                    isinstance(exc, urllib.error.HTTPError)
                    and exc.code == 403
                    and is_budget_refusal(_error_body(exc))
                ):
                    self.budget_exhausted = True
                    raise ModelBudgetExhausted(
                        "the unit's model request allowance is used up (403 grant_denied)"
                    ) from exc
                if (
                    isinstance(exc, urllib.error.HTTPError)
                    and exc.code in _CONFIG_HTTP_STATUSES
                ) or (
                    isinstance(exc, urllib.error.URLError)
                    and str(exc.reason).startswith("unknown url type")
                ):
                    raise ModelConfigError(
                        f"the model endpoint rejected the request ({exc}); check "
                        "MODEL_ENDPOINT and MODEL_TOKEN"
                    ) from exc
                last_error = exc
                time.sleep(min(2**attempt, 8))
        raise ModelCallError(
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
