"""Shared types for the LLM provider layer.

Providers are wrapped behind our own interface rather than used through a
framework abstraction, because the failover behaviour - which error is worth
retrying, which should disable a provider, how usage is reported - is the part
that has to be trustworthy, and it is provider-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Tier = Literal["fast", "strong"]
Role = Literal["system", "user", "assistant"]


@dataclass
class Msg:
    role: Role
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    tier: str
    latency_ms: int
    attempts: int = 1
    fell_back: bool = False
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""

    def to_trace(self) -> dict:
        """Compact form for the SSE trace panel."""
        return {
            "provider": self.provider,
            "model": self.model,
            "tier": self.tier,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "fell_back": self.fell_back,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
        }


# ---------------------------------------------------------------------------
# Errors
#
# The distinction that matters is not "did it fail" but "what should we do
# next": wait and retry the same provider, move to the next one, or stop
# entirely because no amount of retrying will fix a bad key.
# ---------------------------------------------------------------------------
class ProviderError(Exception):
    retryable = False
    #: When true the provider is taken out of rotation for the process lifetime.
    fatal_for_provider = False

    def __init__(self, message: str, provider: str = "", status: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status = status


class RateLimitError(ProviderError):
    """429 - back off and retry, then fail over."""

    retryable = True

    def __init__(self, message: str, provider: str = "", status: int | None = 429,
                 retry_after: float | None = None):
        super().__init__(message, provider, status)
        self.retry_after = retry_after


class TransientError(ProviderError):
    """5xx, timeout, connection reset - retry."""

    retryable = True


class AuthError(ProviderError):
    """Invalid or missing credentials. Retrying is pointless."""

    retryable = False
    fatal_for_provider = True


class BadRequestError(ProviderError):
    """Malformed request. Failing over will not help - the prompt is the problem."""

    retryable = False


class AllProvidersFailedError(RuntimeError):
    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        detail = "; ".join(f"{p}: {e}" for p, e in errors.items()) or "no providers configured"
        super().__init__(f"every LLM provider failed ({detail})")


def classify_http_status(status: int, message: str, provider: str) -> ProviderError:
    """Map an HTTP status onto the error taxonomy above."""
    if status in (401, 403):
        return AuthError(message, provider, status)
    if status == 429:
        return RateLimitError(message, provider, status)
    if status >= 500:
        return TransientError(message, provider, status)
    if status == 400 and "api key not valid" in message.lower():
        # Gemini reports an invalid key as 400 INVALID_ARGUMENT rather than 401.
        return AuthError(message, provider, status)
    if status >= 400:
        return BadRequestError(message, provider, status)
    return TransientError(message, provider, status)


class Provider:
    """Interface every provider implements."""

    name: str = ""

    def is_configured(self) -> bool:  # pragma: no cover - trivial
        raise NotImplementedError

    def model_for(self, tier: Tier) -> str:  # pragma: no cover - trivial
        raise NotImplementedError

    async def complete(
        self,
        messages: list[Msg],
        model: str,
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 45.0,
    ) -> tuple[str, Usage, str]:
        """Return (text, usage, finish_reason). Raise a ProviderError subclass."""
        raise NotImplementedError

    async def stream(
        self,
        messages: list[Msg],
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 90.0,
    ):
        """Yield text deltas."""
        raise NotImplementedError
