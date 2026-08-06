"""Single entry point for every LLM call: maps a tier ("fast"/"strong") to a
provider's model and walks the provider chain on failure, retrying, failing
over, or dropping a provider depending on the error type.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from typing import Any, AsyncIterator

from app.config import settings
from app.llm.base import (
    AllProvidersFailedError,
    BadRequestError,
    LLMResponse,
    Msg,
    Provider,
    ProviderError,
    RateLimitError,
    Tier,
)
from app.llm.providers.gemini import GeminiProvider, check_key_format
from app.llm.providers.groq import GroqProvider
from app.llm.providers.openrouter import OpenRouterProvider
from app.llm.ratelimit import (
    AsyncTokenBucket,
    CircuitBreaker,
    TokenBudget,
    estimate_tokens,
)
from app.logging_conf import get_logger

log = get_logger(__name__)

# Attempts per provider before failing over.
MAX_ATTEMPTS_PER_PROVIDER = settings.llm_max_attempts_per_provider

# How long to wait on a throttled provider when another one is available.
# Beyond this, failing over is faster than queueing.
FAILOVER_WAIT_SECONDS = settings.llm_failover_wait_seconds

# Output tokens reserved against the budget regardless of the caller's max_tokens
# (a safety ceiling, not a prediction); reconcile() corrects it from actual usage.
OUTPUT_RESERVE_TOKENS = settings.llm_output_reserve_tokens


class LLMRouter:
    def __init__(self) -> None:
        available: dict[str, Provider] = {
            "gemini": GeminiProvider(),
            "groq": GroqProvider(),
            "openrouter": OpenRouterProvider(),
        }
        rpm = {
            "gemini": settings.gemini_rpm,
            "groq": settings.groq_rpm,
            "openrouter": settings.openrouter_rpm,
        }

        # Token allowances are per model, not per provider, and differ by tier.
        self._tpm: dict[tuple[str, str], int] = {
            ("gemini", "fast"): settings.gemini_tpm,
            ("gemini", "strong"): settings.gemini_tpm_strong,
            ("groq", "fast"): settings.groq_tpm,
            ("groq", "strong"): settings.groq_tpm_strong,
            # OpenRouter meters free models by request, not token, so these are
            # set high on purpose - see the note in config.py.
            ("openrouter", "fast"): settings.openrouter_tpm,
            ("openrouter", "strong"): settings.openrouter_tpm_strong,
        }

        self.providers: list[Provider] = []
        self.buckets: dict[str, AsyncTokenBucket] = {}
        #: Keyed by ``provider:model``, created lazily by :meth:`_budget`.
        self.budgets: dict[str, TokenBudget] = {}
        self.breakers: dict[str, CircuitBreaker] = {}
        self.disabled: dict[str, str] = {}

        for name in settings.provider_order:
            provider = available.get(name)
            if provider is None:
                log.warning("llm.unknown_provider", provider=name)
                continue
            if not provider.is_configured():
                log.warning("llm.provider_not_configured", provider=name)
                continue
            self.providers.append(provider)
            self.buckets[name] = AsyncTokenBucket(
                rpm.get(name, settings.gemini_rpm), name=name
            )
            self.breakers[name] = CircuitBreaker(name=name)
            for tier in ("fast", "strong"):
                self._budget(name, provider.model_for(tier), tier=tier)

        if not self.providers:
            log.error("llm.no_providers_configured")
        else:
            log.info("llm.ready", providers=[p.name for p in self.providers])

        # Surface the most common misconfiguration up front rather than as a
        # confusing 400 on the first real request.
        warning = check_key_format(settings.gemini_api_key)
        if warning and "gemini" in [p.name for p in self.providers]:
            log.warning("llm.suspicious_gemini_key", detail=warning)

    def _budget(self, provider: str, model: str, tier: str | None = None) -> TokenBudget:
        """Token budget for one provider/model pair, created on first use."""
        key = f"{provider}:{model}"
        existing = self.budgets.get(key)
        if existing is not None:
            return existing

        # An unknown model gets the smaller of the two tier allowances.
        budget = TokenBudget(
            self._tpm.get((provider, tier or "fast"), settings.llm_default_tpm),
            name=key,
        )
        self.budgets[key] = budget
        return budget

    @property
    def active_providers(self) -> list[str]:
        return [p.name for p in self.providers if p.name not in self.disabled]

    def status(self) -> dict:
        return {
            "order": [p.name for p in self.providers],
            "disabled": self.disabled,
            "breakers": {n: b.status() for n, b in self.breakers.items()},
            "request_tokens": {
                n: round(b.tokens_available(), 2) for n, b in self.buckets.items()
            },
            "tpm_remaining": {
                n: round(b.remaining()) for n, b in self.budgets.items()
            },
        }

    async def complete(
        self,
        messages: list[Msg] | list[dict],
        *,
        tier: Tier = "fast",
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = settings.llm_default_max_tokens,
        timeout: float = settings.llm_request_timeout_seconds,
    ) -> LLMResponse:
        msgs = _coerce(messages)
        if not self.providers:
            raise AllProvidersFailedError({})

        started = time.perf_counter()
        errors: dict[str, str] = {}
        total_attempts = 0
        first_choice = self.active_providers[0] if self.active_providers else None

        # Providers still to try after the current one. Used to decide whether
        # waiting out a throttle is better than moving on.
        eligible = [
            p.name for p in self.providers
            if p.name not in self.disabled and not self.breakers[p.name].is_open
        ]

        for provider in self.providers:
            name = provider.name
            if name in self.disabled:
                continue

            breaker = self.breakers[name]
            if breaker.is_open:
                errors[name] = f"circuit open ({breaker.seconds_until_retry():.0f}s left)"
                log.info("llm.skipped_open_circuit", provider=name)
                continue

            model = provider.model_for(tier)

            for attempt in range(1, settings.llm_max_attempts_per_provider + 1):
                total_attempts += 1
                try:
                    # Fail over fast if an untried provider remains; otherwise wait it out.
                    untried = [n for n in eligible if n != name]
                    wait_budget = (
                        settings.llm_failover_wait_seconds if untried else timeout
                    )

                    waited = await self.buckets[name].acquire(timeout=wait_budget)

                    # TPM is the binding constraint on free tiers, so the
                    # budget is checked in addition to the request bucket.
                    estimated = estimate_tokens(
                        [m.content for m in msgs],
                        max_output=min(max_tokens, settings.llm_output_reserve_tokens),
                    )
                    budget = self._budget(name, model)
                    waited += await budget.acquire(estimated, timeout=wait_budget)
                    if waited > 0.5:
                        log.info("llm.rate_limit_wait", provider=name,
                                 seconds=round(waited, 1), est_tokens=estimated)

                    text, usage, finish = await provider.complete(
                        msgs, model,
                        json_schema=json_schema,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                    )
                    breaker.record_success()
                    # Replace the estimate with what was actually consumed so
                    # estimation error does not accumulate across a long run.
                    budget.reconcile(estimated, usage.total)

                    response = LLMResponse(
                        text=text,
                        provider=name,
                        model=model,
                        tier=tier,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        attempts=total_attempts,
                        fell_back=(name != first_choice),
                        usage=usage,
                        finish_reason=finish,
                    )
                    log.info("llm.ok", provider=name, model=model, tier=tier,
                             ms=response.latency_ms, attempts=total_attempts,
                             fell_back=response.fell_back,
                             tokens=usage.total)
                    return response

                except TimeoutError as exc:
                    # Throttled, not broken - don't record a breaker failure for this.
                    errors[name] = f"throttled: {exc}"
                    log.info("llm.throttled_failing_over", provider=name,
                             untried=untried, detail=str(exc)[:120])
                    if name in eligible:
                        eligible.remove(name)
                    break

                except BadRequestError as exc:
                    # The prompt is at fault; other providers will reject it too.
                    log.error("llm.bad_request", provider=name, error=str(exc)[:200])
                    raise

                except ProviderError as exc:
                    if exc.fatal_for_provider:
                        # Bad key or unusable model id: not retryable, drop the provider.
                        self.disabled[name] = str(exc)
                        errors[name] = f"{type(exc).__name__}: {exc}"
                        log.error("llm.provider_disabled", provider=name,
                                  reason=str(exc)[:200])
                        break

                    errors[name] = f"{type(exc).__name__}: {exc}"
                    breaker.record_failure()
                    if (
                        not exc.retryable
                        or attempt == settings.llm_max_attempts_per_provider
                    ):
                        log.warning("llm.failing_over", provider=name,
                                    error=str(exc)[:160])
                        break
                    delay = _backoff(attempt, exc)
                    log.warning("llm.retrying", provider=name, attempt=attempt,
                                delay=round(delay, 2), error=str(exc)[:120])
                    await asyncio.sleep(delay)

        raise AllProvidersFailedError(errors)

    async def complete_json(
        self,
        messages: list[Msg] | list[dict],
        json_schema: dict[str, Any],
        *,
        tier: Tier = "fast",
        temperature: float = 0.0,
        max_tokens: int = settings.llm_default_max_tokens,
    ) -> tuple[dict, LLMResponse]:
        """Structured output with tolerant parsing (handles fenced/preambled JSON)."""
        response = await self.complete(
            messages, tier=tier, json_schema=json_schema,
            temperature=temperature, max_tokens=max_tokens,
        )
        return parse_json_loose(response.text), response

    async def stream(
        self,
        messages: list[Msg] | list[dict],
        *,
        tier: Tier = "strong",
        temperature: float = 0.0,
        max_tokens: int = settings.llm_default_max_tokens,
    ) -> AsyncIterator[str]:
        """Stream deltas from the first provider that yields a token; no failover after the first token."""
        msgs = _coerce(messages)
        errors: dict[str, str] = {}

        for provider in self.providers:
            name = provider.name
            if name in self.disabled or self.breakers[name].is_open:
                continue

            model = provider.model_for(tier)
            emitted = False
            try:
                await self.buckets[name].acquire(
                    timeout=settings.llm_stream_timeout_seconds
                )
                async for delta in provider.stream(
                    msgs, model, temperature=temperature, max_tokens=max_tokens
                ):
                    emitted = True
                    yield delta
                self.breakers[name].record_success()
                return
            except Exception as exc:
                self.breakers[name].record_failure()
                errors[name] = str(exc)
                if isinstance(exc, ProviderError) and exc.fatal_for_provider:
                    self.disabled[name] = str(exc)
                if emitted:
                    log.error("llm.stream_failed_midway", provider=name,
                              error=str(exc)[:160])
                    raise
                log.warning("llm.stream_failing_over", provider=name,
                            error=str(exc)[:160])

        raise AllProvidersFailedError(errors)


def _coerce(messages: list[Msg] | list[dict]) -> list[Msg]:
    out: list[Msg] = []
    for m in messages:
        out.append(m if isinstance(m, Msg) else Msg(role=m["role"], content=m["content"]))
    return out


# exponential backoff with jitter, but use Retry-After if the provider sent one
def _backoff(attempt: int, exc: ProviderError) -> float:
    if isinstance(exc, RateLimitError) and exc.retry_after:
        return min(float(exc.retry_after), settings.llm_max_retry_after_seconds)
    return min(2.0**attempt, settings.llm_max_backoff_seconds) * (
        0.5 + random.random() / 2
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


# Models wrap JSON in markdown fences, or add a sentence before it, or both.
# Try the most likely shapes in order and take the first one that parses.
def parse_json_loose(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("empty response, expected JSON")

    candidates = []

    # fenced block first - if there is one it's almost always the real payload
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    candidates.append(text.strip())

    # last resort: grab everything between the outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"could not parse JSON from response: {text[:200]!r}")


_router: LLMRouter | None = None


def get_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router


def reset_router() -> None:
    """Force re-reading configuration (used by tests)."""
    global _router
    _router = None
