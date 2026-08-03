"""Provider-agnostic LLM entry point with rate limiting and failover.

Every LLM call in the system goes through :meth:`LLMRouter.complete`. Callers
ask for a *tier* rather than a model:

``fast``    routing, query rewriting, summarising, SQL planning - high volume,
            latency-sensitive, low reasoning demand.
``strong``  answer synthesis and Text-to-SQL generation - lower volume, where
            quality actually shows.

The router maps the tier onto each provider's configured model and walks the
provider chain until one succeeds. What happens on failure depends on *why*:

* rate limited or transient  -> back off and retry the same provider, then move on
* bad credentials            -> drop the provider for the process lifetime
* malformed request          -> stop immediately; another provider will fail too

Every response records which provider served it and whether a fallback
occurred, so the UI trace panel can show it rather than asking anyone to take
it on trust.
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
    AuthError,
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
from app.llm.ratelimit import (
    AsyncTokenBucket,
    CircuitBreaker,
    TokenBudget,
    estimate_tokens,
)
from app.logging_conf import get_logger

log = get_logger(__name__)

# Attempts per provider before failing over.
MAX_ATTEMPTS_PER_PROVIDER = 2

# How long to wait on a throttled provider when another one is available.
# Beyond this, failing over is faster than queueing.
FAILOVER_WAIT_SECONDS = 1.5

# Output tokens to reserve against the budget, regardless of how high the
# caller sets ``max_tokens``.
#
# ``max_tokens`` is a *safety ceiling*, not a prediction - synthesis sets 2500
# purely so Gemini's thinking tokens cannot truncate the visible answer.
# Charging that ceiling to the budget made a full-directory synthesis estimate
# 5,045 tokens against Groq's 6,000/min, so it was declared throttled and
# handed to Gemini, which was 503-ing under load: 52 seconds for an answer Groq
# produces in under one. The turn that actually ran reported 2,125 tokens
# total. Reserving a realistic figure and letting ``reconcile`` correct it from
# the provider's own usage numbers is both more accurate and self-healing.
OUTPUT_RESERVE_TOKENS = 700


class LLMRouter:
    def __init__(self) -> None:
        available: dict[str, Provider] = {
            "gemini": GeminiProvider(),
            "groq": GroqProvider(),
        }
        rpm = {"gemini": settings.gemini_rpm, "groq": settings.groq_rpm}

        # Token allowances are published per *model*, not per provider, and the
        # two tiers differ by 2x on Groq (6k for llama-3.1-8b-instant, 12k for
        # llama-3.3-70b-versatile). Pacing both against the smaller figure
        # throttles synthesis at half its real allowance.
        self._tpm: dict[tuple[str, str], int] = {
            ("gemini", "fast"): settings.gemini_tpm,
            ("gemini", "strong"): settings.gemini_tpm_strong,
            ("groq", "fast"): settings.groq_tpm,
            ("groq", "strong"): settings.groq_tpm_strong,
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
            self.buckets[name] = AsyncTokenBucket(rpm.get(name, 15), name=name)
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

    # ------------------------------------------------------------------
    def _budget(self, provider: str, model: str, tier: str | None = None) -> TokenBudget:
        """Token budget for one provider/model pair, created on first use.

        ``tier`` is only needed to look up the configured allowance; once the
        budget exists it is found by model name alone, so a caller that knows
        just the model it is about to hit does not have to know its tier.
        """
        key = f"{provider}:{model}"
        existing = self.budgets.get(key)
        if existing is not None:
            return existing

        # Both tiers of every configured provider are created up front, so an
        # unknown model here means a model nobody declared. Give it the smaller
        # of the two allowances: pacing too slowly costs latency, pacing too
        # fast costs 429s.
        budget = TokenBudget(self._tpm.get((provider, tier or "fast"), 30_000), name=key)
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

    # ------------------------------------------------------------------
    async def complete(
        self,
        messages: list[Msg] | list[dict],
        *,
        tier: Tier = "fast",
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 45.0,
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

            for attempt in range(1, MAX_ATTEMPTS_PER_PROVIDER + 1):
                total_attempts += 1
                try:
                    # Waiting only makes sense when there is nothing better to
                    # do. If an *untried* provider remains, a throttled one
                    # should be abandoned quickly rather than queued behind:
                    # Groq's 6k TPM fills in seconds while Gemini's 250k TPM
                    # sits idle. But on the last remaining provider, waiting
                    # out the throttle is far better than failing the turn -
                    # checking "any other provider exists" instead of "any
                    # untried provider remains" makes every provider bail early
                    # and collapses the agent onto its no-LLM fallback.
                    untried = [n for n in eligible if n != name]
                    wait_budget = FAILOVER_WAIT_SECONDS if untried else timeout

                    waited = await self.buckets[name].acquire(timeout=wait_budget)

                    # TPM is the binding constraint on free tiers, so the
                    # budget is checked in addition to the request bucket.
                    estimated = estimate_tokens(
                        [m.content for m in msgs],
                        max_output=min(max_tokens, OUTPUT_RESERVE_TOKENS),
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
                    # Throttled, not broken. Do NOT record a breaker failure -
                    # a provider that is merely at its quota must not be
                    # circuit-opened for a minute; it will be usable again in
                    # seconds and is still the preferred choice.
                    errors[name] = f"throttled: {exc}"
                    log.info("llm.throttled_failing_over", provider=name,
                             untried=untried, detail=str(exc)[:120])
                    if name in eligible:
                        eligible.remove(name)
                    break

                except AuthError as exc:
                    # No amount of retrying fixes a bad key. Take it out of
                    # rotation so every later turn skips it instantly.
                    self.disabled[name] = str(exc)
                    errors[name] = f"auth: {exc}"
                    log.error("llm.provider_disabled", provider=name,
                              reason=str(exc)[:200])
                    break

                except BadRequestError as exc:
                    # The prompt is at fault; other providers will reject it too.
                    log.error("llm.bad_request", provider=name, error=str(exc)[:200])
                    raise

                except ProviderError as exc:
                    errors[name] = f"{type(exc).__name__}: {exc}"
                    breaker.record_failure()
                    if not exc.retryable or attempt == MAX_ATTEMPTS_PER_PROVIDER:
                        log.warning("llm.failing_over", provider=name,
                                    error=str(exc)[:160])
                        break
                    delay = _backoff(attempt, exc)
                    log.warning("llm.retrying", provider=name, attempt=attempt,
                                delay=round(delay, 2), error=str(exc)[:120])
                    await asyncio.sleep(delay)

        raise AllProvidersFailedError(errors)

    # ------------------------------------------------------------------
    async def complete_json(
        self,
        messages: list[Msg] | list[dict],
        json_schema: dict[str, Any],
        *,
        tier: Tier = "fast",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[dict, LLMResponse]:
        """Structured output with tolerant parsing.

        Even in JSON mode models wrap output in code fences or add a sentence of
        preamble often enough that strict ``json.loads`` is a real source of
        turn failures. Salvaging the outermost object is cheaper and safer than
        spending another request on a retry.
        """
        response = await self.complete(
            messages, tier=tier, json_schema=json_schema,
            temperature=temperature, max_tokens=max_tokens,
        )
        return parse_json_loose(response.text), response

    # ------------------------------------------------------------------
    async def stream(
        self,
        messages: list[Msg] | list[dict],
        *,
        tier: Tier = "strong",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Stream deltas from the first provider that yields a token.

        Failover only applies *before* the first token: once text has reached
        the user, switching providers mid-answer would duplicate or contradict
        what they have already read.
        """
        msgs = _coerce(messages)
        errors: dict[str, str] = {}

        for provider in self.providers:
            name = provider.name
            if name in self.disabled or self.breakers[name].is_open:
                continue

            model = provider.model_for(tier)
            emitted = False
            try:
                await self.buckets[name].acquire(timeout=30.0)
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
                if isinstance(exc, AuthError):
                    self.disabled[name] = str(exc)
                if emitted:
                    log.error("llm.stream_failed_midway", provider=name,
                              error=str(exc)[:160])
                    raise
                log.warning("llm.stream_failing_over", provider=name,
                            error=str(exc)[:160])

        raise AllProvidersFailedError(errors)


# ---------------------------------------------------------------------------
def _coerce(messages: list[Msg] | list[dict]) -> list[Msg]:
    out: list[Msg] = []
    for m in messages:
        out.append(m if isinstance(m, Msg) else Msg(role=m["role"], content=m["content"]))
    return out


def _backoff(attempt: int, exc: ProviderError) -> float:
    """Exponential backoff with jitter, honouring Retry-After when present."""
    if isinstance(exc, RateLimitError) and exc.retry_after:
        return min(float(exc.retry_after), 30.0)
    return min(2.0**attempt, 20.0) * (0.5 + random.random() / 2)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json_loose(text: str) -> dict:
    """Best-effort extraction of a JSON object from a model response."""
    if not text or not text.strip():
        raise ValueError("empty response, expected JSON")

    candidates = [text.strip()]

    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

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
