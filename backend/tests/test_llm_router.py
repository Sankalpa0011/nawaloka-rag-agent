"""Failover, rate limiting and circuit-breaker behaviour, with no network calls.

The live behaviour is easy to eyeball once; what matters is that the *policy*
stays correct as the code changes - that an auth failure disables a provider,
that a malformed prompt does not stampede every provider in turn, and that a
stream which has already emitted text never silently switches models mid-answer.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.llm.base import (
    AllProvidersFailedError,
    AuthError,
    BadRequestError,
    Msg,
    Provider,
    RateLimitError,
    TransientError,
    Usage,
    classify_http_status,
)
from app.llm.ratelimit import (
    AsyncTokenBucket,
    CircuitBreaker,
    TokenBudget,
    estimate_tokens,
)
from app.llm.router import LLMRouter, parse_json_loose


class FakeProvider(Provider):
    """Scripted provider. ``script`` is a list of outcomes, consumed in order."""

    def __init__(self, name: str, script: list, stream_chunks: list[str] | None = None):
        self.name = name
        self.script = list(script)
        self.calls = 0
        self.stream_chunks = stream_chunks or ["hello ", "world"]

    def is_configured(self) -> bool:
        return True

    def model_for(self, tier):
        return f"{self.name}-{tier}"

    async def complete(self, messages, model, **kw):
        self.calls += 1
        outcome = self.script.pop(0) if self.script else "ok"
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, Usage(input_tokens=1, output_tokens=1), "stop"

    async def stream(self, messages, model, **kw):
        self.calls += 1
        outcome = self.script.pop(0) if self.script else None
        if isinstance(outcome, Exception):
            raise outcome
        for c in self.stream_chunks:
            yield c
        if outcome == "fail_after_first":
            raise TransientError("died midway", self.name)


def build_router(
    *providers: FakeProvider, rpm: int = 600, tpm: int = 1_000_000
) -> LLMRouter:
    router = LLMRouter.__new__(LLMRouter)  # bypass config-driven __init__
    router.providers = list(providers)
    router.buckets = {p.name: AsyncTokenBucket(rpm, name=p.name) for p in providers}
    router.breakers = {p.name: CircuitBreaker(name=p.name) for p in providers}
    router.disabled = {}
    router._tpm = {}
    router.budgets = {}
    for p in providers:
        for tier in ("fast", "strong"):
            router._tpm[(p.name, tier)] = tpm
            router._budget(p.name, p.model_for(tier), tier=tier)
    return router


def budget_key(provider: str, tier: str = "fast") -> str:
    """Budgets are per provider *and* model, because the limits are."""
    return f"{provider}:{provider}-{tier}"


MSGS = [Msg("user", "hi")]


# ---------------------------------------------------------------------------
# Failover policy
# ---------------------------------------------------------------------------
class TestFailover:
    async def test_primary_success_does_not_fall_back(self):
        a, b = FakeProvider("a", ["A"]), FakeProvider("b", ["B"])
        r = build_router(a, b)
        resp = await r.complete(MSGS)
        assert resp.text == "A" and resp.provider == "a"
        assert resp.fell_back is False
        assert b.calls == 0, "secondary should not be touched when primary works"

    async def test_transient_error_retries_then_falls_over(self):
        a = FakeProvider("a", [TransientError("boom", "a"), TransientError("boom", "a")])
        b = FakeProvider("b", ["B"])
        r = build_router(a, b)
        resp = await r.complete(MSGS)
        assert resp.provider == "b"
        assert resp.fell_back is True
        assert a.calls == 2, "should retry the primary before failing over"

    async def test_auth_error_disables_provider_permanently(self):
        a = FakeProvider("a", [AuthError("bad key", "a"), "A"])
        b = FakeProvider("b", ["B1", "B2"])
        r = build_router(a, b)

        first = await r.complete(MSGS)
        assert first.provider == "b"
        assert "a" in r.disabled
        assert a.calls == 1, "must not retry a bad key"

        second = await r.complete(MSGS)
        assert second.provider == "b"
        assert a.calls == 1, "disabled provider must be skipped entirely"

    async def test_auth_error_is_not_retried(self):
        a = FakeProvider("a", [AuthError("bad key", "a")])
        r = build_router(a)
        with pytest.raises(AllProvidersFailedError):
            await r.complete(MSGS)
        assert a.calls == 1

    async def test_bad_request_does_not_stampede_other_providers(self):
        """A malformed prompt fails everywhere; trying each provider wastes quota."""
        a = FakeProvider("a", [BadRequestError("bad prompt", "a")])
        b = FakeProvider("b", ["B"])
        r = build_router(a, b)
        with pytest.raises(BadRequestError):
            await r.complete(MSGS)
        assert b.calls == 0

    async def test_rate_limit_retries_then_falls_over(self):
        a = FakeProvider("a", [RateLimitError("429", "a", retry_after=0.01)] * 2)
        b = FakeProvider("b", ["B"])
        r = build_router(a, b)
        resp = await r.complete(MSGS)
        assert resp.provider == "b" and a.calls == 2

    async def test_all_providers_failing_reports_each_reason(self):
        a = FakeProvider("a", [TransientError("a-down", "a")] * 2)
        b = FakeProvider("b", [TransientError("b-down", "b")] * 2)
        r = build_router(a, b)
        with pytest.raises(AllProvidersFailedError) as exc:
            await r.complete(MSGS)
        assert set(exc.value.errors) == {"a", "b"}

    async def test_no_providers_configured(self):
        r = build_router()
        with pytest.raises(AllProvidersFailedError):
            await r.complete(MSGS)

    async def test_throttled_provider_fails_over_instead_of_queueing(self):
        """A provider at its token quota should be skipped, not waited on,
        when another provider has headroom."""
        a, b = FakeProvider("a", ["A"]), FakeProvider("b", ["B"])
        r = build_router(a, b)
        r.budgets[budget_key("a")] = TokenBudget(tpm=10, name="a")   # effectively exhausted
        await r.budgets[budget_key("a")].acquire(10)

        started = time.monotonic()
        resp = await r.complete(MSGS)
        assert resp.provider == "b"
        assert time.monotonic() - started < 3.0, "should not have queued"
        assert a.calls == 0

    async def test_throttling_does_not_open_the_circuit(self):
        """Being at quota is not a fault; the provider must stay eligible."""
        a, b = FakeProvider("a", ["A"]), FakeProvider("b", ["B"])
        r = build_router(a, b)
        r.budgets[budget_key("a")] = TokenBudget(tpm=10, name="a")
        await r.budgets[budget_key("a")].acquire(10)

        await r.complete(MSGS)
        assert r.breakers["a"].failures == 0
        assert not r.breakers["a"].is_open

    async def test_waits_when_it_is_the_only_provider(self):
        """With nothing to fail over to, waiting beats failing."""
        a = FakeProvider("a", ["A"])
        r = build_router(a)
        r.budgets[budget_key("a")] = TokenBudget(tpm=600, name="a")
        await r.budgets[budget_key("a")].acquire(300)

        resp = await r.complete(MSGS, max_tokens=100)
        assert resp.provider == "a"

    async def test_last_provider_waits_even_when_others_were_tried(self):
        """Regression: 'is any other provider configured' is the wrong test.

        With every provider throttled, checking for *any* alternative makes
        each one bail after the short failover wait, so the whole chain fails
        and the agent silently degrades to its no-LLM fallback. The last
        untried provider must wait out the throttle instead.
        """
        a, b = FakeProvider("a", ["A"]), FakeProvider("b", ["B"])
        r = build_router(a, b)
        # Both throttled; b has just enough headroom if we are willing to wait.
        r.budgets[budget_key("a")] = TokenBudget(tpm=60, name="a")
        r.budgets[budget_key("b")] = TokenBudget(tpm=600, name="b")
        await r.budgets[budget_key("a")].acquire(60)
        await r.budgets[budget_key("b")].acquire(300)

        resp = await r.complete(MSGS, max_tokens=100)
        assert resp.provider == "b", "last provider must wait rather than fail"
        assert resp.fell_back is True


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
class TestBudgetAccounting:
    """What the budget charges decides which provider serves the turn."""

    async def test_a_high_ceiling_is_not_charged_as_if_it_were_spent(self):
        """Regression: ``max_tokens`` is a ceiling, not a prediction.

        Synthesis sets 2500 only so Gemini's thinking tokens cannot truncate
        the visible answer. Charging that ceiling made a full-directory
        synthesis estimate 5,045 tokens against Groq's 6,000/min, so the fast
        provider was declared throttled and the turn was handed to Gemini,
        which was 503-ing under load: 52 seconds for an answer Groq returns in
        under one. ``reconcile`` corrects the estimate from real usage
        afterwards, so a modest reserve is both more accurate and self-healing.
        """
        a, b = FakeProvider("a", ["A"]), FakeProvider("b", ["B"])
        r = build_router(a, b, tpm=1500)
        await r.budgets[budget_key("a")].acquire(500)

        resp = await r.complete(MSGS, max_tokens=100_000)
        assert resp.provider == "a", "a ceiling of 100k must not evict the fast provider"

    async def test_each_tier_is_budgeted_separately(self):
        """Token allowances are published per model, not per provider.

        Groq allows 6,000 TPM on llama-3.1-8b-instant and 12,000 on
        llama-3.3-70b-versatile. A single shared counter paces synthesis at
        half its real allowance.
        """
        a, b = FakeProvider("a", ["A"]), FakeProvider("b", ["B"])
        r = build_router(a, b, tpm=100)
        await r.budgets[budget_key("a", "fast")].acquire(100)  # fast tier spent

        resp = await r.complete(MSGS, tier="strong", max_tokens=10)
        assert resp.provider == "a", "an exhausted fast tier must not throttle the strong one"

    def test_budgets_are_keyed_by_provider_and_model(self):
        a = FakeProvider("a", [])
        r = build_router(a)
        assert set(r.budgets) == {"a:a-fast", "a:a-strong"}


class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        cb = CircuitBreaker(threshold=3, cooldown=60)
        for _ in range(2):
            cb.record_failure()
        assert not cb.is_open
        cb.record_failure()
        assert cb.is_open and cb.status() == "open"

    def test_success_resets(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert not cb.is_open

    @staticmethod
    def _expire(cb: CircuitBreaker) -> None:
        """Age the breaker past its cooldown without sleeping.

        Sleeping just over the cooldown is inherently racy on a loaded machine;
        rewinding the clock makes the test deterministic. Reading ``is_open``
        afterwards is what flips the breaker to half-open, so the next
        ``record_failure`` counts as a failed trial rather than a fresh one.
        """
        cb.opened_at = time.monotonic() - cb.current_cooldown - 0.01
        assert not cb.is_open

    def test_half_opens_after_cooldown(self):
        cb = CircuitBreaker(threshold=1, cooldown=60)
        cb.record_failure()
        assert cb.is_open
        self._expire(cb)
        assert cb.status() == "half_open"

    def test_failed_trial_reopens_for_full_cooldown(self):
        cb = CircuitBreaker(threshold=1, cooldown=60)
        cb.record_failure()
        self._expire(cb)            # half-open, trial allowed
        cb.record_failure()         # trial fails
        assert cb.is_open, "a failed trial must re-open the circuit"
        assert cb.seconds_until_retry() > 50, "must wait a full cooldown again"

    def test_repeated_failed_trials_back_off_exponentially(self):
        """A provider out of daily quota must not be re-probed every minute.

        An open circuit is what takes a provider out of the router's eligible
        list, and only when no eligible provider is left does the router allow
        the survivor to wait out its own token budget. So a dead provider that
        keeps looking half-open does not merely waste a round-trip - it starves
        the healthy one of the wait it needed to succeed.
        """
        cb = CircuitBreaker(threshold=1, cooldown=60, max_cooldown=900)
        cb.record_failure()
        assert cb.current_cooldown == 60

        for expected in (120, 240, 480):
            self._expire(cb)
            cb.record_failure()
            assert cb.current_cooldown == expected

    def test_backoff_is_capped(self):
        cb = CircuitBreaker(threshold=1, cooldown=60, max_cooldown=900)
        cb.record_failure()
        for _ in range(12):
            self._expire(cb)
            cb.record_failure()
        assert cb.current_cooldown == 900

    def test_success_clears_the_backoff(self):
        """Recovery must be immediate - yesterday's outage cannot park a live key."""
        cb = CircuitBreaker(threshold=1, cooldown=60)
        cb.record_failure()
        self._expire(cb)
        cb.record_failure()
        assert cb.reopens == 1

        cb.record_success()
        assert cb.reopens == 0
        assert not cb.is_open
        assert cb.current_cooldown == 60

    async def test_open_circuit_is_skipped_without_calling_provider(self):
        a, b = FakeProvider("a", ["A"]), FakeProvider("b", ["B"])
        r = build_router(a, b)
        r.breakers["a"].threshold = 1
        r.breakers["a"].record_failure()
        resp = await r.complete(MSGS)
        assert resp.provider == "b"
        assert a.calls == 0, "open circuit must short-circuit before the call"


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------
class TestTokenBucket:
    async def test_burst_is_allowed_then_paced(self):
        bucket = AsyncTokenBucket(rpm=60, burst=2, name="t")  # 1/sec after burst
        assert await bucket.acquire() == 0.0
        assert await bucket.acquire() == 0.0
        waited = await bucket.acquire(timeout=5)
        assert waited > 0.3, "third request should be paced"

    async def test_timeout_when_wait_would_be_too_long(self):
        bucket = AsyncTokenBucket(rpm=1, burst=1, name="t")
        await bucket.acquire()
        with pytest.raises(TimeoutError):
            await bucket.acquire(timeout=0.2)

    async def test_concurrent_callers_are_serialised(self):
        bucket = AsyncTokenBucket(rpm=600, burst=1, name="t")  # 10/sec
        started = time.monotonic()
        await asyncio.gather(*(bucket.acquire(timeout=5) for _ in range(4)))
        assert time.monotonic() - started > 0.2


class TestTokenBudget:
    """TPM, not RPM, is what actually throttles free tiers."""

    async def test_large_prompt_is_paced_within_the_minute(self):
        budget = TokenBudget(tpm=6000, name="groq")
        for _ in range(4):
            assert await budget.acquire(1400) == 0.0   # 5,600 of 6,000 used
        waited = await budget.acquire(1400, timeout=10)
        assert waited > 0, "fifth 1,400-token call must wait for budget to free"

    async def test_timeout_when_budget_cannot_recover_in_time(self):
        budget = TokenBudget(tpm=6000, name="groq")
        await budget.acquire(6000)
        with pytest.raises(TimeoutError):
            await budget.acquire(6000, timeout=0.2)

    async def test_reconcile_corrects_an_over_estimate(self):
        budget = TokenBudget(tpm=6000, name="groq")
        await budget.acquire(2000)
        before = budget.remaining()
        budget.reconcile(estimated=2000, actual=500)
        assert budget.remaining() > before

    async def test_request_larger_than_the_whole_budget_does_not_deadlock(self):
        budget = TokenBudget(tpm=1000, name="tiny")
        assert await budget.acquire(50_000, timeout=5) == 0.0

    def test_estimate_includes_output_reserve(self):
        assert estimate_tokens(["a" * 400], max_output=100) == 200


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
class TestStreaming:
    async def test_stream_falls_over_before_first_token(self):
        a = FakeProvider("a", [TransientError("down", "a")])
        b = FakeProvider("b", [None], stream_chunks=["ok"])
        r = build_router(a, b)
        out = [c async for c in r.stream(MSGS)]
        assert out == ["ok"]

    async def test_stream_does_not_switch_provider_midway(self):
        """Once the user has seen text, silently restarting on another model
        would duplicate or contradict the visible answer."""
        a = FakeProvider("a", ["fail_after_first"], stream_chunks=["par", "tial"])
        b = FakeProvider("b", [None], stream_chunks=["different answer"])
        r = build_router(a, b)

        seen = []
        with pytest.raises(Exception):
            async for c in r.stream(MSGS):
                seen.append(c)
        assert seen == ["par", "tial"]
        assert b.calls == 0


# ---------------------------------------------------------------------------
# Error classification & JSON parsing
# ---------------------------------------------------------------------------
class TestErrorClassification:
    @pytest.mark.parametrize("status,expected,retryable", [
        (401, AuthError, False),
        (403, AuthError, False),
        (429, RateLimitError, True),
        (500, TransientError, True),
        (503, TransientError, True),
        (404, BadRequestError, False),
    ])
    def test_status_mapping(self, status, expected, retryable):
        err = classify_http_status(status, "msg", "p")
        assert isinstance(err, expected)
        assert err.retryable is retryable

    def test_gemini_reports_invalid_key_as_400(self):
        """Gemini returns 400 INVALID_ARGUMENT for a bad key, not 401."""
        err = classify_http_status(400, "API key not valid. Please pass a valid API key.", "gemini")
        assert isinstance(err, AuthError)
        assert err.fatal_for_provider is True


class TestJsonParsing:
    @pytest.mark.parametrize("raw", [
        '{"route": "SQL"}',
        '```json\n{"route": "SQL"}\n```',
        '```\n{"route": "SQL"}\n```',
        'Here is the answer:\n{"route": "SQL"}',
        '{"route": "SQL"}\n\nHope that helps!',
        '  \n {"route": "SQL"} \n ',
    ])
    def test_tolerates_common_wrappers(self, raw):
        assert parse_json_loose(raw)["route"] == "SQL"

    def test_nested_objects_survive(self):
        parsed = parse_json_loose('prefix {"a": {"b": [1, 2]}} suffix')
        assert parsed["a"]["b"] == [1, 2]

    @pytest.mark.parametrize("raw", ["", "   ", "not json at all", "[1,2,3]"])
    def test_rejects_unparseable(self, raw):
        with pytest.raises(ValueError):
            parse_json_loose(raw)
