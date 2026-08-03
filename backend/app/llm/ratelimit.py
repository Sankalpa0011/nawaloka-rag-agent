"""Client-side rate limiting and circuit breaking.

The brief asks for rate limiting to be handled "gracefully", and on free tiers
that means not sending the request in the first place. Gemini's free flash tier
allows roughly 10-15 requests per minute; a 58-query eval run issues a routing
call and a synthesis call for each, so without a limiter the run collapses into
a wall of 429s and the agent looks broken when it is merely throttled.

Two mechanisms:

* **Token bucket** - paces outgoing requests to the provider's published RPM.
  Callers *wait* rather than fail, so a burst is smoothed instead of rejected.
* **Circuit breaker** - after repeated failures a provider is parked for a
  cooldown instead of being retried on every turn. Prevents one dead provider
  from adding its timeout to every single request.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from app.logging_conf import get_logger

log = get_logger(__name__)


class AsyncTokenBucket:
    """Simple async token bucket, refilled continuously at ``rate`` per minute."""

    def __init__(self, rpm: int, burst: int | None = None, name: str = ""):
        self.rpm = max(1, rpm)
        self.capacity = burst if burst is not None else max(1, min(self.rpm, 5))
        self._tokens = float(self.capacity)
        self._refill_per_sec = self.rpm / 60.0
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()
        self.name = name

    def _replenish(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated
        self._updated = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self._refill_per_sec)

    async def acquire(self, timeout: float = 30.0) -> float:
        """Wait for a token. Returns seconds waited. Raises TimeoutError."""
        deadline = time.monotonic() + timeout
        waited = 0.0
        while True:
            async with self._lock:
                self._replenish()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return waited

                deficit = 1.0 - self._tokens
                sleep_for = deficit / self._refill_per_sec

            if time.monotonic() + sleep_for > deadline:
                raise TimeoutError(
                    f"rate limit wait for {self.name} would exceed {timeout:.0f}s"
                )
            # Cap each sleep so several waiters wake and re-check fairly.
            slice_ = min(sleep_for, 0.5)
            await asyncio.sleep(slice_)
            waited += slice_

    def tokens_available(self) -> float:
        self._replenish()
        return self._tokens


class TokenBudget:
    """Tokens-per-minute budget, tracked separately from request count.

    On free tiers TPM - not RPM - is usually what actually bites. Groq allows
    25 requests/minute on llama-3.1-8b-instant but only 6,000 tokens/minute, so
    a 1,400-token prompt is throttled after roughly four requests even though
    the request budget is nowhere near exhausted. Pacing on RPM alone produces
    a burst of 429s that looks like a broken agent.

    Consumption is estimated before the call and reconciled with the provider's
    reported usage afterwards, so the estimate drifting does not compound.
    """

    def __init__(self, tpm: int, name: str = ""):
        self.tpm = max(1, tpm)
        self.name = name
        self._used = 0.0
        self._window_started = time.monotonic()
        self._lock = asyncio.Lock()

    def _roll(self) -> None:
        now = time.monotonic()
        elapsed = now - self._window_started
        if elapsed >= 60.0:
            self._used = 0.0
            self._window_started = now
        else:
            # Continuous decay rather than a hard window, so a burst at the end
            # of one minute does not immediately double up at the start of the
            # next.
            self._used = max(0.0, self._used - (elapsed * self.tpm / 60.0))
            self._window_started = now

    async def acquire(self, estimated_tokens: int, timeout: float = 60.0) -> float:
        deadline = time.monotonic() + timeout
        waited = 0.0
        want = min(float(estimated_tokens), float(self.tpm))

        while True:
            async with self._lock:
                self._roll()
                if self._used + want <= self.tpm:
                    self._used += want
                    return waited
                over = self._used + want - self.tpm
                sleep_for = over / (self.tpm / 60.0)

            if time.monotonic() + sleep_for > deadline:
                raise TimeoutError(
                    f"token budget wait for {self.name} would exceed {timeout:.0f}s "
                    f"(need {want:.0f}, {self.remaining():.0f} left this minute)"
                )
            slice_ = min(sleep_for, 0.5)
            await asyncio.sleep(slice_)
            waited += slice_

    def reconcile(self, estimated: int, actual: int) -> None:
        """Correct the budget once the provider reports real usage."""
        delta = float(actual) - float(min(estimated, self.tpm))
        self._used = max(0.0, min(float(self.tpm), self._used + delta))

    def remaining(self) -> float:
        self._roll()
        return max(0.0, self.tpm - self._used)


def estimate_tokens(texts: list[str], max_output: int = 0) -> int:
    """Rough token count for budgeting. ~4 chars/token plus the output reserve."""
    return sum(len(t) for t in texts) // 4 + max_output


@dataclass
class CircuitBreaker:
    """Closed -> open after ``threshold`` consecutive failures, then half-open."""

    threshold: int = 3
    cooldown: float = 60.0
    max_cooldown: float = 900.0
    name: str = ""
    failures: int = 0
    reopens: int = 0
    opened_at: float | None = None
    _half_open: bool = field(default=False, repr=False)

    @property
    def current_cooldown(self) -> float:
        """Cooldown for this open period, doubling each time a trial fails.

        A provider whose trial request keeps failing is not briefly throttled,
        it is out of quota for the day, and a fixed 60s cooldown re-probes it
        forever. The cost is not just the wasted round-trip: an open circuit is
        what removes a provider from the router's eligible list, and while a
        dead provider still *looks* eligible the router keeps its failover wait
        short - so the one healthy provider is never allowed to wait out its own
        token budget. Observed with Gemini's daily quota exhausted: Groq had
        4,180 of 6,000 tokens free and needed 5,045, which one minute of waiting
        would have covered, but the turn failed outright instead.
        """
        return min(self.max_cooldown, self.cooldown * (2 ** self.reopens))

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at >= self.current_cooldown:
            # Cooldown elapsed: allow a single trial request through.
            self._half_open = True
            return False
        return True

    def record_success(self) -> None:
        if self.failures or self.opened_at:
            log.info("circuit.closed", provider=self.name)
        self.failures = 0
        self.reopens = 0
        self.opened_at = None
        self._half_open = False

    def record_failure(self) -> None:
        self.failures += 1
        if self._half_open:
            # The trial request failed, so back off for twice as long.
            self.reopens += 1
            self.opened_at = time.monotonic()
            self._half_open = False
            log.warning("circuit.reopened", provider=self.name,
                        failures=self.failures, cooldown=self.current_cooldown)
            return
        if self.failures >= self.threshold and self.opened_at is None:
            self.opened_at = time.monotonic()
            log.warning("circuit.opened", provider=self.name,
                        failures=self.failures, cooldown=self.current_cooldown)

    def seconds_until_retry(self) -> float:
        if self.opened_at is None:
            return 0.0
        return max(0.0, self.current_cooldown - (time.monotonic() - self.opened_at))

    def status(self) -> str:
        if self.is_open:
            return "open"
        return "half_open" if self._half_open else "closed"
