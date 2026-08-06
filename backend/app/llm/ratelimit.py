"""Client-side rate limiting and circuit breaking.

Token bucket paces requests to each provider's RPM/TPM; circuit breaker parks
a failing provider for a cooldown instead of retrying it every turn.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from app.config import settings
from app.logging_conf import get_logger

log = get_logger(__name__)


# token bucket, topped up continuously at rpm/60 per second
class AsyncTokenBucket:
    def __init__(self, rpm: int, burst: int | None = None, name: str = ""):
        self.rpm = max(1, rpm)
        self.capacity = (
            burst
            if burst is not None
            else max(1, min(self.rpm, settings.llm_bucket_burst))
        )
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

    async def acquire(
        self, timeout: float = settings.llm_bucket_timeout_seconds
    ) -> float:
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
            slice_ = min(sleep_for, settings.llm_wait_slice_seconds)
            await asyncio.sleep(slice_)
            waited += slice_

    def tokens_available(self) -> float:
        self._replenish()
        return self._tokens


class TokenBudget:
    """Tokens-per-minute budget, tracked separately from request count; estimated then reconciled."""

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
            # Decay continuously rather than a hard window, to avoid a burst doubling up.
            self._used = max(0.0, self._used - (elapsed * self.tpm / 60.0))
            self._window_started = now

    async def acquire(
        self,
        estimated_tokens: int,
        timeout: float = settings.llm_budget_timeout_seconds,
    ) -> float:
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
            slice_ = min(sleep_for, settings.llm_wait_slice_seconds)
            await asyncio.sleep(slice_)
            waited += slice_

    # fix up the budget once we know what the call actually cost
    def reconcile(self, estimated: int, actual: int) -> None:
        delta = float(actual) - float(min(estimated, self.tpm))
        self._used = max(0.0, min(float(self.tpm), self._used + delta))

    def remaining(self) -> float:
        self._roll()
        return max(0.0, self.tpm - self._used)


# Rough token count for budgeting - ~4 chars/token plus whatever we reserved for
# output. Same approximation as chunk.py's _tokens().
def estimate_tokens(texts: list[str], max_output: int = 0) -> int:
    total = 0
    for t in texts:
        total += len(t)
    return total // settings.approx_chars_per_token + max_output


# closed -> open after `threshold` failures in a row, then half-open to retry
@dataclass
class CircuitBreaker:
    threshold: int = settings.llm_breaker_threshold
    cooldown: float = settings.llm_breaker_cooldown_seconds
    max_cooldown: float = settings.llm_breaker_max_cooldown_seconds
    name: str = ""
    failures: int = 0
    reopens: int = 0
    opened_at: float | None = None
    _half_open: bool = field(default=False, repr=False)

    @property
    def current_cooldown(self) -> float:
        """Cooldown for this open period; doubles on every failed retry probe."""
        return min(self.max_cooldown, self.cooldown * (2 ** self.reopens))

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at >= self.current_cooldown:
            # cooldown's up, let one request through and see what happens
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
