"""OpenRouter provider (OpenAI-compatible chat completions, many vendors), over plain httpx.

Unlike Groq: can return 200 with an `error` body object, and its SSE stream
sends non-JSON `: OPENROUTER PROCESSING` keep-alive comments to skip.
"""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import Any

import httpx

from app.config import settings
from app.llm.base import (
    ModelUnavailableError,
    Msg,
    Provider,
    ProviderError,
    RateLimitError,
    Tier,
    TransientError,
    Usage,
    classify_http_status,
)
from app.logging_conf import get_logger

log = get_logger(__name__)

# Optional attribution headers for OpenRouter's public model rankings.
ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://github.com/nawaloka-agent",
    "X-Title": "Nawaloka Agent",
}


@lru_cache(maxsize=1)
def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=settings.openrouter_base_url,
        headers={
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            **ATTRIBUTION_HEADERS,
        },
        # Per-request timeouts are passed explicitly; this is only a backstop.
        timeout=httpx.Timeout(60.0, connect=10.0),
    )


def _raise_for(exc: Exception) -> None:
    if isinstance(exc, ProviderError):
        # Already classified by _check_response; don't downgrade it.
        raise exc
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        raise TransientError(f"timeout: {exc}", "openrouter") from exc
    if isinstance(exc, httpx.RequestError):
        raise TransientError(f"transport: {exc}", "openrouter") from exc
    raise TransientError(str(exc), "openrouter") from exc


def _classify(status: int, message: str, retry_after: str | None = None) -> ProviderError:
    """Map a status onto the shared taxonomy; a 404 means the model id is gone, not a bad prompt."""
    if status == 404 or "not a valid model" in message.lower():
        return ModelUnavailableError(message, "openrouter", status)
    if status == 429:
        return RateLimitError(
            message,
            "openrouter",
            retry_after=float(retry_after) if retry_after else None,
        )
    return classify_http_status(status, message, "openrouter")


def _check_response(resp: httpx.Response) -> dict:
    """Return the decoded body, raising the right ProviderError for HTTP or 200-with-error failures."""
    if resp.status_code != 200:
        raise _classify(
            resp.status_code, resp.text[:500], resp.headers.get("retry-after")
        )

    try:
        body = resp.json()
    except ValueError as exc:
        raise TransientError(f"non-JSON body: {resp.text[:200]}", "openrouter") from exc

    error = body.get("error")
    if error:
        # 200-with-error: the nested code still mirrors an HTTP status.
        code = error.get("code")
        message = str(error.get("message") or error)
        if isinstance(code, int) and code >= 400:
            raise _classify(code, message)
        raise TransientError(message, "openrouter")

    return body


class OpenRouterProvider(Provider):
    name = "openrouter"

    def is_configured(self) -> bool:
        return bool(settings.openrouter_api_key)

    def model_for(self, tier: Tier) -> str:
        return (
            settings.llm_fast_model_openrouter
            if tier == "fast"
            else settings.llm_strong_model_openrouter
        )

    def _payload(
        self,
        messages: list[Msg],
        model: str,
        json_schema: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload = [m.to_dict() for m in messages]

        if json_schema:
            hint = (
                "Respond with a single JSON object matching this schema exactly, "
                "with no commentary:\n" + json.dumps(json_schema, indent=2)
            )
            for m in reversed(payload):
                if m["role"] == "system":
                    m["content"] = f"{m['content']}\n\n{hint}"
                    break
            else:
                payload.insert(0, {"role": "system", "content": hint})

        return {
            "model": model,
            "messages": payload,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Reasoning tokens share the max_tokens budget with the answer and can
            # consume it entirely; disabled for every model (safe even if unsupported).
            "reasoning": {"enabled": False},
        }

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
        body = self._payload(messages, model, json_schema, temperature, max_tokens)

        try:
            resp = await _client().post(
                "/chat/completions", json=body, timeout=timeout
            )
        except Exception as exc:
            _raise_for(exc)
            raise  # unreachable; keeps type checkers happy

        data = _check_response(resp)

        choices = data.get("choices") or []
        if not choices:
            raise TransientError(f"no choices returned by {model}", "openrouter")

        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        finish = str(choices[0].get("finish_reason") or "")

        usage = Usage()
        raw_usage = data.get("usage") or {}
        if raw_usage:
            usage = Usage(
                input_tokens=raw_usage.get("prompt_tokens") or 0,
                output_tokens=raw_usage.get("completion_tokens") or 0,
            )

        if not text.strip():
            # Free slugs return an empty body when the upstream vendor is at
            # capacity. That is a failover signal, not an answer.
            raise TransientError(
                f"empty response from {model} (finish_reason={finish or 'unknown'})",
                "openrouter",
            )
        return text, usage, finish

    async def stream(
        self,
        messages: list[Msg],
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 90.0,
    ):
        body = self._payload(messages, model, None, temperature, max_tokens)
        body["stream"] = True

        try:
            async with _client().stream(
                "POST", "/chat/completions", json=body, timeout=timeout
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    _check_response(resp)

                async for line in resp.aiter_lines():
                    line = line.strip()
                    # ": OPENROUTER PROCESSING" keep-alives arrive as SSE
                    # comments while a cold model spins up.
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("error"):
                        raise TransientError(
                            str(chunk["error"])[:200], "openrouter"
                        )
                    chunk_choices = chunk.get("choices") or []
                    if not chunk_choices:
                        continue
                    delta = (chunk_choices[0].get("delta") or {}).get("content")
                    if delta:
                        yield delta
        except Exception as exc:
            _raise_for(exc)
