"""Groq provider (Llama 3.x, OpenAI-compatible chat completions)."""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import Any

from app.config import settings
from app.llm.base import (
    Msg,
    Provider,
    Tier,
    TransientError,
    Usage,
    classify_http_status,
)
from app.logging_conf import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _client():
    from groq import AsyncGroq

    return AsyncGroq(api_key=settings.groq_api_key, max_retries=0)  # we own retries


def _raise_for(exc: Exception) -> None:
    status = getattr(exc, "status_code", None)
    message = str(exc)
    if status:
        raise classify_http_status(int(status), message, "groq") from exc
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        raise TransientError(f"timeout: {message}", "groq") from exc
    raise TransientError(message, "groq") from exc


class GroqProvider(Provider):
    name = "groq"

    def is_configured(self) -> bool:
        return bool(settings.groq_api_key)

    def model_for(self, tier: Tier) -> str:
        return (
            settings.llm_fast_model_groq
            if tier == "fast"
            else settings.llm_strong_model_groq
        )

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
        payload = [m.to_dict() for m in messages]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": payload,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_schema:
            # Groq has no schema mode, so describe the shape in the prompt instead.
            kwargs["response_format"] = {"type": "json_object"}
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

        try:
            resp = await asyncio.wait_for(
                _client().chat.completions.create(**kwargs), timeout=timeout
            )
        except Exception as exc:
            _raise_for(exc)
            raise

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = Usage()
        if resp.usage:
            usage = Usage(
                input_tokens=resp.usage.prompt_tokens or 0,
                output_tokens=resp.usage.completion_tokens or 0,
            )
        return text, usage, str(choice.finish_reason or "")

    async def stream(
        self,
        messages: list[Msg],
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 90.0,
    ):
        try:
            stream = await _client().chat.completions.create(
                model=model,
                messages=[m.to_dict() for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as exc:
            _raise_for(exc)
