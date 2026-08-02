"""Google Gemini provider (AI Studio free tier)."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

from app.config import settings
from app.llm.base import (
    AuthError,
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
    from google import genai

    return genai.Client(api_key=settings.gemini_api_key)


def _to_gemini(messages: list[Msg]) -> tuple[str | None, list[dict]]:
    """Split out the system instruction; Gemini takes it separately."""
    system: list[str] = []
    contents: list[dict] = []
    for m in messages:
        if m.role == "system":
            system.append(m.content)
            continue
        # Gemini calls the assistant role "model".
        role = "model" if m.role == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m.content}]})
    return ("\n\n".join(system) if system else None), contents


def _json_schema_to_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Translate a JSON-Schema fragment to Gemini's uppercase type names."""
    type_map = {
        "object": "OBJECT", "array": "ARRAY", "string": "STRING",
        "number": "NUMBER", "integer": "INTEGER", "boolean": "BOOLEAN",
    }
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            out["type"] = type_map.get(value.lower(), value.upper())
        elif key == "properties" and isinstance(value, dict):
            out["properties"] = {k: _json_schema_to_gemini(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            out["items"] = _json_schema_to_gemini(value)
        elif key in ("required", "enum", "description", "nullable"):
            out[key] = value
        # Anything else (additionalProperties, $schema, ...) is dropped:
        # Gemini rejects unknown schema keys outright.
    return out


def _raise_for(exc: Exception) -> None:
    """Translate an SDK exception into our error taxonomy."""
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    message = str(exc)

    if status is None:
        # google-genai raises ClientError/ServerError with the code in the text.
        import re

        m = re.search(r"\b(4\d{2}|5\d{2})\b", message)
        status = int(m.group(1)) if m else None

    if status:
        raise classify_http_status(int(status), message, "gemini") from exc
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        raise TransientError(f"timeout: {message}", "gemini") from exc
    raise TransientError(message, "gemini") from exc


class GeminiProvider(Provider):
    name = "gemini"

    def is_configured(self) -> bool:
        return bool(settings.gemini_api_key)

    def model_for(self, tier: Tier) -> str:
        return (
            settings.llm_fast_model_gemini
            if tier == "fast"
            else settings.llm_strong_model_gemini
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
        from google.genai import types as gt

        system, contents = _to_gemini(messages)
        cfg: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system:
            cfg["system_instruction"] = system
        if json_schema:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = _json_schema_to_gemini(json_schema)

        try:
            resp = await asyncio.wait_for(
                _client().aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=gt.GenerateContentConfig(**cfg),
                ),
                timeout=timeout,
            )
        except Exception as exc:
            _raise_for(exc)
            raise  # unreachable; keeps type checkers happy

        text = resp.text or ""
        usage = Usage()
        meta = getattr(resp, "usage_metadata", None)
        if meta:
            usage = Usage(
                input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
                output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            )

        finish = ""
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            finish = str(getattr(candidates[0], "finish_reason", "") or "")

        if not text.strip():
            # An empty body with a safety finish reason is not a transport
            # failure - retrying produces the same result.
            raise TransientError(
                f"empty response from {model} (finish_reason={finish or 'unknown'})",
                "gemini",
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
        from google.genai import types as gt

        system, contents = _to_gemini(messages)
        cfg: dict[str, Any] = {"temperature": temperature, "max_output_tokens": max_tokens}
        if system:
            cfg["system_instruction"] = system

        try:
            # NOTE: generate_content_stream returns an async generator directly,
            # it is not awaitable. Awaiting it raises
            # "object async_generator can't be used in 'await' expression".
            stream = _client().aio.models.generate_content_stream(
                model=model, contents=contents, config=gt.GenerateContentConfig(**cfg)
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            _raise_for(exc)


#: Prefixes Google AI Studio has issued. ``AIza`` is the long-standing format;
#: ``AQ.`` is the newer one now issued to fresh accounts. Both are valid, so
#: this is only used to catch obviously-wrong credentials (OAuth tokens,
#: service-account JSON), never to reject a key.
KNOWN_KEY_PREFIXES = ("AIza", "AQ.")


def check_key_format(key: str) -> str | None:
    """Warn only when a value is clearly not an AI Studio key at all.

    Note: do NOT treat an unfamiliar prefix as fatal. Google has changed the
    key format at least once (``AIza…`` -> ``AQ.…``), and a hard format check
    would reject perfectly valid new keys. Authentication is decided by the
    API, not by us; this exists purely to make a common paste-error obvious.
    """
    if not key:
        return None
    if key.startswith(KNOWN_KEY_PREFIXES):
        return None
    if key.startswith(("ya29.", "1//", "{")):
        return (
            "GEMINI_API_KEY looks like an OAuth token or service-account "
            "credential rather than an AI Studio API key. Create one at "
            "https://aistudio.google.com/apikey"
        )
    return (
        f"GEMINI_API_KEY has an unrecognised prefix {key[:4]!r}. It may still be "
        "valid - the API decides - but if requests fail with "
        "'400 API key not valid', check https://aistudio.google.com/apikey"
    )
