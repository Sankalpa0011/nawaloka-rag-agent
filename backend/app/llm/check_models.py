"""Discover which LLM models the configured keys can actually call.

    python -m app.llm.check_models

Model availability is account-specific and changes without notice. A newly
created Google AI Studio account, for example, cannot call the gemini-2.x
family at all - 2.5 returns 404 "no longer available to new users" and 2.0
returns 429 - even though those model names appear in every tutorial. Rather
than hardcoding a guess, this probes the live keys and prints models that
genuinely respond, so .env can be set from evidence.
"""

from __future__ import annotations

import asyncio
import sys
import time

import httpx

from app.config import settings

SKIP_TOKENS = (
    "image", "tts", "embedding", "aqa", "vision", "live",
    "native-audio", "computer", "lyria", "banana", "robotics",
)


async def gemini_models() -> list[tuple[str, int, str]]:
    if not settings.gemini_api_key:
        print("  GEMINI_API_KEY not set - skipping")
        return []

    headers = {"x-goog-api-key": settings.gemini_api_key, "Content-Type": "application/json"}
    base = "https://generativelanguage.googleapis.com/v1beta"

    async with httpx.AsyncClient(timeout=60) as client:
        listing = await client.get(f"{base}/models", headers=headers)
        if listing.status_code != 200:
            print(f"  ListModels failed {listing.status_code}: {listing.text[:200]}")
            return []

        candidates = [
            m["name"].split("/")[-1]
            for m in listing.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
            and not any(tok in m["name"] for tok in SKIP_TOKENS)
        ]

        working: list[tuple[str, int, str]] = []
        for name in candidates:
            started = time.perf_counter()
            try:
                resp = await client.post(
                    f"{base}/models/{name}:generateContent",
                    headers=headers,
                    json={
                        "contents": [{"parts": [{"text": "Reply with exactly: OK"}]}],
                        "generationConfig": {"maxOutputTokens": 800, "temperature": 0},
                    },
                )
            except Exception as exc:
                print(f"  {'EXC':<5} {name:<34}{str(exc)[:60]}")
                continue

            ms = int((time.perf_counter() - started) * 1000)
            if resp.status_code == 200:
                working.append((name, ms, "gemini"))
                print(f"  {'OK':<5} {name:<34}{ms:>7} ms")
            else:
                msg = resp.json().get("error", {}).get("message", "")[:64]
                print(f"  {resp.status_code:<5} {name:<34}{msg}")
            await asyncio.sleep(0.8)
    return working


async def groq_models() -> list[tuple[str, int, str]]:
    if not settings.groq_api_key:
        print("  GROQ_API_KEY not set - skipping")
        return []

    headers = {"Authorization": f"Bearer {settings.groq_api_key}"}
    async with httpx.AsyncClient(timeout=60) as client:
        listing = await client.get("https://api.groq.com/openai/v1/models", headers=headers)
        if listing.status_code != 200:
            print(f"  list failed {listing.status_code}: {listing.text[:200]}")
            return []

        candidates = [
            m["id"] for m in listing.json().get("data", [])
            if not any(tok in m["id"] for tok in ("whisper", "tts", "guard", "vision"))
        ]

        working: list[tuple[str, int, str]] = []
        for name in candidates:
            started = time.perf_counter()
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": name,
                    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                    "max_tokens": 10,
                    "temperature": 0,
                },
            )
            ms = int((time.perf_counter() - started) * 1000)
            if resp.status_code == 200:
                tpm = resp.headers.get("x-ratelimit-limit-tokens", "?")
                rpm = resp.headers.get("x-ratelimit-limit-requests", "?")
                working.append((name, ms, "groq"))
                print(f"  {'OK':<5} {name:<34}{ms:>7} ms   tpm={tpm} rpm={rpm}")
            else:
                print(f"  {resp.status_code:<5} {name:<34}{resp.text[:64]}")
            await asyncio.sleep(0.5)
    return working


async def main() -> int:
    print("=" * 74)
    print("GEMINI")
    print("=" * 74)
    gem = await gemini_models()

    print("\n" + "=" * 74)
    print("GROQ")
    print("=" * 74)
    grq = await groq_models()

    print("\n" + "=" * 74)
    print("SUGGESTED .env SETTINGS (fastest working model per tier)")
    print("=" * 74)
    for label, models in (("GEMINI", gem), ("GROQ", grq)):
        if not models:
            print(f"  {label}: no working models")
            continue
        ranked = sorted(models, key=lambda x: x[1])
        fast = next((m for m in ranked if "lite" in m[0] or "8b" in m[0]), ranked[0])
        strong = next(
            (m for m in ranked if "lite" not in m[0] and "8b" not in m[0]), ranked[0]
        )
        print(f"  LLM_FAST_MODEL_{label}={fast[0]}")
        print(f"  LLM_STRONG_MODEL_{label}={strong[0]}")

    print(f"\ncurrently configured:")
    print(f"  gemini fast={settings.llm_fast_model_gemini} "
          f"strong={settings.llm_strong_model_gemini}")
    print(f"  groq   fast={settings.llm_fast_model_groq} "
          f"strong={settings.llm_strong_model_groq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
