"""Probes the configured API keys and prints which models actually respond.

    python -m app.llm.check_models
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

# Substrings that mark a model as belonging in the *fast* tier. Crude, but the
# alternative is a hand-maintained table that goes stale the moment a vendor
# ships a new size.
SMALL_MODEL_TOKENS = ("lite", "mini", "small", "3b", "7b", "8b", "9b", "instant")


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


# OpenRouter lists 300+ models. Probing all of them would burn the free tier's
# whole daily request allowance, so only free slugs are tried, newest first,
# capped at this many.
OPENROUTER_PROBE_LIMIT = 12


def _is_free(model: dict) -> bool:
    pricing = model.get("pricing") or {}
    try:
        return float(pricing.get("prompt", 1)) == 0 and float(pricing.get("completion", 1)) == 0
    except (TypeError, ValueError):
        return False


async def openrouter_models() -> list[tuple[str, int, str]]:
    if not settings.openrouter_api_key:
        print("  OPENROUTER_API_KEY not set - skipping")
        return []

    base = settings.openrouter_base_url
    headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"}

    async with httpx.AsyncClient(timeout=60) as client:
        listing = await client.get(f"{base}/models", headers=headers)
        if listing.status_code != 200:
            print(f"  list failed {listing.status_code}: {listing.text[:200]}")
            return []

        free = [
            m for m in listing.json().get("data", [])
            if _is_free(m) and not any(tok in m["id"] for tok in SKIP_TOKENS)
        ]
        free.sort(key=lambda m: m.get("created", 0), reverse=True)
        candidates = [m["id"] for m in free[:OPENROUTER_PROBE_LIMIT]]
        print(f"  {len(free)} free models listed, probing the {len(candidates)} newest")

        working: list[tuple[str, int, str]] = []
        for name in candidates:
            started = time.perf_counter()
            resp = await client.post(
                f"{base}/chat/completions",
                headers=headers,
                json={
                    "model": name,
                    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                    "max_tokens": 10,
                    "temperature": 0,
                },
            )
            ms = int((time.perf_counter() - started) * 1000)
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            # OpenRouter answers 200 with an error object when the upstream
            # vendor rejects the call, so status alone is not enough.
            if resp.status_code == 200 and not body.get("error"):
                working.append((name, ms, "openrouter"))
                print(f"  {'OK':<5} {name:<44}{ms:>7} ms")
            else:
                detail = str(body.get("error") or resp.text)[:64]
                print(f"  {resp.status_code:<5} {name:<44}{detail}")
            # 20 requests/minute on the free tier.
            await asyncio.sleep(3.2)
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
    print("OPENROUTER (free models only)")
    print("=" * 74)
    orr = await openrouter_models()

    print("\n" + "=" * 74)
    print("SUGGESTED .env SETTINGS (fastest working model per tier)")
    print("=" * 74)
    for label, models in (("GEMINI", gem), ("GROQ", grq), ("OPENROUTER", orr)):
        if not models:
            print(f"  {label}: no working models")
            continue
        ranked = sorted(models, key=lambda x: x[1])
        small = lambda name: any(t in name.lower() for t in SMALL_MODEL_TOKENS)
        fast = next((m for m in ranked if small(m[0])), ranked[0])
        strong = next((m for m in ranked if not small(m[0])), ranked[0])
        print(f"  LLM_FAST_MODEL_{label}={fast[0]}")
        print(f"  LLM_STRONG_MODEL_{label}={strong[0]}")

    print(f"\ncurrently configured:")
    print(f"  gemini     fast={settings.llm_fast_model_gemini} "
          f"strong={settings.llm_strong_model_gemini}")
    print(f"  groq       fast={settings.llm_fast_model_groq} "
          f"strong={settings.llm_strong_model_groq}")
    print(f"  openrouter fast={settings.llm_fast_model_openrouter} "
          f"strong={settings.llm_strong_model_openrouter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
