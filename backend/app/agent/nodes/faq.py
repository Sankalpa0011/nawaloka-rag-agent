from __future__ import annotations

import hashlib
import time
from typing import Any

from qdrant_client import models

from app.agent.signals import RECORD_NOUNS
from app.agent.state import AgentState, trace_event
from app.config import settings
from app.logging_conf import get_logger
from app.retrieval.embedder import embed_query
from app.retrieval.qdrant_store import ensure_simple_collection, get_client

log = get_logger(__name__)

CURATED_THRESHOLD = settings.faq_curated_threshold
CACHE_THRESHOLD = settings.faq_cache_threshold
CACHE_TTL_SECONDS = settings.faq_cache_ttl_seconds

# these go stale on their own schedule, don't cache them
UNCACHEABLE_CONTENT = {"news", "promotion"}


def _normalise(query: str) -> str:
    return " ".join((query or "").lower().split()).strip(" ?.!")


def _corpus_version() -> str:
    """Current corpus version, used to scope cache entries."""
    try:
        import json

        from app.config import RAW_HTML_DIR

        manifest = json.loads((RAW_HTML_DIR / "manifest.json").read_text(encoding="utf-8"))
        return str(manifest.get("generated_at", "unknown"))
    except Exception:
        return "unknown"

def faq_lookup(query: str) -> dict[str, Any] | None:
    """Tier 1: curated FAQ collection."""
    if not settings.enable_faq_fastpath:
        return None
    name = settings.qdrant_faq_collection
    client = get_client()
    if not client.collection_exists(name):
        return None

    try:
        hits = client.query_points(
            collection_name=name,
            query=embed_query(_normalise(query)),
            limit=1,
            with_payload=True,
        ).points
    except Exception as exc:
        log.warning("faq.lookup_failed", error=str(exc)[:160])
        return None

    if not hits or hits[0].score < settings.faq_curated_threshold:
        return None

    payload = hits[0].payload or {}
    return {
        "tier": "curated",
        "faq_id": payload.get("faq_id"),
        "answer": payload.get("answer", ""),
        "score": float(hits[0].score),
        "source_urls": payload.get("source_urls", []),
        "volatility": payload.get("volatility", "static"),
    }


def cache_lookup(query: str) -> dict[str, Any] | None:
    """Tier 2: semantic cache of previously produced answers."""
    if not settings.enable_semantic_cache:
        return None
    name = settings.qdrant_cache_collection
    client = get_client()
    if not client.collection_exists(name):
        return None

    try:
        hits = client.query_points(
            collection_name=name,
            query=embed_query(_normalise(query)),
            limit=1,
            with_payload=True,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="corpus_version",
                        match=models.MatchValue(value=_corpus_version()),
                    )
                ]
            ),
        ).points
    except Exception as exc:
        log.warning("faq.cache_lookup_failed", error=str(exc)[:160])
        return None

    if not hits or hits[0].score < settings.faq_cache_threshold:
        return None

    payload = hits[0].payload or {}
    age = time.time() - float(payload.get("created_at", 0))
    if age > settings.faq_cache_ttl_seconds:
        return None

    return {
        "tier": "cache",
        "answer": payload.get("answer", ""),
        "citations": payload.get("citations", []),
        "score": float(hits[0].score),
        "original_query": payload.get("query", ""),
    }


def cache_eligibility(state: AgentState) -> tuple[bool, str]:
    """Decide whether this turn's answer may be cached. Returns (ok, reason)."""
    if not settings.enable_semantic_cache:
        return False, "cache disabled"

    route = state.get("route", "")
    if route in ("SQL", "HYBRID"):
        return False, "database-derived: fees, schedules and prices are mutable"
    if route in ("EMERGENCY", "REFUSE"):
        return False, "guardrail response"
    if route in ("SMALLTALK", "OUT_OF_SCOPE"):
        return False, "canned response, not specific to the question"
    if state.get("abstained"):
        return False, "answer was an abstention"
    if state.get("error"):
        return False, "turn produced an error"

    guard = state.get("guardrail") or {}
    if guard.get("flags"):
        return False, "guardrail flags set"

    if _normalise(state.get("contextual_query", "")) != _normalise(state.get("user_query", "")):
        return False, "answer depended on conversation context"

    for doc in state.get("documents") or []:
        if doc.get("content_type") in UNCACHEABLE_CONTENT:
            return False, f"built on time-sensitive content ({doc.get('content_type')})"

    answer_text = state.get("answer") or ""
    if not answer_text.strip():
        return False, "empty answer"

    unhealthy_indicators = [
        "_Please confirm current prices",
        "_This answer was cut short",
        "I could not finish that answer",
        "error occurred",
        "unable to process"
    ]
    
    for indicator in unhealthy_indicators:
        if indicator in answer_text:
            return False, f"unhealthy answer detected: contains '{indicator}'"

    return True, "cacheable"


def cache_write(state: AgentState) -> bool:
    ok, reason = cache_eligibility(state)
    if not ok:
        log.debug("faq.cache_skipped", reason=reason, route=state.get("route"))
        return False

    query = _normalise(state.get("contextual_query") or state["user_query"])
    client = get_client()
    name = settings.qdrant_cache_collection
    ensure_simple_collection(client, name)

    point_id = int(hashlib.sha1(query.encode("utf-8")).hexdigest()[:15], 16)
    try:
        client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embed_query(query),
                    payload={
                        "query": query,
                        "answer": state.get("answer", ""),
                        "citations": state.get("citations", []),
                        "route": state.get("route"),
                        "corpus_version": _corpus_version(),
                        "created_at": time.time(),
                    },
                )
            ],
            wait=False,
        )
        log.info("faq.cached", query=query[:60], route=state.get("route"))
        return True
    except Exception as exc:
        log.warning("faq.cache_write_failed", error=str(exc)[:160])
        return False


# the curated FAQs only hold general answers.
LIVE_DATA_ENTITIES = {"doctor", "lab_test", "package", "weekday", "specialty"}
LIVE_DATA_NOUNS = tuple(sorted(RECORD_NOUNS))


def needs_live_data(query: str) -> list[str]:

    from app.agent.signals import _contains_term, extract_signals

    signals = extract_signals(query)
    blocking = sorted(set(signals.entities) & LIVE_DATA_ENTITIES)

    text = " " + (query or "").lower().strip() + " "
    nouns = []
    for n in LIVE_DATA_NOUNS:
        if _contains_term(text, n):
            nouns.append(n)
    if nouns:
        blocking.append(f"record type ({nouns[0]})")

    if signals.sql_hits:
        blocking.append(f"transactional terms ({', '.join(sorted(set(signals.sql_hits))[:3])})")

    return blocking


async def faq_node(state: AgentState) -> dict:
    """Check both tiers. A hit short-circuits routing, retrieval and synthesis."""
    started = time.perf_counter()
    query = state.get("contextual_query") or state["user_query"]

    blocking = needs_live_data(query)

    guard = state.get("guardrail") or {}
    if "symptom_context" in (guard.get("flags") or []):
        blocking.append("a symptom")
    if blocking:
        elapsed = int((time.perf_counter() - started) * 1000)
        log.info("faq.skipped_live_data", entities=blocking)
        return {
            "cache": {"status": "bypassed", "reason": f"names {', '.join(blocking)}",
                      "elapsed_ms": elapsed},
            "trace": [trace_event("faq", status="bypassed",
                                  reason=f"query names {', '.join(blocking)}",
                                  ms=elapsed)],
        }

    hit = faq_lookup(query) or cache_lookup(query)
    elapsed = int((time.perf_counter() - started) * 1000)

    if not hit:
        return {
            "cache": {"status": "miss", "elapsed_ms": elapsed},
            "trace": [trace_event("faq", status="miss", ms=elapsed)],
        }

    citations = hit.get("citations") or [
        {"index": i, "label": url, "url": url, "source": "web"}
        for i, url in enumerate(hit.get("source_urls") or [], start=1)
    ]

    log.info("faq.hit", tier=hit["tier"], score=round(hit["score"], 3), ms=elapsed)
    return {
        "route": "FAQ",
        "answer": hit["answer"],
        "citations": citations,
        "abstained": False,
        "cache": {
            "status": "hit",
            "tier": hit["tier"],
            "score": round(hit["score"], 4),
            "faq_id": hit.get("faq_id"),
            "elapsed_ms": elapsed,
        },
        "trace": [
            trace_event("faq", status="hit", tier=hit["tier"],
                        score=round(hit["score"], 3), ms=elapsed)
        ],
    }
