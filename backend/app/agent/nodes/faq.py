"""FAQ fast path and semantic response cache.

Two tiers, both a single embedding + one vector lookup (~40 ms, zero LLM calls)
against a normal turn's ~6 s and three LLM calls:

**Tier 1 - curated FAQs.** Hand-authored, human-vetted answers to the questions
patients actually ask. Threshold 0.90.

**Tier 2 - semantic cache.** Answers this system produced earlier, reused for
semantically equivalent questions. Threshold 0.93, higher because the content
was not human-reviewed.

The interesting part is not the lookup, it is **what must never be cached**.
Everything transactional in this domain is mutable, so:

* `route in {SQL, HYBRID}` is never cached - fees, schedules, prices and
  package contents all live in the database and can change at any time. Serving
  a stale price is the single most damaging thing this cache could do.
* Turns that depended on conversation context are never cached - the answer is
  specific to that thread, not to the question text.
* Guardrail-flagged turns, abstentions, and answers built from news or
  promotions (inherently time-sensitive) are never cached.

Cache entries are scoped by `corpus_version`, so re-ingesting the site
invalidates every cached answer without a delete pass. The net design: **read
broadly, write narrowly.**
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from qdrant_client import models

from app.agent.state import AgentState, trace_event
from app.config import settings
from app.logging_conf import get_logger
from app.retrieval.embedder import embed_query
from app.retrieval.qdrant_store import ensure_simple_collection, get_client

log = get_logger(__name__)

# Calibrated against 13 probes (8 that must hit, 5 that must miss) - see
# docs/faq-calibration.md. With each phrasing indexed as its own point, true
# hits score 0.755-0.987 and near-misses top out at 0.700, so 0.75 separates
# them perfectly. The initially-planned 0.90 matched nothing at all.
CURATED_THRESHOLD = 0.75

# Higher than the curated tier: these answers were generated, not human-vetted,
# and the key is the exact earlier question, so a genuine repeat scores very
# high. Being conservative here costs a cache miss; being loose serves the
# wrong answer.
CACHE_THRESHOLD = 0.88

CACHE_TTL_SECONDS = 24 * 3600

#: Content types whose answers go stale on their own schedule.
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


# ---------------------------------------------------------------------------
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

    if not hits or hits[0].score < CURATED_THRESHOLD:
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

    if not hits or hits[0].score < CACHE_THRESHOLD:
        return None

    payload = hits[0].payload or {}
    if time.time() - float(payload.get("created_at", 0)) > CACHE_TTL_SECONDS:
        return None

    return {
        "tier": "cache",
        "answer": payload.get("answer", ""),
        "citations": payload.get("citations", []),
        "score": float(hits[0].score),
        "original_query": payload.get("query", ""),
    }


# ---------------------------------------------------------------------------
def cache_eligibility(state: AgentState) -> tuple[bool, str]:
    """Decide whether this turn's answer may be cached. Returns (ok, reason)."""
    if not settings.enable_semantic_cache:
        return False, "cache disabled"

    route = state.get("route", "")
    if route in ("SQL", "HYBRID"):
        return False, "database-derived: fees, schedules and prices are mutable"
    if route in ("EMERGENCY", "REFUSE"):
        return False, "guardrail response"
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

    if not (state.get("answer") or "").strip():
        return False, "empty answer"

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


# ---------------------------------------------------------------------------
#: Naming any of these means the patient asked about something *specific*, and
#: a curated FAQ only ever holds general answers.
LIVE_DATA_ENTITIES = {"doctor", "lab_test", "package", "weekday", "specialty"}

#: Bare nouns for the *kinds* of record the database holds.
#:
#: The gazetteer only fires on a specific name, and the transactional lexicon
#: only on words like "fee" or "available", so a question can ask for a whole
#: table while triggering neither: "who are all the doctors in nawaloka?" was
#: answered from the hospital-overview FAQ at similarity 0.80 with a list of
#: specialist centres. "who are the doctors?" - the same question without the
#: hospital name - scored below threshold and routed correctly to SQL, which is
#: exactly the kind of coin-flip a similarity threshold cannot fix.
#:
#: Terms already covered by SQL_LEXICON (package, test, specialist, session)
#: are deliberately not repeated here; this list is only the gap.
LIVE_DATA_NOUNS = (
    "doctor", "doctors", "doctor's", "doctors'",
    "consultant", "consultants", "consultants'",
    "physician", "physicians", "surgeon", "surgeons",
)


def needs_live_data(query: str) -> list[str]:
    """Entity kinds in the query that make a canned FAQ answer unsafe.

    Similarity alone is not a sufficient guard, and this was measured twice:

    * "Which doctors are available on Tuesday?" scored 0.76 against the
      *booking* FAQ and was answered with the channeling hotline instead of the
      doctor list - fast, confident and wrong.
    * "What services does the Heart Centre offer?" scored 0.76 against the
      "which centres do you have" FAQ, returning a list of every centre rather
      than what that one does.

    Raising the threshold only trades these for missing genuine FAQs, because
    the score ranges overlap. Naming a specific doctor, test, package, weekday
    or specialty is instead treated as a deterministic signal to let the router
    decide - so the fast path is bypassed structurally, not by tuning a number.
    """
    from app.agent.signals import _contains_term, extract_signals

    signals = extract_signals(query)
    blocking = sorted(set(signals.entities) & LIVE_DATA_ENTITIES)

    text = " " + (query or "").lower().strip() + " "
    nouns = [n for n in LIVE_DATA_NOUNS if _contains_term(text, n)]
    if nouns:
        blocking.append(f"record type ({nouns[0]})")

    # Entity matching alone is not enough either. It misses plurals with no
    # specific name ("show me all your doctors and their fees") and text buried
    # inside a column rather than in a name ("which package includes a Pap
    # Smear" - "Pap Smear" is inside included_tests_and_services, so it is not
    # a gazetteer entry). Both were answered from a canned FAQ.
    #
    # Any transactional vocabulary - fee, price, schedule, available, package,
    # test - therefore also defers to the router. The cost is that a few
    # legitimately general questions ("how do I book a doctor") take the slow
    # path; the benefit is that no question about live data is ever answered
    # from a static string.
    if signals.sql_hits:
        blocking.append(f"transactional terms ({', '.join(sorted(set(signals.sql_hits))[:3])})")

    return blocking


async def faq_node(state: AgentState) -> dict:
    """Check both tiers. A hit short-circuits routing, retrieval and synthesis."""
    started = time.perf_counter()
    query = state.get("contextual_query") or state["user_query"]

    blocking = needs_live_data(query)

    # Someone describing an illness needs a clinic, not a stored string. The
    # guardrail runs before this node and already knows; not asking it meant
    # "my brother got cold who needs to contact?" was answered with the
    # ambulance number, which is both wrong and alarming.
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
