"""Routes a query: guardrail short-circuit -> LLM classifier -> reconciliation
against deterministic signals (which can only widen the route, never narrow it).
"""


from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, get_args

from pydantic import BaseModel, Field

from app.agent import guardrail
from app.agent.signals import RouteSignals, extract_signals
from app.config import settings
from app.llm.base import Msg
from app.llm.router import get_router
from app.logging_conf import get_logger
from app.prompts.templates import ROUTER, render

log = get_logger(__name__)

#: The two the guardrail decides on its own; the classifier never sees them.
GUARDRAIL_ROUTES = ("EMERGENCY", "REFUSE")
_WIDENABLE = ("SQL", "VECTOR")


def classifier_routes() -> tuple[str, ...]:
    """Every route the classifier can return, derived from the schema."""
    return get_args(RouteLLMOutput.model_fields["route"].annotation)


def all_routes() -> tuple[str, ...]:
    """Every route that can reach the graph, whoever decided it."""
    return (*classifier_routes(), *GUARDRAIL_ROUTES)


# Pydantic model for strict LLM output validation
class RouteLLMOutput(BaseModel):
    route: Literal["SQL", "VECTOR", "HYBRID", "SMALLTALK", "OUT_OF_SCOPE"] = Field(
        description="The determined route for the query."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    rationale: str = Field(description="Short explanation for the route")
    sql_intent: str = Field(default="", description="Extracted intent for SQL search if applicable")
    vector_intent: str = Field(default="", description="Extracted intent for vector search if applicable")

@dataclass
class RouteDecision:
    route: str
    confidence: float = 0.0
    rationale: str = ""
    sql_intent: str = ""
    vector_intent: str = ""
    signals: RouteSignals = field(default_factory=RouteSignals)
    llm_route: str = ""
    overridden: bool = False
    override_reason: str = ""
    latency_ms: int = 0
    provider: str = ""
    error: str = ""
    guardrail: guardrail.GuardrailResult = field(default_factory=guardrail.GuardrailResult)

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale,
            "sql_intent": self.sql_intent,
            "vector_intent": self.vector_intent,
            "llm_route": self.llm_route,
            "overridden": self.overridden,
            "override_reason": self.override_reason,
            "signals": self.signals.to_dict(),
            "guardrail": self.guardrail.to_dict(),
            "latency_ms": self.latency_ms,
            "provider": self.provider,
            "error": self.error,
        }

def _history_block(history: list[list[str]] | None) -> str:
    if not history:
        return ""
    lines = []
    for turn in history[-settings.router_history_turns:]:
        if len(turn) == 2:
            answer = turn[1][: settings.router_history_answer_chars]
            lines.append(f"Patient: {turn[0]}\nAssistant: {answer}")
    return "\n\n".join(lines)

def _build_messages(query: str, signals: RouteSignals,
                    history: list[list[str]] | None) -> list[Msg]:
    parts = []
    prior = _history_block(history)
    if prior:
        parts.append(f"Recent conversation (for context only):\n{prior}")
    parts.append(f"Deterministic signal analysis:\n{signals.summary()}")
    parts.append(f"Question to classify:\n{query}")
    return render(ROUTER, input="\n\n".join(parts))

@dataclass(frozen=True)
class Widening:
    """A rule's verdict: which route to widen to, and why."""

    to: str
    reason: str

def _rule_no_structured_subject(d: RouteDecision) -> Widening | None:
    """SQL chosen but the query names no DB record/attribute: widen to keep the vector branch in play."""
    if d.route != "SQL" or not d.signals.gazetteer_ready:
        return None
    if d.signals.has_structured_record:
        return None
    to = "HYBRID" if d.signals.has_vector else "VECTOR"
    return Widening(
        to,
        "names no known database record and no database attribute, so a SQL "
        f"lookup has nothing to match; widened to {to}",
    )

def _rule_both_signals(d: RouteDecision) -> Widening | None:
    """Transactional and descriptive parts both present, and the model hedged."""
    if d.route in _WIDENABLE and d.signals.both and (
        d.confidence < settings.router_high_confidence
    ):
        return Widening(
            "HYBRID",
            f"confidence {d.confidence:.2f} with both transactional and "
            "descriptive signals present; widened to HYBRID",
        )
    return None

def _rule_low_confidence(d: RouteDecision) -> Widening | None:
    """The model was guessing. Ask both sources rather than back its guess."""
    if d.route in _WIDENABLE and d.confidence < settings.router_min_confidence:
        return Widening(
            "HYBRID",
            f"confidence {d.confidence:.2f} below "
            f"{settings.router_min_confidence}; widened to HYBRID",
        )
    return None

#: Ordered, first match wins.
_RULES = (_rule_no_structured_subject, _rule_both_signals, _rule_low_confidence)

def _reconcile(decision: RouteDecision) -> RouteDecision:
    # Whether a question is in scope at all is not something keyword signals get
    # a vote on.
    if decision.route in ("SMALLTALK", "OUT_OF_SCOPE"):
        return decision

    for rule in _RULES:
        widening = rule(decision)
        if widening is None:
            continue
        decision.overridden = True
        decision.override_reason = widening.reason
        decision.route = widening.to
        break

    if decision.route == "HYBRID":
        # Both branches need something to search for; the classifier only fills
        # in the half it thought was needed.
        if not decision.sql_intent:
            decision.sql_intent = decision.rationale or ""
        if not decision.vector_intent:
            decision.vector_intent = decision.rationale or ""
    return decision

#: Ordered, first match wins; the last entry is the unconditional default.
_FALLBACK: tuple[tuple[Callable[[RouteSignals], bool], str, str], ...] = (
    (lambda s: s.both, "HYBRID", "both signal families present"),
    (lambda s: s.has_sql, "SQL", "transactional signals only"),
    (lambda s: s.has_vector, "VECTOR", "descriptive signals only"),
    (lambda s: True, "VECTOR", "no signals; defaulting to knowledge base"),
)

def _fallback_route(signals: RouteSignals) -> tuple[str, str]:
    """Route on signals alone, for when the classifier is unavailable."""
    for predicate, route, why in _FALLBACK:
        if predicate(signals):
            return route, f"LLM unavailable; {why}"
    return "VECTOR", "LLM unavailable; defaulting to knowledge base"

async def route_query(
    query: str,
    history: list[list[str]] | None = None,
    use_llm: bool = True,
) -> RouteDecision:
    started = time.perf_counter()

    verdict = guardrail.check(query)
    if verdict.short_circuits:
        decision = RouteDecision(
            route=verdict.verdict,
            confidence=1.0,
            rationale=f"guardrail: {verdict.reason} (matched {verdict.matched!r})",
            guardrail=verdict,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        log.info("router.guardrail_intercept", route=decision.route,
                 reason=verdict.reason)
        return decision

    signals = extract_signals(query)
    decision = RouteDecision(route="", signals=signals, guardrail=verdict)

    if not use_llm:
        decision.route, decision.rationale = _fallback_route(signals)
        decision.confidence = settings.router_fallback_confidence
        decision.latency_ms = int((time.perf_counter() - started) * 1000)
        return decision

    try:
        # Schema constrains generation; model_validate catches a hallucinated route name.
        schema = RouteLLMOutput.model_json_schema()
        data, response = await get_router().complete_json(
            _build_messages(query, signals, history),
            schema,
            tier="fast",
            max_tokens=settings.router_max_tokens,
        )

        # Validate output using Pydantic (Throws error if LLM hallucinates route name)
        parsed_data = RouteLLMOutput.model_validate(data)

        decision.route = parsed_data.route
        decision.llm_route = parsed_data.route
        decision.confidence = parsed_data.confidence
        decision.rationale = parsed_data.rationale[: settings.router_rationale_max_chars]
        decision.sql_intent = parsed_data.sql_intent
        decision.vector_intent = parsed_data.vector_intent
        decision.provider = response.provider

    except Exception as exc:
        decision.error = f"{type(exc).__name__}: {exc}"
        # Graceful Fallback on strict parsing failure
        decision.route, decision.rationale = _fallback_route(signals)
        decision.confidence = settings.router_error_fallback_confidence
        decision.overridden = True
        decision.override_reason = "LLM routing failed or hallucinated; used deterministic signals"
        log.warning("router.llm_failed", error=str(exc)[:200], fallback=decision.route)
        decision.latency_ms = int((time.perf_counter() - started) * 1000)
        return decision

    decision = _reconcile(decision)
    decision.latency_ms = int((time.perf_counter() - started) * 1000)

    log.info("router.decided", route=decision.route, llm_route=decision.llm_route,
             confidence=round(decision.confidence, 2), overridden=decision.overridden,
             provider=decision.provider, ms=decision.latency_ms)
    return decision