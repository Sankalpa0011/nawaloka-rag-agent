"""Pre-routing safety checks for a healthcare assistant.

Deliberately **deterministic**. Two reasons:

1. Latency and reliability. Someone typing "my father collapsed" must not wait
   on a rate-limited model, and must not depend on that model choosing to be
   careful rather than helpful.
2. Testability. A regex phrase list can be asserted against in CI; a prompt
   instruction cannot.

Two categories are intercepted before routing:

**EMERGENCY** - short-circuits the entire graph to a fixed response carrying the
Sri Lankan emergency ambulance number (1990 Suwa Seriya).

**REFUSE** - requests for diagnosis, prescriptions or interpretation of the
user's own results. These are declined and redirected to a consultation. Note
what is *not* refused: naming the specialty that normally treats a complaint is
navigation, and is the single most useful thing this assistant can do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

GuardrailVerdict = Literal["EMERGENCY", "REFUSE", "SYMPTOM", "CLEAR"]

AMBULANCE = "1990 Suwa Seriya"
HOSPITAL_EMERGENCY = "0115 777 777"
#: Nawaloka's own emergency ambulance, taken from the scraped
#: /quick-service-contacts page rather than assumed.
ETU_AMBULANCE = "077 951 1929"

# --------------------------------------------------------------------------
# Emergency. Patterns are intentionally specific: "chest pain" alone is a
# routine navigational query ("who treats chest pain?"), but chest pain with
# breathing difficulty is not.
# --------------------------------------------------------------------------
EMERGENCY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bcan'?t\s+breathe|\bnot\s+breathing|\bstopped\s+breathing", "breathing difficulty"),
    (r"\bchest\s+pain\b.{0,40}\b(breath|breathe|sweat|arm|jaw|crush)", "cardiac symptoms"),
    (r"\bcrushing\s+chest\b|\bheart\s+attack\b", "suspected heart attack"),
    (r"\bunconscious\b|\bcollapsed\b|\bpassed\s+out\b|\bnot\s+responding\b", "unresponsive patient"),
    # "stroke" alone is not enough - "do you have a stroke rehabilitation
    # programme" is a service question, not an emergency. Require an acute
    # framing or a hallmark symptom.
    (r"\b(having|had|suffered|suffering)\s+(a\s+)?stroke\b"
     r"|\bstroke\s+(right\s+)?now\b|\bface\s+is\s+drooping\b|\bslurred\s+speech\b",
     "suspected stroke"),
    (r"\bsevere(ly)?\s+bleed|\bbleeding\s+(heavily|badly|severely)|\bhaemorrhag|\bhemorrhag", "severe bleeding"),
    (r"\boverdose\b|\bpoison(ed|ing)\b", "overdose or poisoning"),
    (r"\bsuicid|\bkill\s+myself\b|\bend\s+my\s+life\b", "self-harm risk"),
    (r"\b(having|had|is\s+in)\s+(a\s+)?seizure\b|\bseizure\s+(right\s+)?now\b|\bconvulsing\b",
     "seizure"),
    (r"\bchoking\b|\banaphyla", "airway emergency"),
    (r"\bsevere\s+burn|\bmajor\s+accident\b|\bhit\s+by\s+a\s+(car|bus|vehicle)", "major trauma"),
)

# --------------------------------------------------------------------------
# Clinical advice we must not give.
# --------------------------------------------------------------------------
REFUSE_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (
        r"\bwhat\s+(disease|illness|condition)\s+do\s+i\s+have\b"
        r"|\bwhat'?s\s+wrong\s+with\s+me\b"
        r"|\bdo\s+i\s+have\s+(cancer|diabetes|covid|a\s+tumou?r)\b"
        r"|\bdiagnos(e|is)\s+me\b",
        "diagnosis",
        "I can't diagnose conditions.",
    ),
    (
        r"\b(what\s+)?dos(e|age)\b|\bhow\s+(much|many)\s+(mg|ml|tablets?|pills?)\b"
        r"|\bshould\s+i\s+take\s+\w+\s+(for|to)\b|\bprescri(be|ption)\s+me\b",
        "prescription",
        "I can't advise on medicines or dosages.",
    ),
    (
        r"\b(my|the)\s+(test|blood|lab|report|result)s?\b.{0,50}\b(mean|bad|normal|ok|high|low|worry)"
        r"|\bis\s+(that|this|it)\s+(bad|normal|serious|dangerous)\b"
        r"|\btsh\s+of\b|\bmy\s+\w+\s+level\s+is\b",
        "result interpretation",
        "I can't interpret your test results.",
    ),
    (
        r"\b(stop|quit|skip|double|increase|reduce)\s+(taking\s+)?(my\s+)?(medication|medicine|tablets?|pills?|dose)\b"
        r"|\bcan\s+i\s+stop\s+taking\b",
        "medication change",
        "I can't advise on changing medication.",
    ),
)

# --------------------------------------------------------------------------
# Over-triggering guard.
#
# Emergency patterns necessarily contain alarming words, and a hospital site is
# full of pages about exactly those words. "Do you have a stroke rehabilitation
# programme" is a service question; answering it with an ambulance number makes
# the assistant useless for its actual job. A match is therefore suppressed
# when the query reads as a question *about hospital services* and contains no
# sign that something is happening to someone right now.
# --------------------------------------------------------------------------
SERVICE_QUERY = re.compile(
    r"\b(do you (have|offer|provide|treat|do)|is there (a|an)|what (is|are|does)"
    r"|tell me about|which (doctor|specialist|clinic|centre|center)|who treats"
    r"|how much|what causes|cost of|price of"
    r"|rehabilitation|programme|program|clinic|unit|ward|centre|center"
    r"|package|service|therapy|screening|policy)\b",
    re.I,
)

# Someone describing a live situation: first/third-person plus a present-tense
# or recent-past verb.
ACUTE_FRAMING = re.compile(
    r"\b(i|my|he|she|we|they|someone|somebody)\b[^.?!]{0,40}"
    r"\b(am|is|are|'m|'s|was|were|has|have|had|having|feel|feeling|think|"
    r"cannot|can'?t|just)\b"
    r"|\b(help|emergency|urgent|right now|immediately)\b",
    re.I,
)


def _is_service_question(text: str) -> bool:
    return bool(SERVICE_QUERY.search(text)) and not ACUTE_FRAMING.search(text)


# Mentions a symptom without being an emergency - allowed, but the answer gets
# a "this is not a diagnosis" framing.
SYMPTOM_HINT = re.compile(
    r"\b(pain|ache|aching|fever|cough|rash|swelling|dizzy|dizziness|nausea|"
    r"vomit|bleeding|lump|tired|fatigue|breathless|sore|infection|injur)",
    re.I,
)


@dataclass
class GuardrailResult:
    verdict: GuardrailVerdict = "CLEAR"
    reason: str = ""
    matched: str = ""
    disclaimer_required: bool = False
    response: str = ""
    flags: list[str] = field(default_factory=list)

    @property
    def short_circuits(self) -> bool:
        """True when the graph must answer immediately without retrieval or SQL."""
        return self.verdict in ("EMERGENCY", "REFUSE")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "disclaimer_required": self.disclaimer_required,
            "flags": self.flags,
        }


EMERGENCY_RESPONSE = f"""**This may be a medical emergency. Please get help now.**

- **Ambulance: {AMBULANCE}** — free nationwide emergency ambulance service
- **Nawaloka ETU ambulance: {ETU_AMBULANCE}**
- **Nawaloka Hospital: {HOSPITAL_EMERGENCY}**
- Or go to your nearest emergency department immediately

Nawaloka Hospitals' Accident & Emergency unit operates 24 hours.

I'm an information assistant and cannot help with an emergency. Please contact \
emergency services now rather than continuing this conversation."""


def _refusal_response(explanation: str, category: str) -> str:
    return f"""{explanation} That needs a qualified doctor who can examine you and see your history.

What I can do is help you reach one:

- Tell me your symptom area and I'll point you to the right specialty
- I can give you consultation fees and channeling times for our consultants
- To book, call **{HOSPITAL_EMERGENCY}**

If this is urgent, call **{AMBULANCE}** or go to the nearest emergency department."""


def check(query: str) -> GuardrailResult:
    """Classify a query before it reaches routing."""
    text = (query or "").lower().strip()
    if not text:
        return GuardrailResult()

    service_question = _is_service_question(text)

    for pattern, reason in EMERGENCY_PATTERNS:
        m = re.search(pattern, text, re.I)
        if m:
            if service_question:
                # Alarming word, but the sentence is asking about a service.
                continue
            return GuardrailResult(
                verdict="EMERGENCY",
                reason=reason,
                matched=m.group(0),
                response=EMERGENCY_RESPONSE,
                flags=["emergency", reason],
            )

    for pattern, category, explanation in REFUSE_PATTERNS:
        m = re.search(pattern, text, re.I)
        if m:
            return GuardrailResult(
                verdict="REFUSE",
                reason=category,
                matched=m.group(0),
                response=_refusal_response(explanation, category),
                disclaimer_required=True,
                flags=["refuse", category],
            )

    if SYMPTOM_HINT.search(text):
        return GuardrailResult(
            verdict="SYMPTOM",
            reason="symptom mentioned",
            disclaimer_required=True,
            flags=["symptom_context"],
        )

    return GuardrailResult()


MEDICAL_DISCLAIMER = (
    "_This is general information to help you find the right care, not medical "
    "advice or a diagnosis. Please consult a doctor about your symptoms._"
)
