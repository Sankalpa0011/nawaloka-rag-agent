"""Safety checks that run before routing.

Deterministic (regex-based) rather than model-based, so it's fast and
testable. EMERGENCY short-circuits the graph to a fixed response with the
ambulance number. REFUSE covers diagnosis, prescriptions and result
interpretation, and points the user at a consultation instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

GuardrailVerdict = Literal["EMERGENCY", "REFUSE", "SYMPTOM", "CLEAR"]

from app.contact import ETU_AMBULANCE, HOTLINE, NATIONAL_AMBULANCE

# local aliases for the emergency text below
AMBULANCE = NATIONAL_AMBULANCE
HOSPITAL_EMERGENCY = HOTLINE

# Emergency patterns - deliberately specific so navigational questions
EMERGENCY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bcan'?t\s+breathe|\bnot\s+breathing|\bstopped\s+breathing", "breathing difficulty"),
    (r"\bchest\s+pain\b.{0,40}\b(breath|breathe|sweat|arm|jaw|crush)", "cardiac symptoms"),
    (r"\bcrushing\s+chest\b|\bheart\s+attack\b", "suspected heart attack"),
    (r"\bunconscious\b|\bcollapsed\b|\bpassed\s+out\b|\bnot\s+responding\b", "unresponsive patient"),
    # "stroke" alone isn't enough - require an acute framing or a hallmark symptom
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

# Clinical advice we must not give.
REFUSE_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (
        r"\bwhat\s+(disease|illness|condition)\s+do\s+i\s+have\b"
        r"|\bwhat'?s\s+wrong\s+with\s+me\b"
        r"|\bdo\s+i\s+have\s+(cancer|diabetes|covid|a\s+tumou?r)\b"
        r"|\bdiagnos(e|is)\s+me\b"
        # also catch third-person/hypothetical phrasings of the same request
        r"|\b(possible|likely|differential)\s+diagnos(is|es)\b"
        r"|\bwhat\s+(disease|illness|condition)s?\s+(would|could|might|does)\b"
        r"|\bwhat\s+causes?\s+(my|these|those)\s+\w+",
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
    (
        r"\b(book|cancel|reschedule|make)\s+(an\s+)?(appointment|channeling|booking)\b"
        r"|\bbook\s+(a\s+)?(doctor|session)\b",
        "appointment action",
        "I can only provide information about channeling schedules and fees. I cannot make, cancel, or reschedule bookings for you.",
    ),
)

# Over-triggering guard: suppress an emergency match when the query reads as
SERVICE_QUERY = re.compile(
    r"\b(do you (have|offer|provide|treat|do)|is there (a|an)|what (is|are|does)"
    r"|tell me about|which (doctor|specialist|clinic|centre|center)|who treats"
    r"|how much|what causes|cost of|price of"
    r"|rehabilitation|programme|program|clinic|unit|ward|centre|center"
    r"|package|service|therapy|screening|policy)\b",
    re.I,
)

# someone describing something happening now - a person plus a present or
ACUTE_FRAMING = re.compile(
    r"\b(i|my|he|she|we|they|someone|somebody)\b[^.?!]{0,40}"
    r"\b(am|is|are|'m|'s|was|were|has|have|had|having|feel|feeling|think|"
    r"cannot|can'?t|just)\b"
    r"|\b(help|emergency|urgent|right now|immediately)\b",
    re.I,
)


def _is_service_question(text: str) -> bool:
    return bool(SERVICE_QUERY.search(text)) and not ACUTE_FRAMING.search(text)


# symptom mentioned but not an emergency - reply gets a "not a diagnosis" note
SYMPTOM_HINT = re.compile(
    r"\b(pain|ache|aching|fever|cough|rash|swelling|dizzy|dizziness|nausea|"
    r"vomit|bleeding|lump|tired|fatigue|breathless|sore|infection|injur|"
    r"cold|flu\b|influenza|headache|migraine|allerg|itch|burn|sprain|fracture|"
    # "numbness" not "numb" - bare "numb" would also match "number"
    r"diarrh|constipat|heartburn|wheez|numbness|weakness|blurred|stomach)",
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

    # graph has to answer straight away, no retrieval or SQL
    @property
    def short_circuits(self) -> bool:
        return self.verdict in ("EMERGENCY", "REFUSE")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "disclaimer_required": self.disclaimer_required,
            "flags": self.flags,
        }


EMERGENCY_RESPONSE = f"""**This may be a medical emergency. Please get help now.**

- **Ambulance: {AMBULANCE}** - free nationwide emergency ambulance service
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


# classify a query before it gets anywhere near the router
def check(query: str) -> GuardrailResult:
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
