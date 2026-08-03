"""Curated FAQ pairs - the fast path for common patient questions.

Every answer here is grounded in the scraped corpus (`data/extracted/`) or in
the relational database. Nothing is written from general knowledge: a
hand-authored FAQ that quietly invents a visiting-hours policy is worse than no
FAQ at all, because it bypasses retrieval and is answered with full confidence.
Questions the corpus does not answer are deliberately absent.

`volatility` marks answers that can change. Anything `volatile` is served with
a "confirm by phone" line, and prices are never hard-coded here - price
questions are routed to the database instead, which is the system of record.

    python -m app.ingest.faq_seed          # build the collection
    python -m app.ingest.faq_seed --test   # check retrieval quality
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings
from app.logging_conf import get_logger

log = get_logger(__name__)

CONTACT_GENERAL = "0115 577 111"
CONTACT_HOTLINE = "0115 777 777"
CONTACT_CHANNELING = "011 577 7888"
CONTACT_ETU = "077 951 1929"
CONTACT_ADMISSIONS = "076 670 8383"
CONTACT_LAB = "076 163 4118"
CONTACT_PHARMACY = "077 022 9988"
EMAIL = "nawaloka@slt.lk"
ADDRESS = (
    "Nawaloka Hospitals PLC, 23, Deshamanya H. K. Dharmadasa Mawatha, "
    "Colombo 02, Sri Lanka"
)


@dataclass
class FaqEntry:
    id: str
    question: str
    answer: str
    paraphrases: list[str] = field(default_factory=list)
    volatility: str = "static"          # "static" | "volatile"
    source_urls: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def variants(self) -> list[str]:
        """Each phrasing indexed separately.

        Concatenating the question and its paraphrases into one embedding
        dilutes every individual phrasing - measured, "where are you located"
        then matched the wrong entry entirely. One point per phrasing keeps
        each one sharp, and they all carry the same answer payload.
        """
        return [self.question, *self.paraphrases]


FAQS: list[FaqEntry] = [
    FaqEntry(
        id="faq-emergency",
        question="What is the emergency or ambulance number?",
        paraphrases=[
            "ambulance number", "emergency contact", "how do I call an ambulance",
            "ETU number", "accident and emergency contact",
        ],
        answer=(
            f"For a life-threatening emergency call **1990 Suwa Seriya** (free "
            f"nationwide ambulance) or Nawaloka's ETU ambulance on "
            f"**{CONTACT_ETU}**.\n\n"
            f"Nawaloka's Accident & Emergency unit operates 24 hours. "
            f"Main hospital line: **{CONTACT_HOTLINE}**."
        ),
        source_urls=["https://www.nawaloka.com/quick-service-contacts"],
        tags=["emergency", "contact"],
    ),
    FaqEntry(
        id="faq-phone",
        question="What is the hospital's phone number?",
        paraphrases=[
            "contact number", "how do I call the hospital", "telephone",
            "general enquiries number", "hotline",
        ],
        answer=(
            f"- General line: **{CONTACT_GENERAL}**\n"
            f"- Hotline: **{CONTACT_HOTLINE}**\n"
            f"- Channeling appointments: **{CONTACT_CHANNELING}**\n"
            f"- Email: **{EMAIL}**"
        ),
        source_urls=["https://www.nawaloka.com/contactus"],
        tags=["contact"],
    ),
    FaqEntry(
        id="faq-address",
        question="Where is Nawaloka Hospital located?",
        paraphrases=[
            "hospital address", "how do I get there", "where are you",
            "location", "directions",
        ],
        answer=f"{ADDRESS}.\n\nGeneral line: **{CONTACT_GENERAL}**.",
        source_urls=["https://www.nawaloka.com/contactus"],
        tags=["contact", "location"],
    ),
    FaqEntry(
        id="faq-book-channeling",
        question="How do I book a channeling appointment?",
        paraphrases=[
            "how to channel a doctor", "book an appointment", "make a booking",
            "reserve a doctor appointment", "see a consultant",
        ],
        answer=(
            f"Call the channeling hotline on **{CONTACT_CHANNELING}** or "
            f"**{CONTACT_HOTLINE}**, or book through the website's channeling "
            f"page.\n\nIf you tell me the specialty or doctor you need, I can "
            f"give you their consultation fee and the days they channel."
        ),
        source_urls=["https://www.nawaloka.com/channeling"],
        tags=["channeling", "booking"],
    ),
    FaqEntry(
        id="faq-admissions",
        question="How do I arrange an admission?",
        paraphrases=["get admitted", "admission reservation", "book a room", "hospital admission"],
        answer=f"Call the admissions team on **{CONTACT_ADMISSIONS}**, or the main line on **{CONTACT_HOTLINE}**.",
        source_urls=["https://www.nawaloka.com/quick-service-contacts"],
        tags=["admissions"],
    ),
    FaqEntry(
        id="faq-lab-contact",
        question="How do I contact the laboratory?",
        paraphrases=["lab phone number", "laboratory contact", "call the lab"],
        answer=(
            f"The laboratory can be reached on **{CONTACT_LAB}**.\n\n"
            f"Ask me about a specific test and I can give you its price, "
            f"whether fasting is required, and how long the report takes."
        ),
        source_urls=["https://www.nawaloka.com/laboratory"],
        tags=["laboratory", "contact"],
    ),
    FaqEntry(
        id="faq-pharmacy",
        question="How do I contact the pharmacy?",
        paraphrases=["pharmacy number", "medicine counter", "buy medicines"],
        answer=f"The pharmacy can be reached on **{CONTACT_PHARMACY}**.",
        source_urls=["https://www.nawaloka.com/quick-service-contacts"],
        tags=["pharmacy", "contact"],
    ),
    FaqEntry(
        id="faq-emergency-hours",
        question="Is the emergency unit open 24 hours?",
        paraphrases=[
            "are you open at night", "24 hour service", "emergency opening hours",
            "is the hospital open now",
        ],
        answer=(
            f"Yes. Nawaloka's emergency team operates **24 hours a day**.\n\n"
            f"For an emergency call **1990** or the ETU ambulance on **{CONTACT_ETU}**."
        ),
        source_urls=["https://www.nawaloka.com/ambulance-emergency-patient-care"],
        tags=["emergency", "hours"],
    ),
    FaqEntry(
        id="faq-international",
        question="Do you treat international patients?",
        paraphrases=[
            "foreign patients", "medical tourism", "overseas patients",
            "international patient services",
        ],
        answer=(
            "Yes. Nawaloka has a dedicated international patient service that "
            "supports overseas patients throughout their treatment.\n\n"
            f"Contact **{CONTACT_HOTLINE}** or email **{EMAIL}** to arrange support."
        ),
        source_urls=["https://www.nawaloka.com/international"],
        tags=["international"],
    ),
    FaqEntry(
        id="faq-what-can-you-do",
        question="What can you help me with?",
        paraphrases=[
            "what do you do", "who are you", "how can you help", "your capabilities",
        ],
        answer=(
            "I can help you with:\n\n"
            "- **Doctors** - who practises here, their qualifications and consultation fees\n"
            "- **Channeling** - which days and times a consultant is available\n"
            "- **Lab tests** - prices, fasting requirements and report turnaround\n"
            "- **Health packages** - what each includes and what it costs\n"
            "- **The hospital** - our centres, services and facilities\n\n"
            "What would you like to know?"
        ),
        tags=["meta"],
    ),
    FaqEntry(
        id="faq-parking",
        question="Is there parking at the hospital?",
        paraphrases=["car park", "where do I park", "vehicle parking"],
        answer=(
            "Yes, vehicle parking is available at the hospital. For details on "
            f"access on the day, call **{CONTACT_GENERAL}**."
        ),
        volatility="volatile",
        source_urls=["https://www.nawaloka.com/contactus"],
        tags=["facilities"],
    ),
    FaqEntry(
        id="faq-lab-turnaround",
        question="How long do lab results take?",
        paraphrases=[
            "when will my report be ready", "test results time", "report delivery",
        ],
        answer=(
            "Most routine tests are reported the same day or within 24 hours, "
            "depending on the investigation.\n\n"
            "Tell me which test you need and I can give you its exact reporting "
            "time from our system."
        ),
        source_urls=["https://www.nawaloka.com/laboratory"],
        tags=["laboratory"],
    ),
    FaqEntry(
        id="faq-health-checks",
        question="What health check packages do you offer?",
        paraphrases=[
            "health screening", "full body checkup", "annual checkup",
            "wellness package", "medical checkup",
        ],
        answer=(
            "We offer preventive health screening packages for men, women, "
            "pre-employment and pre-marital checks, and comprehensive cancer "
            "screening.\n\n"
            "Ask me about a specific package and I'll give you its exact price "
            "and everything it includes."
        ),
        source_urls=["https://www.nawaloka.com/health-checkups"],
        tags=["packages"],
    ),
    FaqEntry(
        id="faq-blood-bank",
        question="Do you have a blood bank?",
        paraphrases=["donate blood", "blood donation", "blood transfusion service"],
        answer=(
            "Yes, Nawaloka operates a blood bank.\n\n"
            f"For donation or transfusion enquiries call **{CONTACT_GENERAL}**."
        ),
        source_urls=["https://www.nawaloka.com/blood-bank"],
        tags=["services"],
    ),
    FaqEntry(
        id="faq-centres",
        question="What specialist centres does the hospital have?",
        paraphrases=[
            "centres of excellence", "what departments do you have",
            "specialist units", "list of centres",
        ],
        answer=(
            "Nawaloka's centres of excellence include the Heart Centre, Neuro "
            "Centre, Cancer & Blood Disease Centre, Bone Marrow Transplant Unit, "
            "Fertility Centre, Mother & Baby unit, ENT, Dental, Eye Care and "
            "LASIK, Radiology & Imaging, Diabetes Centre, Rheumatology & Sports "
            "Medicine, Skin Care & Cosmetic Centre, and the Elite VIP and "
            "Corporate centres.\n\n"
            "Ask about any of them and I'll tell you what they offer."
        ),
        source_urls=["https://www.nawaloka.com/"],
        tags=["centres"],
    ),
]


# ---------------------------------------------------------------------------
def build(recreate: bool = True) -> int:
    from qdrant_client import models

    from app.retrieval.embedder import embed_passages
    from app.retrieval.qdrant_store import ensure_simple_collection, get_client

    client = get_client()
    name = settings.qdrant_faq_collection

    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    ensure_simple_collection(client, name)

    # One point per phrasing; several points share an answer.
    pairs: list[tuple[FaqEntry, str]] = [
        (f, variant) for f in FAQS for variant in f.variants()
    ]
    vectors = embed_passages([variant for _, variant in pairs])

    points = [
        models.PointStruct(
            id=i,
            vector=vec,
            payload={
                "faq_id": f.id,
                "matched_phrasing": variant,
                "question": f.question,
                "answer": f.answer,
                "volatility": f.volatility,
                "source_urls": f.source_urls,
                "tags": f.tags,
            },
        )
        for i, ((f, variant), vec) in enumerate(zip(pairs, vectors))
    ]
    client.upsert(collection_name=name, points=points, wait=True)
    log.info("faq.seeded", collection=name, entries=len(FAQS), points=len(points))
    return len(FAQS)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="probe retrieval quality")
    args = parser.parse_args()

    n = build()
    print(f"Seeded {n} curated FAQ entries into '{settings.qdrant_faq_collection}'")

    if args.test:
        from app.agent.nodes.faq import faq_lookup

        probes = [
            ("what's the ambulance number", True),
            ("how do I book a doctor", True),
            ("where are you located", True),
            ("do you take foreign patients", True),
            ("is the ETU open at 3am", True),
            # must MISS - these need live data, not a canned answer
            ("what is Dr. Maiya Gunasekara's fee", False),
            ("which doctors work on Tuesday", False),
            ("how much is the dengue test", False),
            ("what is the capital of France", False),
        ]
        print(f"\n{'query':<44}{'hit':>6}{'score':>8}  entry")
        print("-" * 78)
        correct = 0
        for query, should_hit in probes:
            hit = faq_lookup(query)
            got = hit is not None
            correct += int(got == should_hit)
            mark = "OK " if got == should_hit else "BAD"
            print(f"{mark} {query[:40]:<42}{str(got):>6}"
                  f"{(hit['score'] if hit else 0):>8.3f}  "
                  f"{hit['faq_id'] if hit else '-'}")
        print(f"\n{correct}/{len(probes)} probes behaved correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
