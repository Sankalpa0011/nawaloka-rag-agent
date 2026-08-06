"""Curated FAQ pairs (phone numbers, addresses, hours) - fast path for transactional questions.

    python -m app.ingest.faq_seed          # build the collection
    python -m app.ingest.faq_seed --test   # check retrieval quality
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings
from app.contact import HOTLINE as CONTACT_HOTLINE
from app.logging_conf import get_logger

log = get_logger(__name__)

#Bump on every change to FAQS; ensure_seeded compares this to decide whether
# to rebuild the collection (so deleted entries don't survive forever).
FAQ_REVISION = 3

CONTACT_GENERAL = "0115 577 111"
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
        """Each phrasing indexed separately, all sharing the same answer payload."""
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
        # Phrased without the hospital's name on purpose - naming it made this
        # entry match unrelated "about Nawaloka" questions too broadly.
        id="faq-address",
        question="Where is the hospital located?",
        paraphrases=[
            "hospital address", "how do I get there", "where are you",
            "location", "directions", "how do I find the hospital",
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
]

#: Every tag a curated entry is allowed to carry; enforced by test to keep scope narrow.
TRANSACTIONAL_TAGS = frozenset({
    "contact", "emergency", "hours", "location", "booking", "admissions",
    "laboratory", "pharmacy", "channeling",
})


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
                "faq_revision": FAQ_REVISION,
            },
        )
        for i, ((f, variant), vec) in enumerate(zip(pairs, vectors))
    ]
    client.upsert(collection_name=name, points=points, wait=True)
    log.info("faq.seeded", collection=name, entries=len(FAQS), points=len(points),
             revision=FAQ_REVISION)
    return len(FAQS)


def seeded_revision() -> int | None:
    """FAQ_REVISION a live collection was built from, or None (meaning "rebuild")."""
    from app.retrieval.qdrant_store import get_client

    client = get_client()
    name = settings.qdrant_faq_collection
    if not client.collection_exists(name):
        return None
    points, _ = client.scroll(name, limit=1, with_payload=True, with_vectors=False)
    if not points:
        return None
    revision = (points[0].payload or {}).get("faq_revision")
    return revision if isinstance(revision, int) else None


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
            ("is the ETU open at 3am", True),
            ("what is the pharmacy number", True),
            # must MISS - these need live data, not a canned answer
            ("what is Dr. Maiya Gunasekara's fee", False),
            ("which doctors work on Tuesday", False),
            ("how much is the dengue test", False),
            # must MISS - descriptive questions belong to the vector store now.
            # Each of these used to hit a curated entry and get a brochure.
            ("do you take foreign patients", False),
            ("tell me about nawaloka hospitals", False),
            ("what specialist centres do you have", False),
            ("what services does the heart centre offer", False),
            ("what dental treatments do you offer", False),
            ("do you have a blood bank", False),
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
