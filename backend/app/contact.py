"""Hospital contact details, defined once and imported everywhere (not looked
up from the vector store, so the emergency guardrail never needs a model call).
tests/test_contact.py checks these still match the committed scrape.
"""

from __future__ import annotations

# Main hospital line. Present in 40 of the 41 scraped pages.
HOTLINE = "0115 777 777"

#: Nawaloka's own ETU ambulance.
ETU_AMBULANCE = "077 951 1929"

# The free national ambulance service - not a Nawaloka number, so it is not in the scrape and the drift test deliberately does not cover it.
NATIONAL_AMBULANCE = "1990 Suwa Seriya"
