"""Make the service self-seeding.

Postgres initialises itself from db/init on first start, but Qdrant does not -
it comes up empty. Without this, `docker compose up` produces a system that
answers database questions correctly and silently abstains on every question
that needs the website knowledge base, which is a confusing way to fail.

`ensure_seeded()` is called from the API's startup lifespan: it waits for both
stores, checks whether the vector collections already hold data, and builds them
from the committed HTML cache if not. It is idempotent, so a restart costs one
cheap count query.

Seeding never blocks startup. If it fails, the API still comes up and answers
what it can from Postgres, and /health reports the vector store as missing -
which is far more debuggable than an API that refuses to start.

It also runs standalone (`python -m app.ingest.ensure_seeded`) for seeding
against an API that is already up, or when SEED_ON_STARTUP is turned off.
"""

from __future__ import annotations

import os
import sys
import time

from app.config import settings
from app.logging_conf import get_logger

log = get_logger(__name__)

WAIT_TIMEOUT = float(os.getenv("SEED_WAIT_TIMEOUT", "120"))
POLL_SECONDS = 2.0


def _wait_for_postgres(deadline: float) -> bool:
    from app.tools.db import fetch

    while time.time() < deadline:
        try:
            fetch("SELECT 1")
            log.info("seed.postgres_ready")
            return True
        except Exception as exc:
            log.info("seed.waiting_for_postgres", error=str(exc)[:80])
            time.sleep(POLL_SECONDS)
    return False


def _wait_for_qdrant(deadline: float) -> bool:
    from app.retrieval.qdrant_store import get_client

    while time.time() < deadline:
        try:
            get_client().get_collections()
            log.info("seed.qdrant_ready")
            return True
        except Exception as exc:
            log.info("seed.waiting_for_qdrant", error=str(exc)[:80])
            time.sleep(POLL_SECONDS)
    return False


def _points(collection: str) -> int:
    from app.retrieval.qdrant_store import get_client

    client = get_client()
    if not client.collection_exists(collection):
        return 0
    return client.get_collection(collection).points_count or 0


def ensure_seeded() -> None:
    if os.getenv("SEED_ON_STARTUP", "true").lower() in ("0", "false", "no"):
        log.info("seed.disabled")
        return

    deadline = time.time() + WAIT_TIMEOUT

    if not _wait_for_postgres(deadline):
        log.error("seed.postgres_unavailable", waited=WAIT_TIMEOUT)
        return  # start anyway; /health will show it
    if not _wait_for_qdrant(deadline):
        log.error("seed.qdrant_unavailable", waited=WAIT_TIMEOUT)
        return

    web = _points(settings.qdrant_collection)
    faq = _points(settings.qdrant_faq_collection)
    log.info("seed.current_state", web_points=web, faq_points=faq)

    if web == 0:
        log.info("seed.building_index", note="from committed HTML cache, no network")
        try:
            from app.ingest.pipeline import run

            stats = run(scrape=False)
            log.info("seed.index_built", **{k: v for k, v in stats.items() if k == "index"})
        except Exception as exc:
            log.error("seed.index_failed", error=str(exc)[:300])
    else:
        log.info("seed.index_present", points=web)

    if faq == 0:
        try:
            from app.ingest.faq_seed import build

            log.info("seed.faq_built", entries=build())
        except Exception as exc:
            log.error("seed.faq_failed", error=str(exc)[:300])
    else:
        log.info("seed.faq_present", points=faq)


if __name__ == "__main__":
    ensure_seeded()
    sys.exit(0)
