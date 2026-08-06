"""Seeds Qdrant collections on API startup if they're empty; idempotent.

Never blocks startup - seeding failures are logged and surfaced via /health.
Also runnable standalone: `python -m app.ingest.ensure_seeded`.
"""
# TODO: this seeds three collections now (web corpus, FAQ, symptom ontology) -
# the naming still reads like there's only one.

from __future__ import annotations

import sys
import time

from app.config import settings
from app.logging_conf import get_logger

log = get_logger(__name__)

# Re-exported from app/config.py for backward-compat imports.
WAIT_TIMEOUT = settings.seed_wait_timeout
POLL_SECONDS = settings.seed_poll_seconds


def _wait_for_postgres(deadline: float) -> bool:
    from app.tools.db import fetch

    while time.time() < deadline:
        try:
            fetch("SELECT 1")
            log.info("seed.postgres_ready")
            return True
        except Exception as exc:
            log.info("seed.waiting_for_postgres", error=str(exc)[:80])
            time.sleep(settings.seed_poll_seconds)
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
            time.sleep(settings.seed_poll_seconds)
    return False


def _points(collection: str) -> int:
    from app.retrieval.qdrant_store import get_client

    client = get_client()
    if not client.collection_exists(collection):
        return 0
    return client.get_collection(collection).points_count or 0


def ensure_seeded() -> None:
    if not settings.seed_on_startup:
        log.info("seed.disabled")
        return

    timeout = settings.seed_wait_timeout
    deadline = time.time() + timeout

    if not _wait_for_postgres(deadline):
        log.error("seed.postgres_unavailable", waited=timeout)
        return
    if not _wait_for_qdrant(deadline):
        log.error("seed.qdrant_unavailable", waited=timeout)
        return

    web = _points(settings.qdrant_collection)
    faq = _points(settings.qdrant_faq_collection)
    ontology = _points(settings.qdrant_ontology_collection)
    log.info("seed.current_state", web_points=web, faq_points=faq,
             ontology_points=ontology)

    if web == 0:
        log.info("seed.building_index", note="from committed HTML cache, no network")
        try:
            from app.ingest.pipeline import run

            stats = run(scrape=False)
            log.info("seed.index_built", **{k: v for k, v in stats.items() if k == "index"})
        # Catch SystemExit too: pipeline.run() raises it on a refusal, and
        # letting it through would take the API down on startup.
        except (Exception, SystemExit) as exc:
            log.error("seed.index_failed", error=str(exc)[:300])
    else:
        log.info("seed.index_present", points=web)

    # Point count alone can't detect deletions, so compare against FAQ_REVISION
    # instead; bumping it triggers a rebuild via build(recreate=True).
    try:
        from app.ingest.faq_seed import FAQ_REVISION, build, seeded_revision

        seeded = seeded_revision() if faq else None
        if seeded == FAQ_REVISION:
            log.info("seed.faq_present", points=faq, revision=seeded)
        else:
            log.info("seed.faq_rebuilding", points=faq, seeded=seeded,
                     wanted=FAQ_REVISION)
            log.info("seed.faq_built", entries=build(), revision=FAQ_REVISION)
    except Exception as exc:
        log.error("seed.faq_failed", error=str(exc)[:300])

    if ontology == 0:
        try:
            from app.retrieval.ontology import seed

            log.info("seed.ontology_built", points=seed())
        except Exception as exc:
            log.error("seed.ontology_failed", error=str(exc)[:300])
    else:
        log.info("seed.ontology_present", points=ontology)


if __name__ == "__main__":
    ensure_seeded()
    sys.exit(0)
