"""Playwright scraper for www.nawaloka.com (a hydration-only React SPA).

Writes rendered HTML to backend/data/raw/<slug>.html plus a manifest.json,
committed so the index can be rebuilt offline with `--from-cache`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright

from app.config import RAW_HTML_DIR, settings
from app.logging_conf import get_logger

log = get_logger(__name__)

# Mirrors robots.txt. Query-string URLs are disallowed by `Disallow: /*?*`.
DISALLOWED_PREFIXES = ("/node_modules/", "/ajax/", "/embed/", "/cgi-bin/")

ORIGIN = "https://www.nawaloka.com"

# Sitemap reconciliation: swap known-stale URLs and add missing routes.
STALE_SITEMAP_REPLACEMENTS: dict[str, tuple[str, ...]] = {
    "/healthchecks": ("/health-checkups",),
    "/elite-centre": ("/elite-corporate-centre", "/elite-vip-centre"),
    "/news": ("/blogs-and-news",),
}

# Content-bearing routes the sitemap omits entirely.
SUPPLEMENTAL_PATHS: tuple[str, ...] = (
    "/aboutus/awards",
    "/aboutus/message",
    "/aboutus/reports",
    "/aboutus/technology",
    # "/aboutus/rooms/" is intentionally absent: it is an index that renders
    # nothing without a slug (the real content lives at /hospital-rooms/:slug).
    "/ambulance-emergency-patient-care",
    "/ambulance-terms-and-conditions",
    "/bone-marrow-transplant-unit",
    "/lab-tests-online",
    "/lasik-advanced-eye-care-centre",
    "/quick-service-contacts",
)

EXCLUDED_PREFIXES: tuple[str, ...] = (
    "/elogin",
    "/payments",
    "/booksession",
    "/bookingsummery",
    "/room-booking",
    "/service-bookings",
    "/external-view",
    "/doctorprofile",
    "/doctorsearch",
    "/healthchecks/",  # cart / checkout / success / failure steps
)


def reconcile_urls(sitemap_urls: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Apply stale-URL replacements and add supplemental routes; returns (urls, changes)."""
    changes: dict[str, list[str]] = {"replaced": [], "added": [], "excluded": []}
    out: list[str] = []
    seen: set[str] = set()

    def add(u: str) -> None:
        if u not in seen:
            seen.add(u)
            out.append(u)

    for url in sitemap_urls:
        path = urlparse(url).path.rstrip("/") or "/"
        if any(path.startswith(p) for p in EXCLUDED_PREFIXES):
            changes["excluded"].append(url)
            continue
        if path in STALE_SITEMAP_REPLACEMENTS:
            for repl in STALE_SITEMAP_REPLACEMENTS[path]:
                add(ORIGIN + repl)
                changes["replaced"].append(f"{path} -> {repl}")
            continue
        add(url)

    for path in SUPPLEMENTAL_PATHS:
        add(ORIGIN + path)
        changes["added"].append(path)

    return out, changes

# Waiting for #root to have children is a better hydration signal than networkidle alone.
HYDRATION_SELECTOR = "#root > *"

# Below this, hydration produced no page-specific content beyond nav/footer.
MIN_RENDERED_TEXT_CHARS = settings.scrape_min_rendered_chars


@dataclass
class PageRecord:
    url: str
    path: str
    slug: str
    status: str  # "ok" | "empty" | "error"
    html_path: str | None = None
    sha256: str | None = None
    html_chars: int = 0
    text_chars: int = 0
    title: str = ""
    fetched_at: str = ""
    attempts: int = 0
    error: str | None = None


@dataclass
class ScrapeReport:
    total: int = 0
    ok: int = 0
    empty: int = 0
    failed: int = 0
    pages: list[PageRecord] = field(default_factory=list)

    @property
    def problems(self) -> list[PageRecord]:
        return [p for p in self.pages if p.status != "ok"]


def slugify(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "home"
    return re.sub(r"[^a-z0-9._-]+", "-", path.lower()).strip("-").replace("/", "__")


def is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.query:  # robots: Disallow: /*?*
        return False
    return not any(parsed.path.startswith(p) for p in DISALLOWED_PREFIXES)


async def _sitemap_body(context, sitemap_url: str) -> str:
    """Fetch the sitemap XML through a browser page (see the TLS note above)."""
    page = await context.new_page()
    try:
        resp = await page.goto(sitemap_url, timeout=settings.scrape_timeout_ms)
        if resp is None:
            raise RuntimeError(f"no response for {sitemap_url}")
        if not resp.ok:
            raise RuntimeError(f"{sitemap_url} returned HTTP {resp.status}")
        return await resp.text()
    finally:
        await page.close()


async def fetch_sitemap_urls(
    sitemap_url: str | None = None, context=None
) -> list[str]:
    """Read the published sitemap and return the crawlable URLs, in order."""
    sitemap_url = sitemap_url or settings.scrape_sitemap_url

    if context is not None:
        body = await _sitemap_body(context, sitemap_url)
    else:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            ctx = await browser.new_context(user_agent=settings.scrape_user_agent)
            try:
                body = await _sitemap_body(ctx, sitemap_url)
            finally:
                await ctx.close()
                await browser.close()

    urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", body)
    # Preserve sitemap order but drop duplicates and disallowed paths.
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u in seen or not is_allowed(u):
            continue
        seen.add(u)
        out.append(u)

    reconciled, changes = reconcile_urls(out)
    log.info(
        "sitemap.loaded",
        url=sitemap_url,
        found=len(urls),
        crawlable=len(out),
        after_reconciliation=len(reconciled),
        replaced=len(changes["replaced"]),
        added=len(changes["added"]),
    )
    for note in changes["replaced"]:
        log.info("sitemap.stale_url_replaced", change=note)
    return reconciled


async def _render_one(context, url: str, attempt: int) -> tuple[str, str, str]:
    """Render a single URL. Returns (html, visible_text, title)."""
    page = await context.new_page()
    try:
        # Block images/fonts/media to speed up render and make networkidle reachable.
        await page.route(
            "**/*",
            lambda route: asyncio.ensure_future(
                route.abort()
                if route.request.resource_type in {"image", "media", "font"}
                else route.continue_()
            ),
        )

        # await page.goto(url, wait_until="domcontentloaded", timeout=settings.scrape_timeout_ms)
        # await page.wait_for_selector(HYDRATION_SELECTOR, state="attached",
        #                              timeout=settings.scrape_timeout_ms)

        await page.goto(url, wait_until="domcontentloaded", timeout=settings.scrape_timeout_ms)
        
        # Wait for the root element
        await page.wait_for_selector("#root", state="attached", timeout=settings.scrape_timeout_ms)
        
        # Wait for actual text content to load (bypassing the loading spinner)
        try:
            min_chars = settings.scrape_min_rendered_chars
            await page.wait_for_function(
                f"() => document.body && document.body.innerText.trim().length >= {min_chars}",
                timeout=15_000
            )
        except PlaywrightTimeout:
            log.debug("scrape.text_length_timeout", url=url, attempt=attempt)

        try:
            await page.wait_for_load_state("networkidle", timeout=15_000)
        except PlaywrightTimeout:
            # Analytics beacons and chat widgets can keep the network busy
            # indefinitely. Hydration already happened, so this is survivable.
            log.debug("scrape.networkidle_timeout", url=url, attempt=attempt)

        # Let late client-side fetches (doctor lists, promotions) paint.
        await page.wait_for_timeout(900)

        html = await page.content()
        text = await page.evaluate("() => document.body ? document.body.innerText : ''")
        title = (await page.title()) or ""
        return html, text, title
    finally:
        await page.close()


async def scrape_url(
    context,
    url: str,
    out_dir: Path,
    max_attempts: int = settings.scrape_max_attempts,
) -> PageRecord:
    slug = slugify(url)
    rec = PageRecord(url=url, path=urlparse(url).path or "/", slug=slug, status="error")

    for attempt in range(1, max_attempts + 1):
        rec.attempts = attempt
        try:
            html, text, title = await _render_one(context, url, attempt)
        except (PlaywrightTimeout, PlaywrightError) as exc:
            rec.error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
            log.warning("scrape.attempt_failed", url=url, attempt=attempt, error=rec.error)
            await asyncio.sleep(2 * attempt)
            continue

        text_chars = len(text.strip())
        if text_chars < settings.scrape_min_rendered_chars and attempt < max_attempts:
            # Almost always a hydration race rather than a genuinely empty page.
            log.warning("scrape.too_short_retrying", url=url, attempt=attempt, chars=text_chars)
            await asyncio.sleep(2 * attempt)
            continue

        rec.text_chars = text_chars
        rec.title = title.strip()
        rec.fetched_at = datetime.now(timezone.utc).isoformat()
        rec.error = None
        rec.status = (
            "ok" if text_chars >= settings.scrape_min_rendered_chars else "empty"
        )

        if rec.status == "ok":
            out_path = out_dir / f"{slug}.html"
            out_path.write_text(html, encoding="utf-8")
            rec.html_path = str(out_path.relative_to(out_dir.parent.parent))
            rec.sha256 = hashlib.sha256(html.encode("utf-8")).hexdigest()
            rec.html_chars = len(html)
            log.info("scrape.ok", url=url, text_chars=text_chars, attempts=attempt)
        else:
            # Never persist an empty render into the committed raw cache.
            log.warning("scrape.empty", url=url, text_chars=text_chars, attempts=attempt)
        return rec

    return rec


async def scrape_site(
    urls: list[str] | None = None,
    out_dir: Path | None = None,
    concurrency: int | None = None,
    limit: int | None = None,
) -> ScrapeReport:
    out_dir = out_dir or RAW_HTML_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    concurrency = concurrency or settings.scrape_concurrency

    sem = asyncio.Semaphore(concurrency)
    report = ScrapeReport()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=settings.scrape_user_agent,
            viewport={"width": 1440, "height": 2400},
            locale="en-US",
        )

        # Resolved here rather than by the caller so the sitemap request reuses
        # this browser's TLS stack.
        if urls is None:
            urls = await fetch_sitemap_urls(context=context)
        if limit:
            urls = urls[:limit]
        report.total = len(urls)

        async def worker(url: str) -> PageRecord:
            async with sem:
                rec = await scrape_url(context, url, out_dir)
                await asyncio.sleep(settings.scrape_delay_seconds)  # politeness
                return rec

        results = await asyncio.gather(*(worker(u) for u in urls))
        await context.close()
        await browser.close()

    report.pages = list(results)
    report.ok = sum(1 for r in results if r.status == "ok")
    report.empty = sum(1 for r in results if r.status == "empty")
    report.failed = sum(1 for r in results if r.status == "error")

    # Prune stale cached files, but only if the crawl retrieved something -
    # otherwise a total outage would look like "no pages" and wipe the cache.
    if report.ok:
        keep = {f"{p.slug}.html" for p in report.pages if p.status == "ok"} | {"manifest.json"}
        for stale in out_dir.glob("*.html"):
            if stale.name not in keep:
                stale.unlink()
                log.info("scrape.pruned_stale_cache_file", file=stale.name)
    else:
        log.warning("scrape.prune_skipped", reason="no page fetched successfully",
                    cached_files=len(list(out_dir.glob("*.html"))))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sitemap": settings.scrape_sitemap_url,
        "totals": {
            "total": report.total,
            "ok": report.ok,
            "empty": report.empty,
            "failed": report.failed,
        },
        "pages": [asdict(p) for p in report.pages],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("scrape.done", total=report.total, ok=report.ok,
             empty=report.empty, failed=report.failed)
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Scrape nawaloka.com into data/raw/")
    parser.add_argument("--limit", type=int, default=None, help="only scrape the first N URLs")
    parser.add_argument("--url", action="append", help="scrape specific URL(s) instead of the sitemap")
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()

    async def run() -> ScrapeReport:
        # urls=None makes scrape_site resolve the sitemap using its own browser.
        return await scrape_site(args.url, concurrency=args.concurrency, limit=args.limit)

    report = asyncio.run(run())

    print(f"\nScraped {report.ok}/{report.total} pages successfully "
          f"({report.empty} empty, {report.failed} failed)")
    for p in report.problems:
        print(f"  ! {p.status:<6} {p.url}  chars={p.text_chars}  {p.error or ''}")

    # Non-zero exit so CI / the pipeline cannot silently proceed on a bad crawl.
    return 0 if not report.problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
