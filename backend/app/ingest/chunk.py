"""Structure-aware chunking of extracted markdown into retrieval chunks.

Breadcrumbs are embedded into chunk text; priced chunks are tagged pricing_authority="sql".
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import DATA_DIR, EXTRACTED_DIR, RAW_HTML_DIR, settings
from app.logging_conf import get_logger

log = get_logger(__name__)

SITE_NAME = "Nawaloka Hospitals"
CATALOGUE_MARKER = "<!-- catalogue -->"

# Chunk sizes, in rough tokens (see _tokens()). Sections are packed greedily
# up to TARGET_TOKENS rather than merged only when under a floor.
TARGET_TOKENS = settings.chunk_target_tokens
MAX_TOKENS = settings.chunk_max_tokens
MIN_TOKENS = settings.chunk_min_tokens   # a stray label, never emit it alone
OVERLAP_TOKENS = settings.chunk_overlap_tokens

# how many catalogue entries per batch
CATALOGUE_BATCH = settings.chunk_catalogue_batch

# Pages with a <select> that is a genuine data catalogue, not form furniture
LAB_CATALOGUE_SLUGS = {"lab-tests-online"}

PRICE_RE = re.compile(r"(?:LKR|Rs\.?)\s*[\d,]+(?:\.\d{2})?", re.I)

# Content taxonomy. Drives retrieval filters, cache eligibility and answer framing.
CONTENT_TYPE_BY_SLUG: dict[str, str] = {
    # Clinical centres of excellence
    "heart-centre": "centre",
    "neuro-centre": "centre",
    "cancer-blood-disease": "centre",
    "diabetes-centre": "centre",
    "dental-unit": "centre",
    "ear-nose-throat-care": "centre",
    "eye-care": "centre",
    "lasik-advanced-eye-care-centre": "centre",
    "fertility-care": "centre",
    "mother-and-baby": "centre",
    "radiology": "centre",
    "rheumatology-sports-medicine": "centre",
    "skin-care-cosmetic-centre": "centre",
    "bone-marrow-transplant-unit": "centre",
    "vaccination-centre": "centre",
    "elite-vip-centre": "centre",
    "elite-corporate-centre": "centre",
    # Operational services
    "laboratory": "service",
    "lab-tests-online": "service",
    "physiotherapy": "service",
    "blood-bank": "service",
    "nutrition-and-dietetics": "service",
    "dialysis": "service",
    "surgical-theaters": "service",
    "channeling": "service",
    "international": "service",
    # Corporate / institutional
    "aboutus": "about",
    "aboutus-awards": "about",
    "aboutus-message": "about",
    "aboutus-reports": "about",
    "aboutus-technology": "about",
    "aboutus-consultants": "about",
    "home": "about",
    # Everything else
    "health-checkups": "package",
    "blogs-and-news": "news",
    "media": "news",
    "promotions": "promotion",
    "contactus": "contact",
    "quick-service-contacts": "contact",
    "ambulance-terms-and-conditions": "policy",
    "ambulance-emergency-patient-care": "emergency",
}

# Human-readable page names for the breadcrumb, where the slug is unclear.
PAGE_LABEL_OVERRIDES = {
    "home": "Home",
    "aboutus": "About Us",
    "aboutus-awards": "Awards & Recognitions",
    "aboutus-message": "Chairman's Message",
    "aboutus-reports": "Annual Reports",
    "aboutus-technology": "Technology",
    "aboutus-consultants": "Resident Consultants",
    "health-checkups": "Health Check Packages",
    "lab-tests-online": "Online Lab Tests",
    "blogs-and-news": "News & Blogs",
    "contactus": "Contact Us",
    "quick-service-contacts": "Quick Service Contacts",
    "ambulance-terms-and-conditions": "Ambulance Terms & Conditions",
    "ambulance-emergency-patient-care": "Ambulance & Emergency Patient Care",
}


@dataclass
class Chunk:
    chunk_id: str
    slug: str
    url: str
    path: str
    page_title: str
    page_label: str
    breadcrumb: list[str]
    section_heading: str
    content: str          # breadcrumb-prefixed text that gets embedded
    body: str             # raw text without the breadcrumb header
    content_type: str
    is_catalogue: bool = False
    has_pricing: bool = False
    pricing_authority: str = ""
    token_count: int = 0
    corpus_version: str = ""
    source_sha256: str = ""
    extra: dict = field(default_factory=dict)


# Rough token count (~4 chars/token), avoids pulling in a real tokenizer.
# TODO: revisit if we ever chunk non-English pages, the ratio won't hold
def _tokens(text: str) -> int:
    return max(1, len(text) // settings.approx_chars_per_token)


def _page_label(slug: str) -> str:
    if slug in PAGE_LABEL_OVERRIDES:
        return PAGE_LABEL_OVERRIDES[slug]
    return slug.replace("__", " - ").replace("-", " ").title()


@dataclass
class _Section:
    heading: str
    level: int
    trail: list[str]
    lines: list[str] = field(default_factory=list)
    is_catalogue: bool = False

    @property
    def body(self) -> str:
        return "\n".join(self.lines).strip()


def split_sections(markdown: str) -> list[_Section]:
    """Split markdown at headings, tracking the heading stack for breadcrumbs."""
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    current = _Section(heading="", level=0, trail=[])
    in_catalogue = False

    for line in markdown.splitlines():
        if line.strip() == CATALOGUE_MARKER:
            # Catalogue content becomes its own section so it can be batched.
            if current.lines:
                sections.append(current)
            current = _Section(
                heading=stack[-1][1] if stack else "",
                level=stack[-1][0] if stack else 0,
                trail=[h for _, h in stack],
                is_catalogue=True,
            )
            in_catalogue = True
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            if current.lines:
                sections.append(current)
            level, heading = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            trail = [h for _, h in stack]
            stack.append((level, heading))
            current = _Section(heading=heading, level=level, trail=trail)
            in_catalogue = False
            continue

        if line.strip():
            current.lines.append(line)
        elif current.lines and not in_catalogue:
            current.lines.append("")

    if current.lines:
        sections.append(current)
    return sections


def _pack_sections(sections: list[_Section]) -> list[_Section]:
    """Greedily pack consecutive sections up to settings.chunk_target_tokens.

    Catalogue sections pass through untouched; they are batched separately.
    """
    out: list[_Section] = []
    buf: _Section | None = None

    def flush() -> None:
        nonlocal buf
        if buf is not None and buf.body.strip():
            out.append(buf)
        buf = None

    for sec in sections:
        if sec.is_catalogue:
            flush()
            out.append(sec)
            continue

        if buf is None:
            buf = _Section(
                heading=sec.heading, level=sec.level, trail=sec.trail, lines=list(sec.lines)
            )
            continue

        combined = _tokens(buf.body) + _tokens(sec.body)
        if combined > settings.chunk_target_tokens:
            flush()
            buf = _Section(
                heading=sec.heading, level=sec.level, trail=sec.trail, lines=list(sec.lines)
            )
            continue

        # Keep the absorbed section's heading visible inside the chunk body.
        if sec.heading:
            buf.lines.extend(["", f"{'#' * min(max(sec.level, 2), 6)} {sec.heading}"])
        buf.lines.extend(["", *sec.lines])

    flush()

    # Fold any undersized runt section into an adjacent prose section.
    merged: list[_Section] = []
    for sec in out:
        if (
            not sec.is_catalogue
            and _tokens(sec.body) < settings.chunk_min_tokens
            and merged
            and not merged[-1].is_catalogue
        ):
            merged[-1].lines.extend(["", *sec.lines])
            continue
        merged.append(sec)

    # If the runt was the very first section, push it into the next one.
    if (
        len(merged) > 1
        and not merged[0].is_catalogue
        and not merged[1].is_catalogue
        and _tokens(merged[0].body) < settings.chunk_min_tokens
    ):
        head = merged.pop(0)
        merged[0].lines = [*head.lines, "", *merged[0].lines]
        merged[0].heading = head.heading or merged[0].heading
        merged[0].trail = head.trail or merged[0].trail

    return merged


def _split_long(body: str) -> list[str]:
    """Recursively split an oversized section on paragraph boundaries."""
    if _tokens(body) <= settings.chunk_max_tokens:
        return [body]

    paras = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    parts: list[str] = []
    buf: list[str] = []

    for para in paras:
        candidate = "\n\n".join([*buf, para])
        if buf and _tokens(candidate) > settings.chunk_target_tokens:
            parts.append("\n\n".join(buf))
            # Carry the tail forward as overlap across the split boundary.
            overlap_chars = (
                settings.chunk_overlap_tokens * settings.approx_chars_per_token
            )
            tail = "\n\n".join(buf)[-overlap_chars:]
            buf = [tail, para] if tail else [para]
        else:
            buf.append(para)

    if buf:
        parts.append("\n\n".join(buf))
    return parts or [body]


# Strips the "<code> - " prefix from catalogue entries so batches sort by test name.
CATALOGUE_CODE_RE = re.compile(r"^[^\w\s]?[A-Za-z]{0,4}\d{4,}\s*-\s*")


def _catalogue_sort_key(item: str) -> str:
    return CATALOGUE_CODE_RE.sub("", item).strip().lower()


def _batch_catalogue(section: _Section) -> list[tuple[str, str]]:
    """Group catalogue items into alphabetical batches: (label, text) pairs."""
    items = [ln.lstrip("- ").strip() for ln in section.lines if ln.strip().startswith("-")]
    items = [i for i in items if i]
    if not items:
        return []

    items.sort(key=_catalogue_sort_key)

    out: list[tuple[str, str]] = []
    for i in range(0, len(items), settings.chunk_catalogue_batch):
        batch = items[i : i + settings.chunk_catalogue_batch]
        first = _catalogue_sort_key(batch[0])[:14].strip().title()
        last = _catalogue_sort_key(batch[-1])[:14].strip().title()
        label = f"{first} to {last}"
        text = "\n".join(f"- {b}" for b in batch)
        out.append((label, text))
    return out


def chunk_page(
    slug: str,
    url: str,
    markdown: str,
    page_title: str,
    corpus_version: str,
    source_sha256: str = "",
) -> list[Chunk]:
    content_type = CONTENT_TYPE_BY_SLUG.get(slug, "other")
    label = _page_label(slug)
    path = "/" + slug.replace("__", "/") if slug != "home" else "/"

    sections = _pack_sections(split_sections(markdown))
    chunks: list[Chunk] = []

    for s_idx, sec in enumerate(sections):
        if sec.is_catalogue:
            if slug not in LAB_CATALOGUE_SLUGS:
                log.debug("chunk.dropped_form_dropdown", slug=slug,
                          options=len(sec.lines))
                continue
            for b_idx, (batch_label, text) in enumerate(_batch_catalogue(sec)):
                breadcrumb = [SITE_NAME, label, "Test Catalogue", batch_label]
                header = "# " + " > ".join(breadcrumb)
                note = (
                    "Alphabetical extract of the hospital's bookable laboratory "
                    "test catalogue. Availability only - prices are held in the "
                    "hospital database, not on this page."
                )
                body = f"{note}\n\n{text}"
                chunks.append(
                    Chunk(
                        chunk_id=f"{slug}::cat{b_idx:03d}",
                        slug=slug,
                        url=url,
                        path=path,
                        page_title=page_title,
                        page_label=label,
                        breadcrumb=breadcrumb,
                        section_heading="Test Catalogue",
                        content=f"{header}\n\n{body}",
                        body=body,
                        content_type="lab_catalogue",
                        is_catalogue=True,
                        has_pricing=False,
                        token_count=_tokens(body),
                        corpus_version=corpus_version,
                        source_sha256=source_sha256,
                        extra={"batch": batch_label, "items": len(text.splitlines())},
                    )
                )
            continue

        body = sec.body
        if not body.strip():
            continue

        for c_idx, part in enumerate(_split_long(body)):
            trail = [t for t in sec.trail if t]
            breadcrumb = [SITE_NAME, label, *trail]
            if sec.heading and sec.heading not in breadcrumb:
                breadcrumb.append(sec.heading)
            header = "# " + " > ".join(breadcrumb)
            has_price = bool(PRICE_RE.search(part))

            chunks.append(
                Chunk(
                    chunk_id=f"{slug}::s{s_idx:02d}::c{c_idx}",
                    slug=slug,
                    url=url,
                    path=path,
                    page_title=page_title,
                    page_label=label,
                    breadcrumb=breadcrumb,
                    section_heading=sec.heading,
                    content=f"{header}\n\n{part}",
                    body=part,
                    content_type=content_type,
                    has_pricing=has_price,
                    # The relational database is the system of record for money.
                    pricing_authority="sql" if has_price else "",
                    token_count=_tokens(part),
                    corpus_version=corpus_version,
                    source_sha256=source_sha256,
                )
            )

    return chunks


def chunk_all(
    extracted_dir: Path | None = None,
    raw_dir: Path | None = None,
    corpus_version: str | None = None,
) -> list[Chunk]:
    extracted_dir = extracted_dir or EXTRACTED_DIR
    raw_dir = raw_dir or RAW_HTML_DIR

    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    by_slug = {p["slug"]: p for p in manifest["pages"] if p["status"] == "ok"}
    corpus_version = corpus_version or manifest["generated_at"]

    files = sorted(extracted_dir.glob("*.md"))
    if not files:
        raise SystemExit(
            f"No extracted markdown in {extracted_dir}. Run: python -m app.ingest.extract"
        )

    all_chunks: list[Chunk] = []
    for f in files:
        meta = by_slug.get(f.stem, {})
        all_chunks.extend(
            chunk_page(
                slug=f.stem,
                url=meta.get("url", ""),
                markdown=f.read_text(encoding="utf-8"),
                page_title=meta.get("title", ""),
                corpus_version=corpus_version,
                source_sha256=meta.get("sha256", ""),
            )
        )

    log.info(
        "chunk.done",
        pages=len(files),
        chunks=len(all_chunks),
        catalogue=sum(1 for c in all_chunks if c.is_catalogue),
        priced=sum(1 for c in all_chunks if c.has_pricing),
        corpus_version=corpus_version,
    )
    return all_chunks


def write_chunks(chunks: list[Chunk], out_path: Path | None = None) -> Path:
    out_path = out_path or (DATA_DIR / "chunks" / "chunks.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return out_path


def stable_point_id(chunk_id: str) -> int:
    """Deterministic 63-bit id so re-ingesting updates rather than duplicates."""
    return int(hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()[:15], 16)


def main() -> int:
    from collections import Counter

    chunks = chunk_all()
    path = write_chunks(chunks)

    by_type = Counter(c.content_type for c in chunks)
    toks = sorted(c.token_count for c in chunks)

    print(f"\n{len(chunks)} chunks written to {path}")
    print(f"\n{'content_type':<18}{'chunks':>8}{'tokens':>10}")
    print("-" * 36)
    for t, n in by_type.most_common():
        tot = sum(c.token_count for c in chunks if c.content_type == t)
        print(f"{t:<18}{n:>8}{tot:>10,}")
    print("-" * 36)
    print(f"{'TOTAL':<18}{len(chunks):>8}{sum(toks):>10,}")

    print(f"\ntoken size: min {toks[0]}  p50 {toks[len(toks)//2]}  "
          f"p90 {toks[int(len(toks)*0.9)]}  max {toks[-1]}")
    print(f"chunks with pricing (SQL authoritative): "
          f"{sum(1 for c in chunks if c.has_pricing)}")

    oversized = [c for c in chunks if c.token_count > settings.chunk_max_tokens]
    if oversized:
        print(f"\n{len(oversized)} chunk(s) exceed MAX_TOKENS={settings.chunk_max_tokens}:")
        for c in oversized[:5]:
            print(f"  ! {c.chunk_id} ({c.token_count})")

    print("\nSample chunk (breadcrumb-prefixed text that gets embedded):")
    sample = next(c for c in chunks if c.content_type == "centre")
    print("-" * 70)
    print(sample.content[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
