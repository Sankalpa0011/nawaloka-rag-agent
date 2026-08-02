"""Turn cached raw HTML into clean, structured markdown.

Why not just use trafilatura
----------------------------
Trafilatura is the usual right answer and it is still used here as a
cross-check, but on this site it discards most of the page. Measured across the
41 cached pages it recovers only ~44% of the visible text, and on
``/lab-tests-online`` just 304 of 52,002 characters (0.6%).

The cause is structural. The site is built with Chakra UI, which emits no
semantic containers at all - no ``<main>``, ``<article>`` or ``<section>``,
only bare ``<div class="css-1p1m4ay">``. Trafilatura's extraction relies on
those landmarks plus text-density heuristics, so it has nothing to anchor to.
Content that is a form control rather than prose (the lab-test catalogue is a
``<select>`` with thousands of ``<option>`` elements) is invisible to it.

So extraction here walks the DOM directly:

1. Drop non-content nodes (script/style/svg) and the chrome landmarks that DO
   exist (``<nav>``, ``<footer>``).
2. Flatten the remaining tree to leaf-level text blocks, tagging headings,
   list items, table rows and ``<select>`` catalogues.
3. Remove site-wide boilerplate *statistically* rather than with hand-written
   selectors: any text block that appears on more than ``BOILERPLATE_RATIO``
   of pages is chrome by definition. This is what removes the phone numbers,
   the mega-menu and the footer without a brittle list of CSS classes, and it
   keeps working if the site is restyled.
4. Emit markdown that preserves heading hierarchy, which the chunker then uses
   to build breadcrumb-prefixed chunks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from app.config import EXTRACTED_DIR, RAW_HTML_DIR
from app.logging_conf import get_logger

log = get_logger(__name__)

# Nodes that never contain page content.
DROP_TAGS = ("script", "style", "noscript", "svg", "iframe", "template")

# Chrome landmarks the site does provide.
CHROME_TAGS = ("nav", "footer", "header")

# Treated as part of the surrounding text rather than as their own block.
INLINE_TAGS = {
    "a", "span", "strong", "em", "b", "i", "u", "small", "br", "code",
    "sup", "sub", "abbr", "mark", "label", "img", "figcaption", "time",
}

HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

# A text block seen on more than this fraction of pages is site chrome.
BOILERPLATE_RATIO = 0.35

# Blocks shorter than this are almost always UI labels ("Read More", "01").
MIN_BLOCK_CHARS = 3

# A <select> with at least this many options is a data catalogue, not a form.
CATALOGUE_MIN_OPTIONS = 25


@dataclass
class Block:
    kind: str  # heading | para | item | option | catalogue
    text: str
    level: int = 0  # heading level


@dataclass
class ExtractedPage:
    slug: str
    url: str
    title: str
    markdown: str
    blocks: list[Block] = field(default_factory=list)
    dom_chars: int = 0
    kept_chars: int = 0
    dropped_boilerplate: int = 0

    @property
    def coverage(self) -> float:
        return self.kept_chars / self.dom_chars if self.dom_chars else 0.0


def _norm(text: str) -> str:
    return " ".join(text.split())


def _norm_key(text: str) -> str:
    """Normalised form used for cross-page boilerplate comparison."""
    return re.sub(r"[^a-z0-9 ]+", "", _norm(text).lower())[:120]


def _is_block(node: Tag) -> bool:
    return isinstance(node, Tag) and node.name not in INLINE_TAGS


def _harvest(node: Tag, out: list[Block]) -> None:
    """Depth-first walk emitting leaf-level text blocks in document order."""
    if not isinstance(node, Tag):
        return

    if node.name == "select":
        options = [
            _norm(o.get_text(" ", strip=True))
            for o in node.find_all("option")
            if _norm(o.get_text(" ", strip=True))
        ]
        # Drop placeholder entries like "Select a test".
        options = [o for o in options if not re.fullmatch(r"select( a)?\b.*", o, re.I)]
        if len(options) >= CATALOGUE_MIN_OPTIONS:
            out.append(Block("catalogue", "\n".join(options)))
        elif options:
            out.append(Block("para", ", ".join(options)))
        return

    if node.name in HEADING_TAGS:
        text = _norm(node.get_text(" ", strip=True))
        if text:
            out.append(Block("heading", text, HEADING_TAGS[node.name]))
        return

    child_blocks = [c for c in node.children if _is_block(c)]

    if not child_blocks:
        text = _norm(node.get_text(" ", strip=True))
        if len(text) >= MIN_BLOCK_CHARS:
            kind = "item" if node.name in {"li", "td", "th"} else "para"
            out.append(Block(kind, text))
        return

    # Text sitting directly on this node, interleaved with child elements.
    direct = _norm(
        "".join(c for c in node.children if isinstance(c, NavigableString))
    )
    if len(direct) >= MIN_BLOCK_CHARS:
        out.append(Block("para", direct))

    for child in child_blocks:
        _harvest(child, out)


def page_blocks(html: str) -> tuple[list[Block], str, int]:
    """Parse one page into blocks. Returns (blocks, title, dom_char_count)."""
    soup = BeautifulSoup(html, "lxml")
    title = _norm(soup.title.get_text()) if soup.title else ""

    for tag in soup(list(DROP_TAGS)):
        tag.decompose()

    root = soup.find(id="root") or soup.body or soup
    dom_chars = len(_norm(root.get_text(" ", strip=True)))

    for tag in root.find_all(list(CHROME_TAGS)):
        tag.decompose()

    blocks: list[Block] = []
    _harvest(root, blocks)

    # Collapse consecutive duplicates (Chakra renders mobile + desktop copies
    # of the same node, so the identical text is present twice).
    deduped: list[Block] = []
    seen_in_page: set[str] = set()
    for b in blocks:
        key = (b.kind, _norm_key(b.text))
        if b.kind != "catalogue" and key in seen_in_page:
            continue
        seen_in_page.add(key)
        deduped.append(b)

    return deduped, title, dom_chars


def find_boilerplate(pages: dict[str, list[Block]], ratio: float = BOILERPLATE_RATIO) -> set[str]:
    """Text blocks appearing on more than *ratio* of pages are site chrome."""
    counts: dict[str, int] = {}
    for blocks in pages.values():
        for key in {_norm_key(b.text) for b in blocks if b.kind != "catalogue"}:
            counts[key] = counts.get(key, 0) + 1

    threshold = max(2, int(len(pages) * ratio))
    boiler = {k for k, n in counts.items() if n >= threshold and k}
    log.info("extract.boilerplate_detected",
             pages=len(pages), threshold=threshold, blocks=len(boiler))
    return boiler


def blocks_to_markdown(blocks: list[Block], boilerplate: set[str]) -> tuple[str, int, int]:
    """Render blocks to markdown, skipping boilerplate.

    Returns (markdown, kept_chars, dropped_boilerplate_blocks).
    """
    lines: list[str] = []
    kept = 0
    dropped = 0

    for b in blocks:
        if b.kind != "catalogue" and _norm_key(b.text) in boilerplate:
            dropped += 1
            continue

        if b.kind == "heading":
            lines.append("")
            lines.append(f"{'#' * min(b.level, 6)} {b.text}")
            lines.append("")
        elif b.kind == "item":
            lines.append(f"- {b.text}")
        elif b.kind == "catalogue":
            lines.append("")
            lines.append("<!-- catalogue -->")
            lines.extend(f"- {o}" for o in b.text.split("\n"))
            lines.append("")
        else:
            lines.append(b.text)
            lines.append("")
        kept += len(b.text)

    md = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return md, kept, dropped


def extract_all(raw_dir: Path | None = None, out_dir: Path | None = None) -> list[ExtractedPage]:
    raw_dir = raw_dir or RAW_HTML_DIR
    out_dir = out_dir or EXTRACTED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = raw_dir / "manifest.json"
    url_by_slug: dict[str, str] = {}
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        url_by_slug = {p["slug"]: p["url"] for p in data.get("pages", [])}

    files = sorted(raw_dir.glob("*.html"))
    if not files:
        raise SystemExit(f"No cached HTML in {raw_dir}. Run: python -m app.ingest.scrape")

    parsed: dict[str, list[Block]] = {}
    meta: dict[str, tuple[str, int]] = {}
    for f in files:
        blocks, title, dom_chars = page_blocks(f.read_text(encoding="utf-8"))
        parsed[f.stem] = blocks
        meta[f.stem] = (title, dom_chars)

    boilerplate = find_boilerplate(parsed)

    pages: list[ExtractedPage] = []
    for slug, blocks in parsed.items():
        title, dom_chars = meta[slug]
        md, kept, dropped = blocks_to_markdown(blocks, boilerplate)
        page = ExtractedPage(
            slug=slug,
            url=url_by_slug.get(slug, ""),
            title=title,
            markdown=md,
            blocks=blocks,
            dom_chars=dom_chars,
            kept_chars=kept,
            dropped_boilerplate=dropped,
        )
        pages.append(page)
        (out_dir / f"{slug}.md").write_text(md, encoding="utf-8")

    pages.sort(key=lambda p: p.slug)
    log.info("extract.done", pages=len(pages),
             chars=sum(len(p.markdown) for p in pages))
    return pages


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Extract markdown from cached HTML")
    parser.add_argument("--min-chars", type=int, default=300,
                        help="fail if any page extracts fewer than this many chars")
    args = parser.parse_args()

    pages = extract_all()

    print(f"\n{'page':<36}{'dom':>8}{'kept':>8}{'md':>8}{'cov':>7}{'boiler':>8}")
    print("-" * 75)
    thin: list[ExtractedPage] = []
    for p in sorted(pages, key=lambda x: x.coverage):
        flag = ""
        if len(p.markdown) < args.min_chars:
            thin.append(p)
            flag = "  <<< THIN"
        print(f"{p.slug:<36}{p.dom_chars:>8}{p.kept_chars:>8}{len(p.markdown):>8}"
              f"{p.coverage:>6.0%}{p.dropped_boilerplate:>8}{flag}")

    total_md = sum(len(p.markdown) for p in pages)
    total_dom = sum(p.dom_chars for p in pages)
    print("-" * 75)
    print(f"{len(pages)} pages   dom {total_dom:,}   markdown {total_md:,}   "
          f"mean coverage {sum(p.coverage for p in pages)/len(pages):.0%}")

    if thin:
        print(f"\n{len(thin)} page(s) under {args.min_chars} chars:")
        for p in thin:
            print(f"  ! {p.slug} ({len(p.markdown)})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
