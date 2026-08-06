"""Cached raw HTML -> clean markdown: strip chrome, flatten to blocks, drop
statistical boilerplate, and render heading-preserving markdown for the chunker.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from app.config import EXTRACTED_DIR, RAW_HTML_DIR, settings
from app.logging_conf import get_logger

log = get_logger(__name__)

# these never hold page content
DROP_TAGS = ("script", "style", "noscript", "svg", "iframe", "template")

# the chrome landmarks the site actually gives us
CHROME_TAGS = ("nav", "footer", "header")

# part of the surrounding text, not a block of their own
INLINE_TAGS = {
    "a", "span", "strong", "em", "b", "i", "u", "small", "br", "code",
    "sup", "sub", "abbr", "mark", "label", "img", "figcaption", "time",
}

HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

# seen on more than this fraction of pages = site chrome
BOILERPLATE_RATIO = settings.extract_boilerplate_ratio

# anything shorter is nearly always a UI label - "Read More", "01"
MIN_BLOCK_CHARS = settings.extract_min_block_chars

# a <select> this big is a data catalogue, not a form control
CATALOGUE_MIN_OPTIONS = settings.extract_catalogue_min_options


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
    """Normalised key for cross-page boilerplate comparison."""
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
        # Drop placeholder options like "Select a test".
        options = [o for o in options if not re.fullmatch(r"select( a)?\b.*", o, re.I)]
        if len(options) >= settings.extract_catalogue_min_options:
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
        if len(text) >= settings.extract_min_block_chars:
            kind = "item" if node.name in {"li", "td", "th"} else "para"
            out.append(Block(kind, text))
        return

    # Text directly on this node, interleaved with child elements.
    direct = _norm(
        "".join(c for c in node.children if isinstance(c, NavigableString))
    )
    if len(direct) >= settings.extract_min_block_chars:
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

    # Collapse consecutive duplicates (Chakra renders mobile + desktop copies).
    deduped: list[Block] = []
    seen_in_page: set[str] = set()
    for b in blocks:
        key = (b.kind, _norm_key(b.text))
        if b.kind != "catalogue" and key in seen_in_page:
            continue
        seen_in_page.add(key)
        deduped.append(b)

    return deduped, title, dom_chars


def find_boilerplate(
    pages: dict[str, list[Block]],
    ratio: float = settings.extract_boilerplate_ratio,
) -> set[str]:
    """Blocks appearing on more than *ratio* of pages are treated as site chrome."""
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
    """Render blocks to markdown, skipping boilerplate. Returns (markdown, kept_chars, dropped)."""
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
