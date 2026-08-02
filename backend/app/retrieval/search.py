"""Hybrid retrieval: dense + BM25 fused server-side, then cross-encoder reranked.

Pipeline
--------
1. Dense (bge-small) and sparse (BM25) searches run as Qdrant *prefetches*, so
   both happen in a single round trip.
2. Reciprocal Rank Fusion merges them. RRF is used rather than score-weighted
   blending because cosine similarity and BM25 scores are not on comparable
   scales; RRF only needs the ranks.
3. A cross-encoder reranks the fused candidates. Bi-encoder retrieval scores a
   query and passage independently; the cross-encoder sees them together and is
   far better at rejecting topically-adjacent-but-wrong passages.
4. Anything below the score floor is dropped. If nothing survives, the caller
   gets an empty list and the agent says it does not know - which is the
   correct answer far more often than the best of five bad chunks.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from qdrant_client import models

from app.config import settings
from app.logging_conf import get_logger
from app.retrieval.embedder import embed_query, rerank, sparse_query
from app.retrieval.qdrant_store import DENSE, SPARSE, get_client

log = get_logger(__name__)

# Characters of chunk body fed to the cross-encoder. ~1200 chars is roughly the
# model's 512-token window once the breadcrumb is prepended.
RERANK_DOC_CHARS = 1200


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    body: str
    url: str
    page_label: str
    breadcrumb: list[str]
    section_heading: str
    content_type: str
    has_pricing: bool
    pricing_authority: str
    fusion_score: float
    rerank_score: float | None = None
    rank: int = 0

    @property
    def citation(self) -> str:
        return " > ".join(self.breadcrumb)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "citation": self.citation,
            "url": self.url,
            "content_type": self.content_type,
            "fusion_score": round(self.fusion_score, 4),
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score is not None else None,
            "has_pricing": self.has_pricing,
        }


@dataclass
class SearchResult:
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    candidates_considered: int = 0
    dropped_below_floor: int = 0
    reranked: bool = False
    elapsed_ms: int = 0

    @property
    def abstained(self) -> bool:
        return not self.chunks


def _sigmoid(x: float) -> float:
    """Cross-encoder outputs are logits; squash to 0-1 so the floor is portable."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _build_filter(
    content_types: list[str] | None = None,
    exclude_content_types: list[str] | None = None,
    paths: list[str] | None = None,
) -> models.Filter | None:
    must: list[models.Condition] = []
    must_not: list[models.Condition] = []

    if content_types:
        must.append(
            models.FieldCondition(key="content_type", match=models.MatchAny(any=content_types))
        )
    if paths:
        must.append(models.FieldCondition(key="path", match=models.MatchAny(any=paths)))
    if exclude_content_types:
        must_not.append(
            models.FieldCondition(
                key="content_type", match=models.MatchAny(any=exclude_content_types)
            )
        )

    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


def hybrid_search(
    query: str,
    top_k: int | None = None,
    collection: str | None = None,
    content_types: list[str] | None = None,
    exclude_content_types: list[str] | None = None,
    paths: list[str] | None = None,
    use_rerank: bool | None = None,
    score_floor: float | None = None,
) -> SearchResult:
    started = time.perf_counter()
    collection = collection or settings.qdrant_collection
    top_k = top_k or settings.retrieval_top_k
    use_rerank = settings.enable_rerank if use_rerank is None else use_rerank
    score_floor = settings.retrieval_score_floor if score_floor is None else score_floor

    client = get_client()
    dense_vec = embed_query(query)
    sp_idx, sp_val = sparse_query(query)

    response = client.query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(
                query=dense_vec, using=DENSE, limit=settings.retrieval_prefetch_limit
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sp_idx, values=sp_val),
                using=SPARSE,
                limit=settings.retrieval_prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=settings.retrieval_fusion_limit,
        with_payload=True,
        query_filter=_build_filter(content_types, exclude_content_types, paths),
    )

    candidates: list[RetrievedChunk] = []
    for point in response.points:
        p = point.payload or {}
        candidates.append(
            RetrievedChunk(
                chunk_id=p.get("chunk_id", str(point.id)),
                content=p.get("content", ""),
                body=p.get("body", ""),
                url=p.get("url", ""),
                page_label=p.get("page_label", ""),
                breadcrumb=p.get("breadcrumb", []),
                section_heading=p.get("section_heading", ""),
                content_type=p.get("content_type", "other"),
                has_pricing=bool(p.get("has_pricing")),
                pricing_authority=p.get("pricing_authority", ""),
                fusion_score=float(point.score),
            )
        )

    result = SearchResult(query=query, candidates_considered=len(candidates))

    if not candidates:
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        log.info("search.empty", query=query[:80])
        return result

    if use_rerank:
        # Score against the breadcrumb + body so the model sees which page a
        # passage came from, exactly as it was embedded.
        #
        # Bodies are truncated to RERANK_DOC_CHARS first. The cross-encoder has
        # a 512-token limit and truncates internally anyway, so passing full
        # ~450-token chunks costs CPU without adding signal; the breadcrumb and
        # opening sentences carry the topicality the reranker is judging.
        docs = [
            f"{c.citation}\n{c.body[:RERANK_DOC_CHARS]}" for c in candidates
        ]
        scores = rerank(query, docs)
        for chunk, raw in zip(candidates, scores):
            chunk.rerank_score = _sigmoid(raw)
        candidates.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        result.reranked = True

        kept = [c for c in candidates if (c.rerank_score or 0.0) >= score_floor]
        result.dropped_below_floor = len(candidates) - len(kept)
        candidates = kept

    for i, chunk in enumerate(candidates[:top_k], start=1):
        chunk.rank = i
    result.chunks = candidates[:top_k]
    result.elapsed_ms = int((time.perf_counter() - started) * 1000)

    log.info(
        "search.done",
        query=query[:80],
        candidates=result.candidates_considered,
        returned=len(result.chunks),
        dropped=result.dropped_below_floor,
        ms=result.elapsed_ms,
    )
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Query the vector store from the CLI")
    parser.add_argument("query", nargs="+")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--floor", type=float, default=None)
    parser.add_argument("--type", action="append", dest="content_types")
    parser.add_argument("--full", action="store_true", help="print full chunk bodies")
    args = parser.parse_args()

    q = " ".join(args.query)
    res = hybrid_search(
        q,
        top_k=args.top_k,
        use_rerank=not args.no_rerank,
        score_floor=args.floor,
        content_types=args.content_types,
    )

    print(f"\nquery: {q!r}")
    print(f"candidates {res.candidates_considered}  returned {len(res.chunks)}  "
          f"dropped {res.dropped_below_floor}  {res.elapsed_ms} ms\n")

    if res.abstained:
        print("  (no chunk cleared the score floor - the agent would abstain)")
        return 0

    for c in res.chunks:
        rr = f"{c.rerank_score:.3f}" if c.rerank_score is not None else "  -  "
        print(f"[{c.rank}] rerank={rr}  rrf={c.fusion_score:.4f}  ({c.content_type})")
        print(f"    {c.citation}")
        preview = c.body if args.full else c.body[:220].replace("\n", " ")
        print(f"    {preview}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
