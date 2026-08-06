"""Hybrid search over the scraped site, through LlamaIndex.

Dense (bge-small) and sparse (BM25) both go to Qdrant in one hybrid query and
come back fused via LlamaIndex. Reranking uses our own fastembed cross-encoder
(app/retrieval/rerank.py) rather than LlamaIndex's torch-based one. If nothing
clears the score floor, the caller gets an empty list and the agent abstains.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from app.config import settings
from app.logging_conf import get_logger
from app.retrieval.rerank import build_reranker

log = get_logger(__name__)


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


@lru_cache(maxsize=2)
def get_index(collection: str):
    """Attach to an existing collection. Indexing lives in qdrant_store.py."""
    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    from app.retrieval.embedder import llama_embedding
    from app.retrieval.qdrant_store import get_client

    store = QdrantVectorStore(
        client=get_client(),
        collection_name=collection,
        enable_hybrid=True,
        fastembed_sparse_model=settings.sparse_model,
    )
    return VectorStoreIndex.from_vector_store(store, embed_model=llama_embedding())


def _filters(
    content_types: list[str] | None,
    exclude_content_types: list[str] | None,
    paths: list[str] | None,
) -> MetadataFilters | None:
    filters = []
    if content_types:
        filters.append(MetadataFilter(
            key="content_type", value=content_types, operator=FilterOperator.IN))
    if exclude_content_types:
        filters.append(MetadataFilter(
            key="content_type", value=exclude_content_types, operator=FilterOperator.NIN))
    if paths:
        filters.append(MetadataFilter(key="path", value=paths, operator=FilterOperator.IN))
    return MetadataFilters(filters=filters) if filters else None


def _to_chunk(node) -> RetrievedChunk:
    md = node.node.metadata
    return RetrievedChunk(
        chunk_id=md.get("chunk_id", node.node.node_id),
        content=node.node.get_content(),
        body=md.get("body", ""),
        url=md.get("url", ""),
        page_label=md.get("page_label", ""),
        breadcrumb=md.get("breadcrumb") or [],
        section_heading=md.get("section_heading", ""),
        content_type=md.get("content_type", "other"),
        has_pricing=bool(md.get("has_pricing")),
        pricing_authority=md.get("pricing_authority", ""),
        fusion_score=float(node.score or 0.0),
    )


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

    retriever = get_index(collection).as_retriever(
        vector_store_query_mode="hybrid",
        similarity_top_k=settings.retrieval_fusion_limit,
        sparse_top_k=settings.retrieval_prefetch_limit,
        filters=_filters(content_types, exclude_content_types, paths),
    )
    nodes = retriever.retrieve(query)

    result = SearchResult(query=query, candidates_considered=len(nodes))
    if not nodes:
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        log.info("search.empty", query=query[:80])
        return result

    # Keep the fusion score before the reranker overwrites it - the trace panel shows both.
    chunks = [_to_chunk(n) for n in nodes]

    if use_rerank:
        from llama_index.core.schema import QueryBundle

        reranker = build_reranker(top_k=top_k, score_floor=score_floor)
        kept = reranker.postprocess_nodes(nodes, QueryBundle(query))
        result.reranked = True
        result.dropped_below_floor = reranker.dropped

        chunks = []
        for node in kept:
            chunk = _to_chunk(node)
            chunk.rerank_score = node.score
            chunk.fusion_score = float(
                next((n.score or 0.0) for n in nodes if n.node.node_id == node.node.node_id)
            )
            chunks.append(chunk)

    for i, chunk in enumerate(chunks[:top_k], start=1):
        chunk.rank = i
    result.chunks = chunks[:top_k]
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
        print(f"[{c.rank}] rerank={rr}  hybrid={c.fusion_score:.4f}  ({c.content_type})")
        print(f"    {c.citation}")
        preview = c.body if args.full else c.body[:220].replace("\n", " ")
        print(f"    {preview}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
