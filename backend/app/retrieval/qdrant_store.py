"""Qdrant collection management and indexing."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from typing import Iterable, Sequence

from qdrant_client import QdrantClient, models

from app.config import settings
from app.ingest.chunk import Chunk, stable_point_id
from app.logging_conf import get_logger
from app.retrieval.embedder import DENSE_DIM, embed_passages, sparse_passages

log = get_logger(__name__)

DENSE = "dense"
SPARSE = "bm25"

# Payload fields worth an index: everything we filter on.
KEYWORD_INDEXES = ("path", "slug", "content_type", "corpus_version")
BOOL_INDEXES = ("has_pricing", "is_catalogue")


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Process-wide client.

    Cached deliberately: constructing a QdrantClient per call opens a fresh
    HTTP connection and costs ~700 ms per search on this corpus, which is two
    orders of magnitude more than the actual vector query.
    """
    return QdrantClient(url=settings.qdrant_url, timeout=60)


def ensure_collection(
    client: QdrantClient, name: str, recreate: bool = False
) -> None:
    exists = client.collection_exists(name)
    if exists and recreate:
        log.info("qdrant.dropping_collection", collection=name)
        client.delete_collection(name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE: models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                # IDF is REQUIRED for BM25. Without this modifier Qdrant stores
                # raw term frequencies and scores come out meaningless - a
                # silent correctness bug rather than an error.
                SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        log.info("qdrant.created_collection", collection=name)

    for field in KEYWORD_INDEXES:
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:  # already indexed
            pass
    for field in BOOL_INDEXES:
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.BOOL,
            )
        except Exception:
            pass


def chunk_payload(chunk: Chunk) -> dict:
    d = asdict(chunk)
    # `content` is the breadcrumb-prefixed text we embed; keep it for display
    # and citation, but the reranker scores against `body` plus breadcrumb.
    d.pop("extra", None)
    d["breadcrumb_text"] = " > ".join(chunk.breadcrumb)
    d.update(chunk.extra)
    return d


def index_chunks(
    chunks: Sequence[Chunk],
    collection: str | None = None,
    recreate: bool = False,
    batch_size: int = 64,
) -> int:
    collection = collection or settings.qdrant_collection
    client = get_client()
    ensure_collection(client, collection, recreate=recreate)

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = list(chunks[start : start + batch_size])
        texts = [c.content for c in batch]

        dense_vecs = embed_passages(texts)
        sparse_vecs = sparse_passages(texts)

        points = []
        for chunk, dv, (idx, val) in zip(batch, dense_vecs, sparse_vecs):
            points.append(
                models.PointStruct(
                    # Deterministic id: re-ingesting the same corpus updates
                    # points in place instead of duplicating them.
                    id=stable_point_id(chunk.chunk_id),
                    vector={
                        DENSE: dv,
                        SPARSE: models.SparseVector(indices=idx, values=val),
                    },
                    payload=chunk_payload(chunk),
                )
            )

        client.upsert(collection_name=collection, points=points, wait=True)
        total += len(points)
        log.info("qdrant.upserted", collection=collection, batch=len(points), total=total)

    return total


def prune_stale(collection: str, corpus_version: str) -> int:
    """Delete points from older corpus versions.

    Chunk ids are content-derived, so a page that loses a section leaves an
    orphan point behind. Deleting anything not matching the current
    corpus_version keeps the index consistent with data/raw.
    """
    client = get_client()
    result = client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must_not=[
                    models.FieldCondition(
                        key="corpus_version",
                        match=models.MatchValue(value=corpus_version),
                    )
                ]
            )
        ),
        wait=True,
    )
    log.info("qdrant.pruned_stale", collection=collection, status=str(result.status))
    return 0


def collection_stats(collection: str | None = None) -> dict:
    collection = collection or settings.qdrant_collection
    client = get_client()
    if not client.collection_exists(collection):
        return {"exists": False}
    info = client.get_collection(collection)
    return {
        "exists": True,
        "points": info.points_count,
        "indexed_vectors": info.indexed_vectors_count,
        "status": str(info.status),
    }


def ensure_simple_collection(client: QdrantClient, name: str) -> None:
    """Dense-only collection, used by the FAQ store and the semantic cache."""
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=DENSE_DIM, distance=models.Distance.COSINE
            ),
        )
        log.info("qdrant.created_collection", collection=name)


def iter_all_payloads(collection: str, limit: int = 1000) -> Iterable[dict]:
    client = get_client()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=limit, offset=offset, with_payload=True
        )
        for p in points:
            yield p.payload or {}
        if offset is None:
            break
