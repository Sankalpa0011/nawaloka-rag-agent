"""Creating Qdrant collections and pushing chunks into them."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from functools import lru_cache
from typing import Iterable, Sequence

from qdrant_client import QdrantClient, models

from app.config import settings
from app.ingest.chunk import Chunk
from app.logging_conf import get_logger
from app.retrieval.embedder import DENSE_DIM

log = get_logger(__name__)

DENSE = "dense"
SPARSE = "bm25"

# everything we filter on gets a payload index
KEYWORD_INDEXES = ("path", "slug", "content_type", "corpus_version")
BOOL_INDEXES = ("has_pricing", "is_catalogue")


# One cached client per process - avoids reopening an HTTP connection per call.
@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url,
                        timeout=settings.qdrant_timeout_seconds)


# LlamaIndex creates the collection itself; this adds the payload indexes we filter on.
def ensure_payload_indexes(client: QdrantClient, name: str) -> None:
    fields = [(f, models.PayloadSchemaType.KEYWORD) for f in KEYWORD_INDEXES]
    fields += [(f, models.PayloadSchemaType.BOOL) for f in BOOL_INDEXES]

    for field, schema in fields:
        try:
            client.create_payload_index(
                collection_name=name, field_name=field, field_schema=schema
            )
        except Exception:  # already indexed, or the collection has no such key yet
            pass


def chunk_payload(chunk: Chunk) -> dict:
    d = asdict(chunk)
    d.pop("extra", None)
    d["breadcrumb_text"] = " > ".join(chunk.breadcrumb)
    d.update(chunk.extra)
    return d


def _to_nodes(chunks: Sequence[Chunk]):
    """Chunk -> TextNode, keeping every field read back in search.py."""
    from llama_index.core.schema import TextNode

    skip = [
        "chunk_id", "slug", "url", "path", "page_title", "page_label",
        "breadcrumb", "breadcrumb_text", "section_heading", "body",
        "content_type", "is_catalogue", "has_pricing", "pricing_authority",
        "token_count", "corpus_version", "source_sha256",
    ]

    nodes = []
    for chunk in chunks:
        metadata = chunk_payload(chunk)
        metadata.pop("content", None)
        nodes.append(
            TextNode(
                # Deterministic id (uuid5) so re-ingesting the same corpus overwrites
                # points instead of duplicating them; Qdrant requires a UUID or int here.
                id_=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                text=chunk.content,
                metadata=metadata,
                excluded_embed_metadata_keys=skip,
                excluded_llm_metadata_keys=skip,
            )
        )
    return nodes


def index_chunks(
    chunks: Sequence[Chunk],
    collection: str | None = None,
    recreate: bool = False,
    batch_size: int = settings.qdrant_index_batch_size,
) -> int:
    """Write chunks through LlamaIndex so search.py can read them back."""
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    from app.retrieval.embedder import llama_embedding

    collection = collection or settings.qdrant_collection
    client = get_client()

    if recreate and client.collection_exists(collection):
        log.info("qdrant.dropping_collection", collection=collection)
        client.delete_collection(collection)

    store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        enable_hybrid=True,
        fastembed_sparse_model=settings.sparse_model,
        batch_size=batch_size,
    )
    storage = StorageContext.from_defaults(vector_store=store)

    nodes = _to_nodes(chunks)
    VectorStoreIndex(
        nodes,
        storage_context=storage,
        embed_model=llama_embedding(),
        show_progress=False,
    )

    ensure_payload_indexes(client, collection)
    log.info("qdrant.indexed", collection=collection, nodes=len(nodes))
    return len(nodes)


# Delete points from older corpus versions, keeping the index in step with data/raw.
def prune_stale(collection: str, corpus_version: str) -> int:
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


# Dense-only collection, used by the FAQ store and the semantic cache.
def ensure_simple_collection(client: QdrantClient, name: str) -> None:
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=DENSE_DIM, distance=models.Distance.COSINE
            ),
        )
        log.info("qdrant.created_collection", collection=name)


def iter_all_payloads(
    collection: str, limit: int = settings.qdrant_scroll_limit
) -> Iterable[dict]:
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
