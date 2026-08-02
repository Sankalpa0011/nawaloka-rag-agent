"""Embedding models, loaded lazily and shared process-wide.

Everything here runs locally through fastembed (ONNX runtime, no PyTorch).
That is a deliberate choice: the system already depends on two rate-limited
LLM providers, and putting embeddings behind a third network quota would mean
a re-index could fail halfway and retrieval latency would ride on someone
else's API. Local inference is ~10-30 ms per query on CPU, costs nothing, and
works offline.

Models
------
dense   BAAI/bge-small-en-v1.5   384-dim, cosine. Small, strong on short
                                 factual passages, and its asymmetric
                                 query/passage prefixes are handled for us by
                                 fastembed's query_embed/passage_embed.
sparse  Qdrant/bm25              Lexical match for exact terms - test codes,
                                 doctor surnames, "Pap Smear". This is what
                                 dense vectors are worst at.
rerank  ms-marco-MiniLM-L-6-v2   Cross-encoder, scores query+passage jointly.
                                 The single largest precision gain available.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Iterable

from app.config import settings
from app.logging_conf import get_logger

log = get_logger(__name__)

_lock = threading.Lock()


def _threads() -> int:
    """ONNX intra-op threads.

    fastembed defaults to letting onnxruntime decide, which under-subscribes on
    this workload - reranking is the dominant cost in a retrieval call and is
    embarrassingly parallel across the candidate batch. One thread is left for
    the event loop so streaming responses are not starved.
    """
    if settings.onnx_threads > 0:
        return settings.onnx_threads
    return max(1, (os.cpu_count() or 2) - 1)


@lru_cache(maxsize=1)
def dense_model():
    from fastembed import TextEmbedding

    with _lock:
        log.info("embedder.loading_dense", model=settings.embedding_model)
        return TextEmbedding(model_name=settings.embedding_model, threads=_threads())


@lru_cache(maxsize=1)
def sparse_model():
    from fastembed import SparseTextEmbedding

    with _lock:
        log.info("embedder.loading_sparse", model=settings.sparse_model)
        return SparseTextEmbedding(model_name=settings.sparse_model, threads=_threads())


@lru_cache(maxsize=1)
def rerank_model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    with _lock:
        log.info("embedder.loading_reranker", model=settings.rerank_model,
                 threads=_threads())
        return TextCrossEncoder(model_name=settings.rerank_model, threads=_threads())


DENSE_DIM = 384


def embed_passages(texts: Iterable[str]) -> list[list[float]]:
    """Embed documents for indexing (uses the passage-side representation)."""
    return [v.tolist() for v in dense_model().passage_embed(list(texts))]


def embed_query(text: str) -> list[float]:
    """Embed a user query (uses the query-side representation)."""
    return next(iter(dense_model().query_embed([text]))).tolist()


def _to_sparse(emb) -> tuple[list[int], list[float]]:
    return [int(i) for i in emb.indices], [float(v) for v in emb.values]


def sparse_passages(texts: Iterable[str]) -> list[tuple[list[int], list[float]]]:
    return [_to_sparse(e) for e in sparse_model().embed(list(texts))]


def sparse_query(text: str) -> tuple[list[int], list[float]]:
    # BM25 query embedding differs from the document side (no term saturation),
    # so use query_embed rather than embed here.
    return _to_sparse(next(iter(sparse_model().query_embed([text]))))


def rerank(query: str, documents: list[str]) -> list[float]:
    """Cross-encoder relevance scores, aligned with *documents* order."""
    if not documents:
        return []
    return [float(s) for s in rerank_model().rerank(query, documents)]


def warm_up() -> None:
    """Pre-load models so the first user request is not slowed by cold start."""
    embed_query("warm up")
    sparse_query("warm up")
    if settings.enable_rerank:
        rerank("warm up", ["warm up"])
    log.info("embedder.warm")
