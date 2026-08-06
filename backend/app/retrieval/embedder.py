"""Embedding models (dense, sparse, rerank), loaded lazily via fastembed (ONNX, local)."""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Iterable

from llama_index.core.base.embeddings.base import BaseEmbedding

from app.config import settings
from app.logging_conf import get_logger

log = get_logger(__name__)

_lock = threading.Lock()


# ONNX intra-op thread count; leaves one thread free for the event loop.
def _threads() -> int:
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


# Passage side of the model, for indexing documents.
def embed_passages(texts: Iterable[str]) -> list[list[float]]:
    out = []
    for v in dense_model().passage_embed(list(texts)):
        out.append(v.tolist())
    return out


# Query side of the model, for a user query.
def embed_query(text: str) -> list[float]:
    return next(iter(dense_model().query_embed([text]))).tolist()


def _to_sparse(emb) -> tuple[list[int], list[float]]:
    return [int(i) for i in emb.indices], [float(v) for v in emb.values]


def sparse_passages(texts: Iterable[str]) -> list[tuple[list[int], list[float]]]:
    out = []
    for e in sparse_model().embed(list(texts)):
        out.append(_to_sparse(e))
    return out


def sparse_query(text: str) -> tuple[list[int], list[float]]:
    # query_embed, not embed - BM25 treats the query side differently (no term saturation).
    return _to_sparse(next(iter(sparse_model().query_embed([text]))))


# Relevance scores, same order as `documents`.
def rerank(query: str, documents: list[str]) -> list[float]:
    if not documents:
        return []
    scores = []
    for s in rerank_model().rerank(query, documents):
        scores.append(float(s))
    return scores


class SharedFastEmbed(BaseEmbedding):
    """LlamaIndex embedding wrapper that reuses the models loaded above, instead of
    LlamaIndex loading its own second copy."""

    @classmethod
    def class_name(cls) -> str:
        return "SharedFastEmbed"

    def _get_query_embedding(self, query: str) -> list[float]:
        return embed_query(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return embed_passages([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return embed_passages(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)


@lru_cache(maxsize=1)
def llama_embedding() -> SharedFastEmbed:
    return SharedFastEmbed()


# Load the models (and optionally the retrieval index) up front to avoid a cold start.
def warm_up(retrieval: bool = True) -> None:
    embed_query("warm up")
    sparse_query("warm up")
    if settings.enable_rerank:
        rerank("warm up", ["warm up"])

    if retrieval:
        try:
            from app.retrieval.search import get_index

            get_index(settings.qdrant_collection)
        except Exception as exc:
            # Qdrant not up yet, or collection not seeded - not fatal.
            log.warning("embedder.retrieval_warm_failed", error=str(exc)[:160])

    log.info("embedder.warm")
