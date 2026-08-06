"""Cross-encoder reranking, as a LlamaIndex node postprocessor.

Wraps our fastembed ONNX cross-encoder instead of LlamaIndex's
SentenceTransformerRerank (which would pull in torch). Scores are passed
through a sigmoid so retrieval_score_floor is a portable 0-1 value.
"""

from __future__ import annotations

import math

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from app.config import settings
from app.logging_conf import get_logger

log = get_logger(__name__)

# How much chunk body the cross-encoder sees (~its 512-token window).
RERANK_DOC_CHARS = settings.rerank_doc_chars


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class FastEmbedRerank(BaseNodePostprocessor):
    """Rerank with the fastembed ONNX cross-encoder, then drop weak nodes."""

    top_n: int = 5
    score_floor: float = 0.0
    dropped: int = 0

    @classmethod
    def class_name(cls) -> str:
        return "FastEmbedRerank"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes or query_bundle is None:
            return nodes[: self.top_n]

        from app.retrieval.embedder import rerank

        # Score breadcrumb + body together, matching how it was embedded.
        docs = []
        for node in nodes:
            breadcrumb = node.node.metadata.get("breadcrumb_text", "")
            body = node.node.metadata.get("body") or node.node.get_content()
            docs.append(f"{breadcrumb}\n{body[: settings.rerank_doc_chars]}")

        for node, raw in zip(nodes, rerank(query_bundle.query_str, docs)):
            node.score = sigmoid(raw)

        nodes.sort(key=lambda n: n.score or 0.0, reverse=True)

        kept = []
        for node in nodes:
            if (node.score or 0.0) >= self.score_floor:
                kept.append(node)

        # Recorded (not just logged) so the trace panel can show how many were dropped.
        self.dropped = len(nodes) - len(kept)
        return kept[: self.top_n]


def build_reranker(top_k: int | None = None, score_floor: float | None = None):
    return FastEmbedRerank(
        top_n=top_k if top_k is not None else settings.retrieval_top_k,
        score_floor=(
            score_floor if score_floor is not None else settings.retrieval_score_floor
        ),
    )
