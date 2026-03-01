"""Embedding model for semantic memory search.

Uses fastembed (ONNX-based, no PyTorch) with BAAI/bge-small-en-v1.5 (33MB, 384-dim).
Runs on CPU — doesn't compete with Qwen for GPU memory.
"""

import logging
from typing import Generator

import numpy as np
from fastembed import TextEmbedding

from .config import settings

logger = logging.getLogger("memory-server.embeddings")

# Singleton
_embedder: "Embedder | None" = None


class Embedder:
    """Thin wrapper around fastembed's TextEmbedding with numpy cosine similarity."""

    def __init__(self) -> None:
        self._model: TextEmbedding | None = None

    def warm_up(self) -> None:
        """Pre-load the model (triggers download on first call)."""
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._model = TextEmbedding(model_name=settings.embedding_model)
            logger.info("Embedding model ready")

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            self.warm_up()
        return self._model  # type: ignore[return-value]

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts (documents/memories)."""
        # fastembed returns a generator of numpy arrays
        embeddings: Generator[np.ndarray, None, None] = self.model.embed(
            texts, batch_size=settings.embedding_batch_size
        )
        return [e.astype(np.float32) for e in embeddings]

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Uses query-optimized encoding for bge models."""
        # fastembed's query_embed adds the "query: " prefix for asymmetric models
        results = list(self.model.query_embed(query))
        return results[0].astype(np.float32)

    def to_bytes(self, vec: np.ndarray) -> bytes:
        """Serialize embedding to bytes for SQLite BLOB storage."""
        return vec.astype(np.float32).tobytes()

    def from_bytes(self, blob: bytes) -> np.ndarray:
        """Deserialize embedding from SQLite BLOB."""
        return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity_batch(query_vec: np.ndarray, candidate_matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and a matrix of candidates.

    Args:
        query_vec: (D,) query embedding
        candidate_matrix: (N, D) matrix of candidate embeddings

    Returns:
        (N,) similarity scores in [-1, 1]
    """
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True) + 1e-10
    candidate_norms = candidate_matrix / norms
    return candidate_norms @ query_norm


def get_embedder() -> Embedder:
    """Get the singleton embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
