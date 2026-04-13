"""Semantic similarity with Sentence-BERT (all-MiniLM-L6-v2)."""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load model once (cached). First call downloads weights."""
    return SentenceTransformer(MODEL_NAME)


def encode_sentences(sentences: List[str]) -> np.ndarray:
    """Return L2-normalized sentence embeddings (n_sentences, dim)."""
    if not sentences:
        return np.zeros((0, 384), dtype=np.float32)

    model = get_model()

    emb = model.encode(
        sentences,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    return emb.astype(np.float32)


def cosine_from_embeddings(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for single normalized vectors (dot product)."""
    if a.size == 0 or b.size == 0:
        return 0.0

    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def document_embedding_similarity(text_a: str, text_b: str) -> float:
    """Encode whole documents as single strings; return cosine similarity."""
    if not text_a.strip() or not text_b.strip():
        return 0.0

    emb = encode_sentences([text_a, text_b])
    return cosine_from_embeddings(emb[0], emb[1])