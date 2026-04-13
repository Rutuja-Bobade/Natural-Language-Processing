"""Lexical similarity via TF-IDF + cosine similarity (scikit-learn)."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Cosine similarity between two preprocessed document strings.
    Returns a float in [0, 1] when vectors are non-empty; 0.0 if degenerate.
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
    )

    try:
        matrix = vectorizer.fit_transform([text_a, text_b])
    except ValueError:
        return 0.0

    sim = cosine_similarity(matrix[0:1], matrix[1:2])[0, 0]

    return float(max(0.0, min(1.0, sim)))


def tfidf_cosine_pair(sentence_a: str, sentence_b: str) -> float:
    """TF-IDF cosine for two short strings (sentence-level)."""
    return tfidf_cosine_similarity(sentence_a, sentence_b)