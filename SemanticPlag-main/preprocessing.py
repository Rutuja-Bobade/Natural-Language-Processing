"""
Text preprocessing: lowercase, tokenization (regex), stopword removal, lemmatization.

Stopwords: sklearn ENGLISH_STOP_WORDS (no download).
Lemmatization: NLTK WordNetLemmatizer when `wordnet` is available (`nltk.download('wordnet')`);
otherwise tokens are left as-is (still valid for TF-IDF and SBERT).
"""

from __future__ import annotations

import re
from typing import Callable, List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_lemmatize: Callable[[str], str] | None = None


def _build_lemmatizer() -> Callable[[str], str]:
    """Prefer NLTK WordNet; gracefully degrade if corpus or SSL is unavailable."""
    lem = None

    try:
        import nltk
        from nltk.stem import WordNetLemmatizer

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)

        try:
            nltk.data.find("corpora/omw-1.4")
        except LookupError:
            nltk.download("omw-1.4", quiet=True)

        lem = WordNetLemmatizer()

    except Exception:
        lem = None

    def lemmatize(w: str) -> str:
        if lem is None:
            return w
        try:
            return lem.lemmatize(w)
        except LookupError:
            return w

    return lemmatize


def _get_lemmatizer() -> Callable[[str], str]:
    global _lemmatize

    if _lemmatize is None:
        _lemmatize = _build_lemmatizer()

    return _lemmatize


def preprocess_text(text: str) -> str:
    """
    Lowercase, tokenize, remove English stopwords, lemmatize when WordNet is present.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens = re.findall(r"[a-z0-9]+", text)

    stop = ENGLISH_STOP_WORDS
    lem = _get_lemmatizer()

    out: List[str] = []

    for w in tokens:
        if w in stop:
            continue
        out.append(lem(w))

    return " ".join(out)


def preprocess_document(raw: str) -> str:
    return preprocess_text(raw)