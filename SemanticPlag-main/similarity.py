"""
Hybrid similarity, sentence-pair mining, optional section weighting, report object.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bert_module import cosine_from_embeddings, encode_sentences
from tfidf_module import tfidf_cosine_pair

# Hybrid blend: lexical + semantic (spec: 0.4 TF-IDF + 0.6 SBERT)
W_TFIDF = 0.4
W_BERT = 0.6


def hybrid_score(tfidf_sim: float, bert_sim: float) -> float:
    return W_TFIDF * tfidf_sim + W_BERT * bert_sim


@dataclass
class SentencePair:
    idx_a: int
    idx_b: int
    text_a: str
    text_b: str
    tfidf: float
    bert: float
    hybrid: float


@dataclass
class PlagiarismReport:
    """Result of comparing document A (e.g. source) to document B (suspect)."""

    tfidf_doc: float
    bert_doc: float
    hybrid_doc: float
    plagiarism_percent: float
    top_pairs: List[SentencePair] = field(default_factory=list)
    section_weighted_hybrid: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def _heading_key(line: str) -> Optional[str]:
    from utils import SECTION_PATTERNS

    for pattern, key, _w in SECTION_PATTERNS:
        if re.match(pattern, line, re.IGNORECASE):
            return key
    return None


def _split_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split text by heading lines (Introduction, Methodology, ...).
    Returns [(section_key, body_text), ...].
    """
    lines = text.splitlines()

    if not lines:
        return [("all", "")]

    out: List[Tuple[str, str]] = []
    current_key = "preamble"
    buffer: List[str] = []
    saw_heading = False

    for line in lines:
        h = _heading_key(line)

        if h:
            saw_heading = True
            out.append((current_key, "\n".join(buffer).strip()))
            buffer = []
            current_key = h
        else:
            buffer.append(line)

    out.append((current_key, "\n".join(buffer).strip()))

    if not saw_heading:
        return [("all", text.strip())]

    return out


def _section_weights() -> Dict[str, float]:
    from utils import SECTION_PATTERNS

    w: Dict[str, float] = {}

    for _p, key, weight in SECTION_PATTERNS:
        w[key] = weight

    return w


def section_weighted_hybrid(
    text_a_raw: str,
    text_b_raw: str,
    pre_a: str,
    pre_b: str,
) -> Optional[float]:
    """
    If both documents contain detected section headings, compute a weighted
    average of per-section hybrid scores (TF-IDF + SBERT on section bodies).
    """
    from bert_module import document_embedding_similarity
    from tfidf_module import tfidf_cosine_similarity

    secs_a = _split_into_sections(text_a_raw)
    secs_b = _split_into_sections(text_b_raw)

    # Need at least one recognized section key besides 'all' / 'preamble'
    keys_a = {k for k, _ in secs_a if k not in ("all", "preamble")}
    keys_b = {k for k, _ in secs_b if k not in ("all", "preamble")}

    common = keys_a & keys_b

    if not common:
        return None

    weights = _section_weights()
    num = 0.0
    den = 0.0

    for key in common:
        body_a = next((b for k, b in secs_a if k == key), "")
        body_b = next((b for k, b in secs_b if k == key), "")

        if not body_a.strip() or not body_b.strip():
            continue

        # Preprocess section bodies the same way as full docs
        from preprocessing import preprocess_text

        pa = preprocess_text(body_a)
        pb = preprocess_text(body_b)

        t = tfidf_cosine_similarity(pa, pb)
        s = document_embedding_similarity(pa, pb)
        h = hybrid_score(t, s)

        wgt = weights.get(key, 0.25)
        num += wgt * h
        den += wgt

    if den <= 0:
        return None

    return num / den


def top_sentence_pairs(
    raw_sents_a: Sequence[str],
    raw_sents_b: Sequence[str],
    pre_sents_a: Sequence[str],
    pre_sents_b: Sequence[str],
    top_k: int = 10,
) -> List[SentencePair]:
    """Cross-compare sentences; rank by hybrid score."""
    if not raw_sents_a or not raw_sents_b:
        return []

    emb_a = encode_sentences(list(pre_sents_a))
    emb_b = encode_sentences(list(pre_sents_b))

    pairs: List[SentencePair] = []

    for i in range(len(raw_sents_a)):
        for j in range(len(raw_sents_b)):
            b_sim = cosine_from_embeddings(emb_a[i], emb_b[j])
            t_sim = tfidf_cosine_pair(pre_sents_a[i], pre_sents_b[j])
            h = hybrid_score(t_sim, b_sim)

            pairs.append(
                SentencePair(
                    idx_a=i,
                    idx_b=j,
                    text_a=raw_sents_a[i],
                    text_b=raw_sents_b[j],
                    tfidf=t_sim,
                    bert=b_sim,
                    hybrid=h,
                )
            )

    pairs.sort(key=lambda p: p.hybrid, reverse=True)
    return pairs[:top_k]


def run_plagiarism_analysis(
    raw_a: str,
    raw_b: str,
    top_k: int = 10,
) -> PlagiarismReport:
    """End-to-end: preprocess, document scores, sentence pairs, optional section blend."""
    from bert_module import document_embedding_similarity
    from preprocessing import preprocess_text
    from tfidf_module import tfidf_cosine_similarity
    from utils import split_sentences

    pre_a = preprocess_text(raw_a)
    pre_b = preprocess_text(raw_b)

    t_doc = tfidf_cosine_similarity(pre_a, pre_b)
    b_doc = document_embedding_similarity(pre_a, pre_b)
    h_doc = hybrid_score(t_doc, b_doc)

    sec_hw = section_weighted_hybrid(raw_a, raw_b, pre_a, pre_b)
    display_hybrid = sec_hw if sec_hw is not None else h_doc

    raw_sa = split_sentences(raw_a)
    raw_sb = split_sentences(raw_b)

    pre_sa = [preprocess_text(s) for s in raw_sa]
    pre_sb = [preprocess_text(s) for s in raw_sb]

    pairs = top_sentence_pairs(
        raw_sa,
        raw_sb,
        pre_sa,
        pre_sb,
        top_k=top_k,
    )

    plag_pct = round(float(display_hybrid) * 100.0, 2)

    return PlagiarismReport(
        tfidf_doc=t_doc,
        bert_doc=b_doc,
        hybrid_doc=h_doc,
        plagiarism_percent=plag_pct,
        top_pairs=pairs,
        section_weighted_hybrid=sec_hw,
        meta={"used_section_weighting": sec_hw is not None},
    )