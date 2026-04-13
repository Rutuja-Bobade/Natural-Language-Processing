#!/usr/bin/env python3
"""
Semantic plagiarism checker: hybrid TF-IDF + Sentence-BERT.

Usage:
  python main.py --doc-a path/to/a.txt --doc-b path/to/b.txt
  python main.py --doc-a "short text one" --doc-b "short text two"

Use --adk to run the same analysis through Google ADK SequentialAgent + Runner.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from similarity import (
    SentencePair,
    W_BERT,
    W_TFIDF,
    PlagiarismReport,
    run_plagiarism_analysis,
)
from utils import read_document


def format_report(report: PlagiarismReport) -> str:
    lines = [
        "",
        "========== SEMANTIC PLAGIARISM REPORT ==========",
        f"Document TF-IDF (lexical):     {report.tfidf_doc:.4f}",
        f"Document SBERT (semantic):     {report.bert_doc:.4f}",
        f"Document hybrid ({W_TFIDF:.1f}·TF-IDF + {W_BERT:.1f}·SBERT): {report.hybrid_doc:.4f}",
    ]

    if report.section_weighted_hybrid is not None:
        lines.append(
            f"Section-weighted hybrid:       {report.section_weighted_hybrid:.4f}  (used for % below)"
        )

    lines.append(f"Estimated similarity / risk: {report.plagiarism_percent}%")
    lines.append("")
    lines.append("--- Top similar sentence pairs (doc A vs doc B) ---")

    for i, p in enumerate(report.top_pairs, 1):
        lines.append(
            f"  [{i}] hybrid={p.hybrid:.4f}  tfidf={p.tfidf:.4f}  bert={p.bert:.4f}"
        )
        lines.append(
            f"      A: {p.text_a[:200]}{'…' if len(p.text_a) > 200 else ''}"
        )
        lines.append(
            f"      B: {p.text_b[:200]}{'…' if len(p.text_b) > 200 else ''}"
        )

    lines.append("==============================================")
    lines.append("")

    return "\n".join(lines)


def _report_from_state(obj: object) -> PlagiarismReport | None:
    """Restore PlagiarismReport from ADK session (dataclass may be dict)."""
    if isinstance(obj, PlagiarismReport):
        return obj

    if not isinstance(obj, dict):
        return None

    pairs_raw = obj.get("top_pairs") or []
    pairs = [SentencePair(**p) for p in pairs_raw]

    return PlagiarismReport(
        tfidf_doc=float(obj["tfidf_doc"]),
        bert_doc=float(obj["bert_doc"]),
        hybrid_doc=float(obj["hybrid_doc"]),
        plagiarism_percent=float(obj["plagiarism_percent"]),
        top_pairs=pairs,
        section_weighted_hybrid=obj.get("section_weighted_hybrid"),
        meta=dict(obj.get("meta") or {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid semantic plagiarism detection"
    )

    parser.add_argument(
        "--doc-a",
        required=True,
        help="Plain text or path to .txt (document A)",
    )
    parser.add_argument(
        "--doc-b",
        required=True,
        help="Plain text or path to .txt (document B)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top sentence pairs",
    )
    parser.add_argument(
        "--adk",
        action="store_true",
        help="Run analysis through Google ADK SequentialAgent + Runner",
    )

    args = parser.parse_args()

    text_a = read_document(args.doc_a)
    text_b = read_document(args.doc_b)

    if args.adk:
        from adk_orchestrator import run_pipeline_with_runner

        state = asyncio.run(
            run_pipeline_with_runner(
                text_a,
                text_b,
                top_k=args.top_k,
            )
        )

        report = _report_from_state(state.get("report"))

        if report is None:
            print("ADK run did not produce a report.", file=sys.stderr)
            sys.exit(1)
    else:
        report = run_plagiarism_analysis(
            text_a,
            text_b,
            top_k=args.top_k,
        )

    print(format_report(report))


if __name__ == "__main__":
    main()