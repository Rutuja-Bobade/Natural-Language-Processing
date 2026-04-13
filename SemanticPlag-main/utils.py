"""Small helpers: load text from files or strings, split sentences, heading detection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

# Common academic section headings (case-insensitive match, line-start)
SECTION_PATTERNS = [
    (r"^\s*introduction\s*:?\s*$", "intro", 0.15),
    (r"^\s*(methodology|methods?)\s*:?\s*$", "method", 0.35),
    (r"^\s*(results?|findings?)\s*:?\s*$", "results", 0.25),
    (r"^\s*(discussion|conclusion|conclusions?)\s*:?\s*$", "conclusion", 0.10),
]


def read_document(path_or_text: str) -> str:
    """Load plain text from a file path or return the string as-is."""
    p = Path(path_or_text)

    if p.is_file() and p.suffix.lower() == ".txt":
        return p.read_text(encoding="utf-8", errors="replace")

    return path_or_text


def split_sentences(text: str) -> List[str]:
    """Simple sentence split (good enough for demos; avoids heavy deps)."""
    text = text.strip()

    if not text:
        return []

    # Split on . ! ? when followed by space or end
    parts = re.split(r"(?<=[.!?])\s+", text)

    return [s.strip() for s in parts if s.strip()]


def detect_sections(text: str) -> List[Tuple[str, str, float, int]]:
    """
    Return list of (matched_line, key, weight, line_index).
    If no headings found, returns empty list.
    """
    lines = text.splitlines()
    found: List[Tuple[str, str, float, int]] = []

    for i, line in enumerate(lines):
        for pattern, key, weight in SECTION_PATTERNS:
            if re.match(pattern, line, re.IGNORECASE):
                found.append((line.strip(), key, weight, i))
                break

    return found


def section_weight_map(text: str) -> dict[str, float] | None:
    """If headings detected, map section keys to weights (normalized). Else None."""
    secs = detect_sections(text)

    if not secs:
        return None

    w: dict[str, float] = {}

    for _, key, weight, _ in secs:
        w[key] = weight

    s = sum(w.values())

    if s <= 0:
        return None

    return {k: v / s for k, v in w.items()}