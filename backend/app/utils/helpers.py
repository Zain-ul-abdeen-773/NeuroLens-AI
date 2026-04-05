"""
NeuroLens Utility Helpers
Common utility functions used across the application.
"""

import re
import hashlib
import json
from typing import Any, Dict, List, Optional
from datetime import datetime


def clean_text(text: str) -> str:
    """
    Comprehensive text cleaning pipeline.

    Removes HTML tags, URLs, email addresses, extra whitespace,
    and normalizes unicode characters.
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " [URL] ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " [EMAIL] ", text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_hash(text: str) -> str:
    """Generate a stable hash for caching purposes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chunk_text(text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
    """
    Split long text into overlapping chunks for processing.

    Args:
        text: Input text
        max_length: Maximum words per chunk
        overlap: Number of overlapping words between chunks
    """
    words = text.split()
    if len(words) <= max_length:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_length
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks


def compute_text_statistics(text: str) -> Dict[str, Any]:
    """Compute basic text statistics for meta-features."""
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
        "avg_sentence_length": len(words) / max(len(sentences), 1),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "punctuation_density": sum(1 for c in text if c in ".,!?;:-") / max(len(words), 1),
    }


def timestamp_now() -> str:
    """Return ISO format timestamp."""
    return datetime.utcnow().isoformat() + "Z"


def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely parse JSON string."""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else {}
