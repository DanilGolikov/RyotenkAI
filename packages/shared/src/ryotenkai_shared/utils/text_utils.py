"""
Generic text extraction utilities.

Functions here are domain-agnostic: they operate on arbitrary Python objects
(str, dict, list, None) and return plain strings. No domain-specific patterns.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def extract_nested_text(value: Any) -> str:
    """Recursively extract a plain string from a nested Python value.

    Handles the common cases produced by HuggingFace datasets and ChatML
    message dicts without importing any domain-specific knowledge.

    Rules:
    - None  → ""
    - str   → value as-is
    - Mapping → looks for keys ``content``, ``text``, ``answer``, ``expected_answer``
                in that order; falls back to str(value)
    - list  → joins non-empty extracted parts with newlines
    - other → str(value)
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("content", "text", "answer", "expected_answer"):
            if key in value:
                return extract_nested_text(value[key])
        return str(value)
    if isinstance(value, list):
        if not value:
            return ""
        parts = [extract_nested_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return str(value)


__all__ = ["extract_nested_text"]
