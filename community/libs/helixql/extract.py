"""Extraction helpers — pull HelixQL query/schema text out of plugin samples.

Two original helpers (``extract_query_text``, ``extract_schema_block``)
move here verbatim from the old ``src/utils/domains/helixql.py``.
:func:`extract_schema_and_query` is a new helper that consolidates an
inline pattern duplicated across two plugins
(``helixql_gold_syntax_backend`` and ``helixql_generated_syntax_backend``)
into a single call.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from src.utils.text_utils import extract_nested_text

# Schema fences look like ```helixschema\n...\n``` in plugin prompts.
# Compile once at import time — the regex is hot-path on validation
# and reward callbacks.
SCHEMA_FENCE_RE = re.compile(r"```helixschema\n([\s\S]*?)\n```", flags=re.MULTILINE)

# Domain alias: plugin authors reach for ``extract_query_text`` because
# that's what they're extracting; ``extract_nested_text`` is the
# generic platform helper. Same callable, both names exported.
extract_query_text = extract_nested_text


def extract_schema_block(text: str) -> str:
    """Pull the contents of the first ``helixschema`` fence out of ``text``.

    Returns ``""`` when no fence matches or the input is empty/None.
    Surrounding whitespace inside the fence is stripped.
    """
    match = SCHEMA_FENCE_RE.search(text or "")
    if not match:
        return ""
    return (match.group(1) or "").strip()


def extract_schema_and_query(
    sample: Mapping[str, Any] | Any,
    *,
    prompt_keys: tuple[str, ...] = ("prompt", "user", "messages"),
    query_keys: tuple[str, ...] = ("reference_answer", "expected", "completion"),
) -> tuple[str, str]:
    """Resolve ``(schema, query)`` from a plugin sample mapping.

    Two plugins repeated this dispatch by hand:

    * ``helixql_gold_syntax_backend`` — looks at messages, then falls
      back to ``reference_answer`` / dataset fields.
    * ``helixql_generated_syntax_backend`` — same, with a slightly
      different field-name preference.

    The unified helper takes the same approach the original code did —
    look up a prompt-shaped value (string, list, nested chat-messages)
    via :func:`extract_nested_text`, then run :func:`extract_schema_block`
    on it for the schema; for the query value, look up the first
    matching ``query_keys`` entry and run :func:`extract_nested_text`
    on that. Either side returning ``""`` is a soft signal — callers
    decide whether to skip the sample or treat it as a hard error.

    Parameters
    ----------
    sample
        Anything indexable by string keys (typical: dict, dataset row).
        Non-mapping inputs are tolerated by reading attributes lazily.
    prompt_keys
        Where to look for the prompt blob that *contains* the schema
        fence. Tried in order; first hit wins.
    query_keys
        Where to look for the gold/expected query text. Tried in order;
        first hit wins.
    """
    schema_source: Any = ""
    for key in prompt_keys:
        candidate = _get(sample, key)
        if candidate:
            schema_source = candidate
            break
    schema = extract_schema_block(extract_nested_text(schema_source))

    query_source: Any = ""
    for key in query_keys:
        candidate = _get(sample, key)
        if candidate:
            query_source = candidate
            break
    query = extract_nested_text(query_source)

    return schema, query


def _get(sample: Any, key: str) -> Any:
    """Pluck ``key`` out of a mapping or attribute container, ``None`` if absent."""
    if isinstance(sample, Mapping):
        return sample.get(key)
    return getattr(sample, key, None)


__all__ = [
    "SCHEMA_FENCE_RE",
    "extract_query_text",
    "extract_schema_and_query",
    "extract_schema_block",
]
