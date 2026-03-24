from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from src.utils.text_utils import extract_nested_text

# Semantic scoring weights (WPS432)
_SCORE_WEIGHT_SEQUENCE = 0.55
_SCORE_WEIGHT_JACCARD = 0.25
_SCORE_WEIGHT_EXACT = 0.20
_PENALTY_MAX = 0.45
_PENALTY_PER_ERROR = 0.15
_NEAR_MATCH_THRESHOLD = 0.8

QUERY_LINE_RE = re.compile(r"^\s*QUERY\s+", flags=re.MULTILINE)
QUERY_SIG_RE = re.compile(r"^\s*QUERY\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*=>", flags=re.MULTILINE)
SCHEMA_FENCE_RE = re.compile(r"```helixschema\n([\s\S]*?)\n```", flags=re.MULTILINE)
EXCLUSION_RE = re.compile(r"::!\{([^}]*)\}")
RERANK_RRF_CALL_RE = re.compile(r"::RerankRRF\s*\(([^)]*)\)", flags=re.MULTILINE)
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Backward-compatible alias — domain plugins keep importing this name.
extract_query_text = extract_nested_text


def extract_schema_block(text: str) -> str:
    match = SCHEMA_FENCE_RE.search(text or "")
    if not match:
        return ""
    return (match.group(1) or "").strip()


def normalize_query_text(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    lines = [" ".join(line.strip().split()) for line in stripped.splitlines() if line.strip()]
    return "\n".join(lines)


def _parse_query_params(query_text: str) -> dict[str, str]:
    match = QUERY_SIG_RE.search(query_text or "")
    if not match:
        return {}
    params_str = (match.group(2) or "").strip()
    if not params_str:
        return {}
    out: dict[str, str] = {}
    for part in params_str.split(","):
        chunk = part.strip()
        if not chunk or ":" not in chunk:
            continue
        name, type_name = chunk.split(":", 1)
        name = name.strip()
        type_name = type_name.strip()
        if name and type_name:
            out[name] = type_name
    return out


def _has_embed_vector_misuse(query_text: str) -> bool:
    params = _parse_query_params(query_text)
    if not params:
        return False
    vector_params = [name for name, type_name in params.items() if type_name.replace(" ", "") == "[F64]"]
    return any(re.search(rf"\bEmbed\s*\(\s*{re.escape(name)}\s*\)", query_text or "") for name in vector_params)


def _has_invalid_rerank_rrf_args(query_text: str) -> bool:
    for match in RERANK_RRF_CALL_RE.finditer(query_text or ""):
        args = (match.group(1) or "").strip()
        if not args:
            continue
        parts = [part.strip() for part in args.split(",") if part.strip()]
        if len(parts) != 1:
            return True
        if re.match(r"^k\s*:\s*.+$", parts[0]):
            continue
        return True
    return False


def _extract_required_exclusions_from_prompt(user_text: str) -> list[str]:
    if not user_text or re.search(r"(exclude|exclusion)", user_text, flags=re.IGNORECASE) is None:
        return []
    lines = [line for line in user_text.splitlines() if re.search(r"(exclude|exclusion)", line, flags=re.IGNORECASE)]
    fields = re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", "\n".join(lines))
    seen: set[str] = set()
    ordered: list[str] = []
    for field in fields:
        if field not in seen:
            seen.add(field)
            ordered.append(field)
    return ordered


def _extract_excluded_fields(query_text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in EXCLUSION_RE.finditer(query_text or ""):
        inner = (match.group(1) or "").strip()
        for field in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", inner):
            if field not in seen:
                seen.add(field)
                ordered.append(field)
    return ordered


def hard_eval_errors(user_text: str, query_text: str) -> list[str]:
    errors: list[str] = []
    output = query_text or ""
    if not output.lstrip().startswith("QUERY"):
        errors.append("not_starting_with_query")
    if "```" in output:
        errors.append("contains_markdown_fence")
    if ":=" in output:
        errors.append("contains_colon_equals")
    if not QUERY_LINE_RE.search(output):
        errors.append("missing_query_keyword")
    if _has_embed_vector_misuse(output):
        errors.append("embed_vector_misuse")
    if _has_invalid_rerank_rrf_args(output):
        errors.append("rerank_rrf_invalid_args")

    required_exclusions = _extract_required_exclusions_from_prompt(user_text)
    if required_exclusions:
        excluded = set(_extract_excluded_fields(output))
        if not set(required_exclusions).issubset(excluded):
            errors.append("missing_required_exclusions")
    return errors


def semantic_match_details(*, candidate: str, expected: str, user_text: str = "") -> dict[str, Any]:
    candidate_norm = normalize_query_text(candidate)
    expected_norm = normalize_query_text(expected)
    if not candidate_norm or not expected_norm:
        return {
            "score": 0.0,
            "exact_match": False,
            "near_match": False,
            "sequence_ratio": 0.0,
            "token_jaccard": 0.0,
            "hard_eval_pass": False,
            "hard_eval_errors": ["empty_candidate_or_expected"],
        }

    exact_match = candidate_norm == expected_norm
    candidate_lower = candidate_norm.lower()
    expected_lower = expected_norm.lower()
    sequence_ratio = SequenceMatcher(a=candidate_lower, b=expected_lower).ratio()

    candidate_tokens = set(_TOKEN_RE.findall(candidate_lower))
    expected_tokens = set(_TOKEN_RE.findall(expected_lower))
    union = candidate_tokens | expected_tokens
    token_jaccard = (len(candidate_tokens & expected_tokens) / len(union)) if union else 0.0

    hard_errors = hard_eval_errors(user_text, candidate)
    hard_eval_pass = not hard_errors
    penalty = min(_PENALTY_MAX, _PENALTY_PER_ERROR * len(hard_errors))

    base_score = (
        (_SCORE_WEIGHT_SEQUENCE * sequence_ratio)
        + (_SCORE_WEIGHT_JACCARD * token_jaccard)
        + (_SCORE_WEIGHT_EXACT if exact_match else 0.0)
    )
    score = max(0.0, min(1.0, base_score - penalty))
    near_match = score >= _NEAR_MATCH_THRESHOLD and hard_eval_pass

    if exact_match:
        score = 1.0
        near_match = True

    return {
        "score": round(score, 4),
        "exact_match": exact_match,
        "near_match": near_match,
        "sequence_ratio": round(sequence_ratio, 4),
        "token_jaccard": round(token_jaccard, 4),
        "hard_eval_pass": hard_eval_pass,
        "hard_eval_errors": hard_errors,
    }


__all__ = [
    "extract_query_text",
    "extract_schema_block",
    "semantic_match_details",
]
