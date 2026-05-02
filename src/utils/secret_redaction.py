"""Centralized secret-redaction filter (PR-B RP4 / Phase 1 R5).

Single source of truth for "what counts as a secret in our logs". Used
on every code path where untrusted text (trainer stdout/stderr, SSH
command echoes, MLflow event payloads sent through the WS bridge,
postmortem dumps) might leak credentials embedded in the user's
environment.

Patterns are intentionally conservative — false-negatives (a leaked
custom secret) are worse than false-positives (mangling a benign string
that happened to look like a token). Add a pattern when in doubt.

The list mirrors the legacy ``ssh_client._SECRET_PATTERNS`` from before
PR-B; that module now imports from here so we have one rule set instead
of two slowly diverging copies.
"""

from __future__ import annotations

import os
import re

# Replacement for ``KEY=value`` style hits. Keeps the key name visible
# (so operators can tell *which* secret was redacted) while masking the
# value. Three groups: ``(prefix)(value)(suffix)`` — quoting may differ
# between matches.
_MASK_KV_REPL = r"\1***\3"

# Replacement for token-shape hits (e.g. ``hf_xxxxx``). Keeps the prefix
# so operators can identify the token *kind* without exposing the body.
_MASK_PREFIX_REPL = r"\1***"

# Default redaction patterns. Order does not matter (each is applied in
# turn against the residue of prior substitutions). Each entry is
# ``(compiled_pattern, replacement_template)``.
_DEFAULT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # KEY=value pairs commonly seen in env-file dumps and trainer
    # ``os.environ`` echoes.
    (re.compile(r'(\bHF_TOKEN\b\s*=\s*["\']?)([^"\s\n]+)(["\']?)'), _MASK_KV_REPL),
    (re.compile(r'(\bHF_HUB_TOKEN\b\s*=\s*["\']?)([^"\s\n]+)(["\']?)'), _MASK_KV_REPL),
    (re.compile(r'(\bAPI_KEY\b\s*=\s*["\']?)([^"\s\n]+)(["\']?)'), _MASK_KV_REPL),
    (re.compile(r'(\bSECRET\b\s*=\s*["\']?)([^"\s\n]+)(["\']?)'), _MASK_KV_REPL),
    (re.compile(r'(\bPASSWORD\b\s*=\s*["\']?)([^"\s\n]+)(["\']?)'), _MASK_KV_REPL),
    (re.compile(r'(\bRUNPOD_API_KEY\b\s*=\s*["\']?)([^"\s\n]+)(["\']?)'), _MASK_KV_REPL),
    # Token-shape patterns (occur in error tracebacks, debug prints, etc.)
    # without being part of a KEY=value assignment.
    (re.compile(r"(hf_)[A-Za-z0-9]+"), _MASK_PREFIX_REPL),
    (re.compile(r"(sk-)[A-Za-z0-9_-]{16,}"), _MASK_PREFIX_REPL),  # OpenAI
    (re.compile(r"(rpa_)[A-Za-z0-9_-]+"), _MASK_PREFIX_REPL),     # RunPod API key prefix
)

# Operators can extend the redaction list at runtime via the env var
# ``RYOTENKAI_REDACT_PATTERNS``. Format: comma-separated regexes; each
# regex is wrapped in a group so the whole match is replaced with
# ``***``. Bad regexes are silently ignored — failing to redact is
# safer than crashing the post-mortem path that surfaces the diagnostic
# in the first place.
_EXTRA_ENV = "RYOTENKAI_REDACT_PATTERNS"


def _load_extra_patterns() -> tuple[tuple[re.Pattern[str], str], ...]:
    raw = os.environ.get(_EXTRA_ENV, "").strip()
    if not raw:
        return ()
    extras: list[tuple[re.Pattern[str], str]] = []
    for part in raw.split(","):
        pattern = part.strip()
        if not pattern:
            continue
        try:
            extras.append((re.compile(pattern), "***"))
        except re.error:
            continue
    return tuple(extras)


def redact_secrets(text: str) -> str:
    """Mask sensitive substrings in ``text`` so it is safe to log, push
    over WS, or embed in a control-plane event payload.

    Returns the redacted text unchanged when ``text`` is empty.

    The rule set is intentionally fixed (no per-call overrides): every
    consumer must trust the same baseline. Operators wanting site-local
    rules can add them via ``RYOTENKAI_REDACT_PATTERNS`` (see module
    docstring).
    """
    if not text:
        return text

    patterns = _DEFAULT_PATTERNS + _load_extra_patterns()
    for pattern, replacement in patterns:
        text = pattern.sub(replacement, text)
    return text


__all__ = ["redact_secrets"]
