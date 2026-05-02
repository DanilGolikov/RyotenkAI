"""Unit tests for src.utils.secret_redaction.

Centralized redaction is the only thing that stops trainer stdout/stderr
(now embedded in the ``trainer_exited`` WS payload via PR-B) from
leaking HF tokens, OpenAI keys, RunPod credentials etc. to whoever has
WS access. The tests below cover each pattern explicitly and the
override surface (``RYOTENKAI_REDACT_PATTERNS``).
"""

from __future__ import annotations

import pytest

from src.utils.secret_redaction import redact_secrets

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Positive: each default pattern actually masks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "must_not_contain"),
    [
        ("HF_TOKEN=hf_realtokenABC123", "hf_realtokenABC123"),
        ('HF_TOKEN="hf_quoted"', "hf_quoted"),
        ("HF_HUB_TOKEN=hf_xxx", "hf_xxx"),
        ("API_KEY=topsecret", "topsecret"),
        ("SECRET=hush", "hush"),
        ("PASSWORD=p4ssw0rd", "p4ssw0rd"),
        ("RUNPOD_API_KEY=rpa_abcDEF123", "rpa_abcDEF123"),
        # token-shape (no KEY=value framing)
        ("logging hf_AAAAAAAAAAA mid-line", "hf_AAAAAAAAAAA"),
        ("openai sk-1234567890abcdef leaked", "sk-1234567890abcdef"),
        ("RunPod rpa_abcdef123 leaked", "rpa_abcdef123"),
    ],
)
def test_default_patterns_mask_known_secrets(raw: str, must_not_contain: str):
    redacted = redact_secrets(raw)
    assert must_not_contain not in redacted
    assert "***" in redacted


def test_key_name_preserved_after_redaction():
    """Operator should still see *which* secret was redacted (KEY name kept,
    value masked). Helps post-mortem triage without exposing the value."""
    redacted = redact_secrets("HF_TOKEN=hf_supersecret123")
    assert "HF_TOKEN" in redacted
    assert "hf_supersecret123" not in redacted


def test_token_prefix_preserved_for_kind_visibility():
    """``hf_`` / ``sk-`` / ``rpa_`` prefixes are kept so we know what kind of
    token leaked, but the body is masked."""
    redacted = redact_secrets("hf_AAAABBBBCCCC")
    assert redacted.startswith("hf_")
    assert "AAAABBBBCCCC" not in redacted


# ---------------------------------------------------------------------------
# Negative: empty / benign inputs are passthrough
# ---------------------------------------------------------------------------


def test_empty_string_returns_empty():
    assert redact_secrets("") == ""


def test_benign_text_unchanged():
    benign = "Trainer started step 42 of 1000, loss=0.123"
    assert redact_secrets(benign) == benign


def test_non_secret_substring_with_hf_prefix_only_partial():
    """``hf_`` followed by 0 alphanumeric chars does NOT match — would be
    false-positive on common English words. Only ``hf_<at-least-one-char>``
    triggers."""
    assert redact_secrets("hf_") == "hf_"


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


def test_multiple_secrets_in_same_string_all_masked():
    raw = "HF_TOKEN=hf_aaa hf_bbb tail API_KEY=ccc"
    redacted = redact_secrets(raw)
    assert "hf_aaa" not in redacted
    assert "hf_bbb" not in redacted
    assert "ccc" not in redacted


def test_multiline_string_redacted_per_line():
    raw = "line1\nHF_TOKEN=hf_secret1\nline3\nAPI_KEY=secret2\nlast"
    redacted = redact_secrets(raw)
    assert "hf_secret1" not in redacted
    assert "secret2" not in redacted
    # Surrounding lines preserved
    assert "line1" in redacted
    assert "line3" in redacted
    assert "last" in redacted


# ---------------------------------------------------------------------------
# Operator extension via env var
# ---------------------------------------------------------------------------


def test_extra_patterns_from_env_var(monkeypatch: pytest.MonkeyPatch):
    """RYOTENKAI_REDACT_PATTERNS comma-separated regexes are applied in
    addition to the defaults."""
    monkeypatch.setenv("RYOTENKAI_REDACT_PATTERNS", r"COMPANY_TOKEN_[A-Z0-9]+")
    redacted = redact_secrets("leaked COMPANY_TOKEN_ABCDEF123 in trace")
    assert "COMPANY_TOKEN_ABCDEF123" not in redacted
    assert "***" in redacted


def test_invalid_extra_pattern_silently_skipped(monkeypatch: pytest.MonkeyPatch):
    """A bad regex must not crash the redactor — failing to redact is bad,
    crashing the post-mortem caller is worse. Defaults still apply."""
    monkeypatch.setenv("RYOTENKAI_REDACT_PATTERNS", "(?P<unbalanced")
    redacted = redact_secrets("HF_TOKEN=hf_secret")
    assert "hf_secret" not in redacted


def test_no_env_var_uses_defaults_only(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RYOTENKAI_REDACT_PATTERNS", raising=False)
    assert "hf_secret" not in redact_secrets("hf_secret")
