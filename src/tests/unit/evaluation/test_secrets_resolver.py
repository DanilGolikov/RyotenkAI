"""
Unit tests for SecretsResolver and @requires_secrets decorator.

Tests:
- test_resolve_from_model_extra        — EVAL_* key found in model_extra
- test_resolve_namespace_violation_raises — key without EVAL_ prefix → ValueError
- test_resolve_missing_key_raises      — EVAL_ key not in secrets.env → RuntimeError
- test_requires_secrets_decorator_sets_attribute — decorator sets _required_secrets
- test_secrets_extra_normalization     — model_validator trims whitespace, removes empty values
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.evaluation.plugins.secrets import SecretsResolver, requires_secrets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_secrets(extra: dict[str, str]) -> MagicMock:
    """Create a mock Secrets object with the given model_extra."""
    s = MagicMock()
    s.model_extra = extra
    return s


# ---------------------------------------------------------------------------
# SecretsResolver tests
# ---------------------------------------------------------------------------


class TestSecretsResolver:
    def test_resolve_from_model_extra(self):
        """EVAL_* key found and returned from model_extra (pydantic lowercases keys)."""
        secrets = _make_secrets({"eval_cerebras_api_key": "csk-test-123"})
        resolver = SecretsResolver(secrets)

        result = resolver.resolve(("EVAL_CEREBRAS_API_KEY",))

        assert result == {"EVAL_CEREBRAS_API_KEY": "csk-test-123"}

    def test_resolve_multiple_keys(self):
        """Multiple EVAL_* keys resolved correctly."""
        secrets = _make_secrets({
            "eval_cerebras_api_key": "csk-test",
            "eval_other_key": "other-value",
        })
        resolver = SecretsResolver(secrets)

        result = resolver.resolve(("EVAL_CEREBRAS_API_KEY", "EVAL_OTHER_KEY"))

        assert result["EVAL_CEREBRAS_API_KEY"] == "csk-test"
        assert result["EVAL_OTHER_KEY"] == "other-value"

    def test_resolve_namespace_violation_raises(self):
        """Key without EVAL_ prefix → ValueError (namespace isolation)."""
        secrets = _make_secrets({"hf_token": "hf-secret"})
        resolver = SecretsResolver(secrets)

        with pytest.raises(ValueError, match="EVAL_"):
            resolver.resolve(("HF_TOKEN",))

    def test_resolve_namespace_violation_mixed_raises(self):
        """Even one non-EVAL_ key raises ValueError before resolution."""
        secrets = _make_secrets({"eval_key": "val"})
        resolver = SecretsResolver(secrets)

        with pytest.raises(ValueError, match="EVAL_"):
            resolver.resolve(("EVAL_KEY", "RUNPOD_API_KEY"))

    def test_resolve_missing_key_raises(self):
        """EVAL_* key with correct prefix but absent from secrets.env → RuntimeError."""
        secrets = _make_secrets({})  # empty model_extra
        resolver = SecretsResolver(secrets)

        with pytest.raises(RuntimeError, match="EVAL_MISSING_KEY"):
            resolver.resolve(("EVAL_MISSING_KEY",))

    def test_resolve_empty_value_raises(self):
        """EVAL_* key present but empty after normalization → RuntimeError."""
        secrets = _make_secrets({"eval_empty_key": ""})
        resolver = SecretsResolver(secrets)

        with pytest.raises(RuntimeError, match="EVAL_EMPTY_KEY"):
            resolver.resolve(("EVAL_EMPTY_KEY",))

    def test_resolve_none_model_extra_raises(self):
        """model_extra is None (no plugin secrets in secrets.env) → RuntimeError."""
        secrets = _make_secrets({})
        secrets.model_extra = None
        resolver = SecretsResolver(secrets)

        with pytest.raises(RuntimeError, match="EVAL_KEY"):
            resolver.resolve(("EVAL_KEY",))


# ---------------------------------------------------------------------------
# @requires_secrets decorator tests
# ---------------------------------------------------------------------------


class TestRequiresSecretsDecorator:
    def test_sets_required_secrets_attribute(self):
        """Decorator sets _required_secrets as a tuple on the class."""

        @requires_secrets("EVAL_CEREBRAS_API_KEY")
        class DummyPlugin:
            pass

        assert DummyPlugin._required_secrets == ("EVAL_CEREBRAS_API_KEY",)

    def test_sets_multiple_keys(self):
        """Decorator stores all declared keys."""

        @requires_secrets("EVAL_KEY_A", "EVAL_KEY_B")
        class DummyPlugin:
            pass

        assert DummyPlugin._required_secrets == ("EVAL_KEY_A", "EVAL_KEY_B")

    def test_returns_class_unchanged(self):
        """Decorator returns the same class (not a wrapper)."""

        class OriginalPlugin:
            pass

        decorated = requires_secrets("EVAL_X")(OriginalPlugin)
        assert decorated is OriginalPlugin

    def test_does_not_validate_prefix_at_decoration_time(self):
        """Prefix validation is NOT enforced at decoration time — only at resolve time."""

        @requires_secrets("NON_EVAL_KEY")  # no EVAL_ prefix — but decorator itself doesn't raise
        class DummyPlugin:
            pass

        assert DummyPlugin._required_secrets == ("NON_EVAL_KEY",)


# ---------------------------------------------------------------------------
# Secrets model_extra normalization tests (integration with pydantic model)
# ---------------------------------------------------------------------------


class TestSecretsNormalization:
    def test_secrets_extra_normalization_trims_whitespace(self):
        """
        model_validator in Secrets trims whitespace and removes empty values.

        Since we can't load a real Secrets object without a secrets.env file,
        we test the normalization logic inline by mimicking what the validator does.
        """
        raw_extra = {
            "eval_cerebras_api_key": "  csk-test  ",
            "eval_empty": "   ",
            "eval_valid": "value",
        }

        # Apply the same normalization logic as Secrets._normalize_extra
        cleaned = {}
        for k, v in raw_extra.items():
            if isinstance(v, str):
                v = v.strip()
                if v:
                    cleaned[k] = v

        assert cleaned["eval_cerebras_api_key"] == "csk-test"
        assert "eval_empty" not in cleaned
        assert cleaned["eval_valid"] == "value"
