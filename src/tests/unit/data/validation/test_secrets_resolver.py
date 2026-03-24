"""
Unit tests for DTST_* SecretsResolver and @requires_secrets decorator
for the dataset validation plugin system.

Tests:
- test_resolve_from_model_extra          — DTST_* key found in model_extra
- test_resolve_multiple_keys             — multiple DTST_* keys resolved
- test_resolve_namespace_violation_raises — key without DTST_ prefix → ValueError
- test_resolve_namespace_violation_mixed  — mixed DTST_/non-DTST_ → ValueError
- test_resolve_missing_key_raises        — DTST_* key absent → RuntimeError
- test_resolve_empty_value_raises        — DTST_* key empty → RuntimeError
- test_resolve_none_model_extra_raises   — model_extra is None → RuntimeError
- test_requires_secrets_decorator_sets_attribute
- test_decorator_multiple_keys
- test_decorator_returns_class_unchanged
- test_decorator_no_prefix_validation_at_decoration_time
- test_namespace_isolation_eval_vs_dtst  — EVAL_* prefix rejected by DTST_ resolver
- test_namespace_isolation_dtst_vs_eval  — DTST_* prefix rejected by EVAL_ resolver
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.data.validation.secrets import SecretsResolver as DtstSecretsResolver
from src.data.validation.secrets import requires_secrets
from src.evaluation.plugins.secrets import SecretsResolver as EvalSecretsResolver


def _make_secrets(extra: dict[str, str]) -> MagicMock:
    """Create a mock Secrets object with the given model_extra."""
    s = MagicMock()
    s.model_extra = extra
    return s


class TestDtstSecretsResolver:
    def test_resolve_from_model_extra(self):
        """DTST_* key found and returned from model_extra (pydantic lowercases keys)."""
        secrets = _make_secrets({"dtst_schema_validator_token": "tok-abc"})
        resolver = DtstSecretsResolver(secrets)

        result = resolver.resolve(("DTST_SCHEMA_VALIDATOR_TOKEN",))

        assert result == {"DTST_SCHEMA_VALIDATOR_TOKEN": "tok-abc"}

    def test_resolve_multiple_keys(self):
        """Multiple DTST_* keys resolved correctly."""
        secrets = _make_secrets({
            "dtst_schema_validator_token": "tok-abc",
            "dtst_quality_api_key": "key-xyz",
        })
        resolver = DtstSecretsResolver(secrets)

        result = resolver.resolve(("DTST_SCHEMA_VALIDATOR_TOKEN", "DTST_QUALITY_API_KEY"))

        assert result["DTST_SCHEMA_VALIDATOR_TOKEN"] == "tok-abc"
        assert result["DTST_QUALITY_API_KEY"] == "key-xyz"

    def test_resolve_namespace_violation_raises(self):
        """Key without DTST_ prefix → ValueError (namespace isolation)."""
        secrets = _make_secrets({"hf_token": "hf-secret"})
        resolver = DtstSecretsResolver(secrets)

        with pytest.raises(ValueError, match="DTST_"):
            resolver.resolve(("HF_TOKEN",))

    def test_resolve_namespace_violation_mixed_raises(self):
        """Even one non-DTST_ key raises ValueError before resolution."""
        secrets = _make_secrets({"dtst_key": "val"})
        resolver = DtstSecretsResolver(secrets)

        with pytest.raises(ValueError, match="DTST_"):
            resolver.resolve(("DTST_KEY", "RUNPOD_API_KEY"))

    def test_resolve_missing_key_raises(self):
        """DTST_* key with correct prefix but absent from secrets.env → RuntimeError."""
        secrets = _make_secrets({})
        resolver = DtstSecretsResolver(secrets)

        with pytest.raises(RuntimeError, match="DTST_MISSING_KEY"):
            resolver.resolve(("DTST_MISSING_KEY",))

    def test_resolve_empty_value_raises(self):
        """DTST_* key present but empty → RuntimeError."""
        secrets = _make_secrets({"dtst_empty_key": ""})
        resolver = DtstSecretsResolver(secrets)

        with pytest.raises(RuntimeError, match="DTST_EMPTY_KEY"):
            resolver.resolve(("DTST_EMPTY_KEY",))

    def test_resolve_none_model_extra_raises(self):
        """model_extra is None → RuntimeError."""
        secrets = _make_secrets({})
        secrets.model_extra = None
        resolver = DtstSecretsResolver(secrets)

        with pytest.raises(RuntimeError, match="DTST_KEY"):
            resolver.resolve(("DTST_KEY",))

    def test_prefix_property(self):
        """Resolver exposes its prefix via property."""
        secrets = _make_secrets({})
        resolver = DtstSecretsResolver(secrets)
        assert resolver.prefix == "DTST_"


class TestRequiresSecretsDecorator:
    def test_sets_required_secrets_attribute(self):
        """Decorator sets _required_secrets as a tuple on the class."""

        @requires_secrets("DTST_SCHEMA_VALIDATOR_TOKEN")
        class DummyPlugin:
            pass

        assert DummyPlugin._required_secrets == ("DTST_SCHEMA_VALIDATOR_TOKEN",)

    def test_sets_multiple_keys(self):
        """Decorator stores all declared keys."""

        @requires_secrets("DTST_KEY_A", "DTST_KEY_B")
        class DummyPlugin:
            pass

        assert DummyPlugin._required_secrets == ("DTST_KEY_A", "DTST_KEY_B")

    def test_returns_class_unchanged(self):
        """Decorator returns the same class (not a wrapper)."""

        class OriginalPlugin:
            pass

        decorated = requires_secrets("DTST_X")(OriginalPlugin)
        assert decorated is OriginalPlugin

    def test_does_not_validate_prefix_at_decoration_time(self):
        """Prefix validation is NOT enforced at decoration time."""

        @requires_secrets("NON_DTST_KEY")
        class DummyPlugin:
            pass

        assert DummyPlugin._required_secrets == ("NON_DTST_KEY",)


class TestNamespaceIsolation:
    """Cross-namespace isolation: EVAL resolver rejects DTST_ keys and vice versa."""

    def test_eval_resolver_rejects_dtst_key(self):
        """EVAL_ resolver must reject DTST_* keys."""
        secrets = _make_secrets({"dtst_key": "val"})
        resolver = EvalSecretsResolver(secrets)

        with pytest.raises(ValueError, match="EVAL_"):
            resolver.resolve(("DTST_KEY",))

    def test_dtst_resolver_rejects_eval_key(self):
        """DTST_ resolver must reject EVAL_* keys."""
        secrets = _make_secrets({"eval_key": "val"})
        resolver = DtstSecretsResolver(secrets)

        with pytest.raises(ValueError, match="DTST_"):
            resolver.resolve(("EVAL_KEY",))

    def test_both_resolvers_share_requires_secrets_decorator(self):
        """Both systems use the same @requires_secrets decorator from shared base."""
        from src.evaluation.plugins.secrets import requires_secrets as eval_requires
        from src.data.validation.secrets import requires_secrets as dtst_requires
        from src.utils.plugin_secrets import requires_secrets as base_requires

        assert eval_requires is base_requires
        assert dtst_requires is base_requires
