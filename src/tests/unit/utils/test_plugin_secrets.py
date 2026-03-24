"""
Unit tests for the shared PluginSecretsResolver base.

Verifies prefix-agnostic behavior and that custom prefixes work correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.utils.plugin_secrets import PluginSecretsResolver, requires_secrets


def _make_secrets(extra: dict[str, str]) -> MagicMock:
    s = MagicMock()
    s.model_extra = extra
    return s


class TestPluginSecretsResolverGeneric:
    def test_custom_prefix_resolves(self):
        """Resolver with arbitrary prefix resolves matching keys."""
        secrets = _make_secrets({"myns_api_key": "val123"})
        resolver = PluginSecretsResolver(secrets, prefix="MYNS_")

        result = resolver.resolve(("MYNS_API_KEY",))
        assert result == {"MYNS_API_KEY": "val123"}

    def test_custom_prefix_rejects_wrong_namespace(self):
        """Resolver rejects keys outside its namespace."""
        secrets = _make_secrets({})
        resolver = PluginSecretsResolver(secrets, prefix="MYNS_")

        with pytest.raises(ValueError, match="MYNS_"):
            resolver.resolve(("OTHER_KEY",))

    def test_prefix_property(self):
        secrets = _make_secrets({})
        resolver = PluginSecretsResolver(secrets, prefix="ABC_")
        assert resolver.prefix == "ABC_"

    def test_empty_keys_resolves_empty(self):
        """Resolving an empty tuple returns an empty dict."""
        secrets = _make_secrets({"myns_key": "val"})
        resolver = PluginSecretsResolver(secrets, prefix="MYNS_")

        result = resolver.resolve(())
        assert result == {}


class TestRequiresSecretsGeneric:
    def test_decorator_sets_attribute(self):
        @requires_secrets("SOME_KEY")
        class Dummy:
            pass

        assert Dummy._required_secrets == ("SOME_KEY",)

    def test_decorator_preserves_class_identity(self):
        class Orig:
            pass

        dec = requires_secrets("K")(Orig)
        assert dec is Orig

    def test_decorator_no_args(self):
        @requires_secrets()
        class Dummy:
            pass

        assert Dummy._required_secrets == ()
