"""Tests for the manifest ``schema_version`` gate.

The loader treats missing values as :data:`LATEST_SCHEMA_VERSION`
(legacy TOMLs keep loading), accepts any equal-or-lower version, and
rejects future versions with a clear upgrade-the-host error.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.community.manifest import LATEST_SCHEMA_VERSION, PluginManifest


def _minimal_manifest_dict(**overrides):
    base = {
        "plugin": {
            "id": "tiny",
            "kind": "validation",
            "version": "1.0.0",
            "entry_point": {"module": "plugin", "class": "TestPlugin"},
        }
    }
    base.update(overrides)
    return base


def test_default_schema_version_is_latest() -> None:
    """Manifests omitting ``schema_version`` default to LATEST."""
    manifest = PluginManifest.model_validate(_minimal_manifest_dict())
    assert manifest.schema_version == LATEST_SCHEMA_VERSION


def test_explicit_current_version_accepted() -> None:
    manifest = PluginManifest.model_validate(
        _minimal_manifest_dict(schema_version=LATEST_SCHEMA_VERSION),
    )
    assert manifest.schema_version == LATEST_SCHEMA_VERSION


def test_explicit_lower_version_accepted() -> None:
    """Lower numbers are accepted — back-compat is the loader's job."""
    if LATEST_SCHEMA_VERSION <= 1:
        pytest.skip("no lower version available to test")
    manifest = PluginManifest.model_validate(
        _minimal_manifest_dict(schema_version=1),
    )
    assert manifest.schema_version == 1


def test_zero_or_negative_rejected() -> None:
    with pytest.raises(ValidationError, match=r"schema_version must be >= 1"):
        PluginManifest.model_validate(_minimal_manifest_dict(schema_version=0))


def test_future_version_rejected_with_upgrade_hint() -> None:
    """Versions higher than LATEST point the user at a host upgrade."""
    future = LATEST_SCHEMA_VERSION + 1
    with pytest.raises(ValidationError, match=r"Upgrade the host"):
        PluginManifest.model_validate(_minimal_manifest_dict(schema_version=future))


def test_ui_manifest_includes_schema_version() -> None:
    """Surface the version through the UI payload so the front-end can
    show a "this manifest looks new" hint when it diverges from what
    the front-end itself was generated against."""
    manifest = PluginManifest.model_validate(_minimal_manifest_dict())
    out = manifest.ui_manifest()
    assert out["schema_version"] == LATEST_SCHEMA_VERSION
