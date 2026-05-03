"""Unit tests for ``src.community.validate_manifest``.

Cover the contract documented in the module:

- File-IO errors are returned as ``error_io`` issues, never raised.
- TOML decode errors are returned as ``error_toml``.
- Missing or ambiguous top-level kind tables produce kind issues.
- Pydantic ``ValidationError`` flattens into ``error_schema`` issues with
  dotted ``location`` paths.
- Soft warnings: missing ``schema_version`` for plugins, missing
  ``[preset.scope]`` for presets — accepted but flagged.
- ``ManifestValidationResult.passes(strict=True)`` promotes warnings to
  errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.community.constants import MANIFEST_FILENAME
from src.community.manifest import LATEST_SCHEMA_VERSION
from src.community.validate_manifest import (
    ManifestValidationResult,
    validate_manifest_dir,
    validate_manifest_file,
)

# ---------------------------------------------------------------------------
# Fixtures: minimal valid manifests as TOML text
# ---------------------------------------------------------------------------


def _valid_plugin_toml(*, with_schema_version: bool = True) -> str:
    header = f"schema_version = {LATEST_SCHEMA_VERSION}\n" if with_schema_version else ""
    return header + (
        "[plugin]\n"
        'id = "min_samples"\n'
        'kind = "validation"\n'
        'name = "Min Samples"\n'
        'version = "1.0.0"\n'
        'category = "basic"\n'
        'stability = "stable"\n'
        'description = "Checks dataset has >= N rows"\n'
        "\n"
        "[plugin.entry_point]\n"
        'module = "plugin"\n'
        'class = "MinSamplesValidator"\n'
    )


def _valid_preset_toml(*, with_scope: bool = True) -> str:
    body = (
        "[preset]\n"
        'id = "04-sft-quickstart"\n'
        'name = "SFT quickstart"\n'
        'description = "Minimal SFT loop"\n'
        'size_tier = "small"\n'
        'version = "1.0.0"\n'
        "\n"
        "[preset.entry_point]\n"
        'file = "preset.yaml"\n'
    )
    if with_scope:
        body += (
            "\n[preset.scope]\n"
            'replaces = ["model", "training"]\n'
            'preserves = ["datasets"]\n'
        )
    return body


@pytest.fixture()
def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / MANIFEST_FILENAME


# ---------------------------------------------------------------------------
# IO + TOML errors
# ---------------------------------------------------------------------------


def test_missing_file_returns_error_io(tmp_path: Path) -> None:
    result = validate_manifest_file(tmp_path / "does-not-exist.toml")
    assert not result.is_valid
    assert result.kind == "unknown"
    assert result.errors[0].code == "error_io"
    assert "not found" in result.errors[0].message


def test_unreadable_file_returns_error_io(
    monkeypatch: pytest.MonkeyPatch, manifest_path: Path,
) -> None:
    """Permission denied surfaces as error_io, not a stack trace."""
    manifest_path.write_text(_valid_plugin_toml(), encoding="utf-8")
    real_read_text = Path.read_text

    def boom(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self == manifest_path:
            raise PermissionError("denied")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", boom)
    result = validate_manifest_file(manifest_path)
    assert result.errors[0].code == "error_io"
    assert "denied" in result.errors[0].message


def test_invalid_toml_returns_error_toml(manifest_path: Path) -> None:
    manifest_path.write_text("this = is = not = toml", encoding="utf-8")
    result = validate_manifest_file(manifest_path)
    assert not result.is_valid
    assert result.errors[0].code == "error_toml"


# ---------------------------------------------------------------------------
# Kind detection
# ---------------------------------------------------------------------------


def test_missing_kind_table(manifest_path: Path) -> None:
    manifest_path.write_text("foo = 1\n", encoding="utf-8")
    result = validate_manifest_file(manifest_path)
    assert result.errors[0].code == "error_kind_missing"
    assert result.kind == "unknown"


def test_ambiguous_kind_table(manifest_path: Path) -> None:
    manifest_path.write_text(
        _valid_plugin_toml() + "\n" + _valid_preset_toml(),
        encoding="utf-8",
    )
    result = validate_manifest_file(manifest_path)
    assert result.errors[0].code == "error_kind_ambiguous"
    assert result.kind == "unknown"


# ---------------------------------------------------------------------------
# Plugin manifest happy path + warnings
# ---------------------------------------------------------------------------


def test_valid_plugin_manifest_passes(manifest_path: Path) -> None:
    manifest_path.write_text(_valid_plugin_toml(), encoding="utf-8")
    result = validate_manifest_file(manifest_path)
    assert result.is_valid
    assert result.kind == "plugin"
    assert result.manifest_id == "min_samples"
    assert result.schema_version == LATEST_SCHEMA_VERSION
    assert result.warnings == []


def test_plugin_without_schema_version_warns(manifest_path: Path) -> None:
    manifest_path.write_text(
        _valid_plugin_toml(with_schema_version=False), encoding="utf-8"
    )
    result = validate_manifest_file(manifest_path)
    assert result.is_valid  # warning ≠ error
    assert len(result.warnings) == 1
    assert result.warnings[0].code == "warn_no_schema_version"
    assert "schema_version" in result.warnings[0].message


def test_plugin_with_future_schema_version_fails(manifest_path: Path) -> None:
    manifest_path.write_text(
        f"schema_version = {LATEST_SCHEMA_VERSION + 99}\n" + _valid_plugin_toml(
            with_schema_version=False,
        ),
        encoding="utf-8",
    )
    result = validate_manifest_file(manifest_path)
    assert not result.is_valid
    assert result.errors[0].code == "error_schema"
    # Pydantic's wrapped message includes both the value and the upper bound.
    msg = result.errors[0].message
    assert str(LATEST_SCHEMA_VERSION) in msg
    assert "Upgrade" in msg or "upgrade" in msg


def test_plugin_with_invalid_kind_returns_error_schema(manifest_path: Path) -> None:
    manifest_path.write_text(
        _valid_plugin_toml().replace('kind = "validation"', 'kind = "nonsense"'),
        encoding="utf-8",
    )
    result = validate_manifest_file(manifest_path)
    assert not result.is_valid
    err = result.errors[0]
    assert err.code == "error_schema"
    assert err.location.startswith("plugin.kind")


def test_reward_plugin_without_supported_strategies_fails(manifest_path: Path) -> None:
    """Cross-validator from PluginSpec — reward plugins must declare strategies."""
    manifest_path.write_text(
        _valid_plugin_toml().replace(
            'kind = "validation"', 'kind = "reward"',
        ),
        encoding="utf-8",
    )
    result = validate_manifest_file(manifest_path)
    assert not result.is_valid
    assert any("supported_strategies" in i.message for i in result.errors)


# ---------------------------------------------------------------------------
# Preset manifest happy path + warnings
# ---------------------------------------------------------------------------


def test_valid_preset_manifest_passes(manifest_path: Path) -> None:
    manifest_path.write_text(_valid_preset_toml(), encoding="utf-8")
    result = validate_manifest_file(manifest_path)
    assert result.is_valid
    assert result.kind == "preset"
    assert result.manifest_id == "04-sft-quickstart"
    assert result.schema_version is None  # presets have no schema_version
    assert result.warnings == []


def test_preset_without_scope_warns(manifest_path: Path) -> None:
    manifest_path.write_text(_valid_preset_toml(with_scope=False), encoding="utf-8")
    result = validate_manifest_file(manifest_path)
    assert result.is_valid
    assert len(result.warnings) == 1
    assert result.warnings[0].code == "warn_no_preset_scope"
    assert result.warnings[0].location == "preset.scope"


def test_preset_with_overlapping_scope_fails(manifest_path: Path) -> None:
    manifest_path.write_text(
        _valid_preset_toml().replace(
            'preserves = ["datasets"]', 'preserves = ["model"]',
        ),
        encoding="utf-8",
    )
    result = validate_manifest_file(manifest_path)
    assert not result.is_valid
    assert any("overlap" in i.message.lower() for i in result.errors)


# ---------------------------------------------------------------------------
# strict mode
# ---------------------------------------------------------------------------


def test_strict_mode_promotes_warnings_to_errors(manifest_path: Path) -> None:
    manifest_path.write_text(
        _valid_plugin_toml(with_schema_version=False), encoding="utf-8"
    )
    result = validate_manifest_file(manifest_path)
    assert result.is_valid is True
    assert result.passes(strict=False) is True
    assert result.passes(strict=True) is False  # warning blocks


def test_strict_mode_preset_no_scope(manifest_path: Path) -> None:
    manifest_path.write_text(_valid_preset_toml(with_scope=False), encoding="utf-8")
    result = validate_manifest_file(manifest_path)
    assert result.passes(strict=True) is False


# ---------------------------------------------------------------------------
# validate_manifest_dir convenience
# ---------------------------------------------------------------------------


def test_validate_manifest_dir_finds_manifest_toml(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "min_samples"
    plugin_dir.mkdir()
    (plugin_dir / MANIFEST_FILENAME).write_text(
        _valid_plugin_toml(), encoding="utf-8",
    )
    result = validate_manifest_dir(plugin_dir)
    assert result.is_valid
    assert result.kind == "plugin"


def test_validate_manifest_dir_missing_manifest(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "empty"
    plugin_dir.mkdir()
    result = validate_manifest_dir(plugin_dir)
    assert result.errors[0].code == "error_io"


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


def test_result_errors_and_warnings_partition(manifest_path: Path) -> None:
    """``errors`` and ``warnings`` together cover ``issues``."""
    manifest_path.write_text(
        _valid_plugin_toml(with_schema_version=False), encoding="utf-8"
    )
    result = validate_manifest_file(manifest_path)
    assert len(result.errors) + len(result.warnings) == len(result.issues)


def test_result_is_valid_when_no_errors() -> None:
    result = ManifestValidationResult(
        path=Path("/x"), kind="plugin", manifest_id="x",
        schema_version=LATEST_SCHEMA_VERSION, issues=[],
    )
    assert result.is_valid
    assert result.passes(strict=True)
