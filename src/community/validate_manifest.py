"""Standalone manifest validation — TOML + Pydantic, no Python import.

The community :mod:`loader` validates a manifest as a side effect of
loading the plugin's Python class. That coupling is fine for the runtime
(if the manifest is valid but the class fails to import, the loader
should reject it as one unit), but it's wrong for two new CLI flows:

- ``ryotenkai plugin validate <path>`` — the user wants a yes/no answer
  on the manifest alone, before they wire up the Python side.
- ``ryotenkai plugin install <source>`` — must reject a malformed
  manifest *before* it copies anything into ``community/<kind>/``.

Both call into :func:`validate_manifest_file`, which:

1. Reads the file (``OSError`` → single ``error_io`` issue).
2. Parses TOML (``tomllib.TOMLDecodeError`` → ``error_toml`` issue).
3. Picks the right Pydantic model from the top-level table
   (``[plugin]`` vs ``[preset]``) and runs ``model_validate``. Pydantic
   ``ValidationError`` is unrolled into one :class:`ValidationIssue`
   per field path with stable severity ``error``.
4. Surfaces "soft" issues as warnings:
   - plugin manifests that omit ``schema_version`` are accepted but
     warned about (recommended to pin so the loader can refuse newer
     schemas with a clear message).
   - presets without a ``[preset.scope]`` block are accepted but
     warned about (v1-compat full-replace mode; explicit scope is the
     v2 contract).

The function never imports the plugin class — that's the loader's job.
This means a malformed plugin.py won't mask a fine manifest, and CI
linting can run on a checkout without all plugin deps installed.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from src.community.constants import MANIFEST_FILENAME
from src.community.manifest import (
    LATEST_SCHEMA_VERSION,
    PluginManifest,
    PresetManifest,
)

#: Top-level TOML tables we recognise. ``"unknown"`` is reserved for
#: malformed inputs (no recognised table or both tables present).
ManifestKind = Literal["plugin", "preset", "unknown"]

#: Stable, machine-readable error categories. Mirrors the loader's
#: ``LoadFailure.error_type`` vocabulary so callers can reuse the same
#: switch when surfacing issues to users.
IssueCode = Literal[
    "error_io",
    "error_toml",
    "error_kind_ambiguous",
    "error_kind_missing",
    "error_schema",
    "warn_no_schema_version",
    "warn_no_preset_scope",
]

#: ``severity`` literal so callers (CLI, API) can render different
#: glyphs per level without string-typoing the constants.
Severity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One row in the validation report.

    ``location`` is a dotted path through the TOML tree
    (``"plugin.entry_point.class"``) when known, or an empty string for
    file-level errors (``error_io``, ``error_toml``).
    """

    severity: Severity
    code: IssueCode
    location: str
    message: str


@dataclass(frozen=True, slots=True)
class ManifestValidationResult:
    """Outcome of :func:`validate_manifest_file`.

    Designed for both human (CLI text rendering) and machine (JSON via
    ``ryotenkai plugin validate -o json``) consumers.
    """

    path: Path
    kind: ManifestKind
    manifest_id: str | None
    schema_version: int | None
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Issues that should fail validation (exit non-zero)."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Issues a strict caller should treat as errors but a normal
        run lets pass."""
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        """True iff there are no ``error`` issues. Warnings don't count."""
        return not self.errors

    def passes(self, *, strict: bool) -> bool:
        """Strict mode promotes warnings to errors."""
        if strict:
            return not self.issues
        return self.is_valid


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_manifest_file(path: Path) -> ManifestValidationResult:
    """Validate the ``manifest.toml`` at ``path`` without importing any
    plugin code.

    Always returns a :class:`ManifestValidationResult` — never raises.
    Callers branch on ``.is_valid`` / ``.passes(strict=…)`` to pick exit
    codes.
    """
    text, io_issue = _read_text(path)
    if io_issue is not None:
        return ManifestValidationResult(
            path=path, kind="unknown", manifest_id=None,
            schema_version=None, issues=[io_issue],
        )
    assert text is not None  # for type-checker — handled above

    payload, toml_issue = _parse_toml(text)
    if toml_issue is not None:
        return ManifestValidationResult(
            path=path, kind="unknown", manifest_id=None,
            schema_version=None, issues=[toml_issue],
        )
    assert payload is not None

    kind, kind_issue = _detect_kind(payload)
    if kind_issue is not None:
        return ManifestValidationResult(
            path=path, kind=kind, manifest_id=None,
            schema_version=None, issues=[kind_issue],
        )

    if kind == "plugin":
        return _validate_plugin(path, payload)
    return _validate_preset(path, payload)


def validate_manifest_dir(folder: Path) -> ManifestValidationResult:
    """Convenience: validate ``<folder>/manifest.toml``.

    Returns a result with ``error_io`` if the folder has no manifest —
    matches what callers expect from ``plugin validate <folder>``.
    """
    return validate_manifest_file(folder / MANIFEST_FILENAME)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> tuple[str | None, ValidationIssue | None]:
    try:
        return path.read_text(encoding="utf-8"), None
    except FileNotFoundError:
        return None, ValidationIssue(
            severity="error", code="error_io", location="",
            message=f"manifest file not found: {path}",
        )
    except OSError as exc:
        return None, ValidationIssue(
            severity="error", code="error_io", location="",
            message=f"cannot read {path}: {exc}",
        )


def _parse_toml(text: str) -> tuple[dict | None, ValidationIssue | None]:
    try:
        return tomllib.loads(text), None
    except tomllib.TOMLDecodeError as exc:
        return None, ValidationIssue(
            severity="error", code="error_toml", location="",
            message=f"TOML parse error: {exc}",
        )


def _detect_kind(payload: dict) -> tuple[ManifestKind, ValidationIssue | None]:
    has_plugin = isinstance(payload.get("plugin"), dict)
    has_preset = isinstance(payload.get("preset"), dict)
    if has_plugin and has_preset:
        return "unknown", ValidationIssue(
            severity="error", code="error_kind_ambiguous", location="",
            message="manifest declares both [plugin] and [preset]; pick one",
        )
    if has_plugin:
        return "plugin", None
    if has_preset:
        return "preset", None
    return "unknown", ValidationIssue(
        severity="error", code="error_kind_missing", location="",
        message="manifest missing required top-level table: expected [plugin] or [preset]",
    )


def _validate_plugin(path: Path, payload: dict) -> ManifestValidationResult:
    """Run :class:`PluginManifest` validation + plugin-specific warnings."""
    manifest_id = _safe_get_str(payload, "plugin", "id")
    schema_version_raw = payload.get("schema_version")
    declared_schema_version = (
        int(schema_version_raw) if isinstance(schema_version_raw, int) else None
    )
    issues: list[ValidationIssue] = []

    try:
        manifest = PluginManifest.model_validate(payload)
    except ValidationError as exc:
        issues.extend(_pydantic_issues(exc))
        return ManifestValidationResult(
            path=path, kind="plugin", manifest_id=manifest_id,
            schema_version=declared_schema_version, issues=issues,
        )

    # Soft warning: missing schema_version means the manifest is locked
    # to LATEST at load time but the user has no signal in the file
    # itself. Recommend pinning explicitly.
    if declared_schema_version is None:
        issues.append(ValidationIssue(
            severity="warning", code="warn_no_schema_version",
            location="schema_version",
            message=(
                f"schema_version is not declared; loader treats it as "
                f"v{LATEST_SCHEMA_VERSION}. Add `schema_version = "
                f"{LATEST_SCHEMA_VERSION}` to make the contract explicit."
            ),
        ))

    return ManifestValidationResult(
        path=path, kind="plugin", manifest_id=manifest.plugin.id,
        schema_version=manifest.schema_version, issues=issues,
    )


def _validate_preset(path: Path, payload: dict) -> ManifestValidationResult:
    manifest_id = _safe_get_str(payload, "preset", "id")
    issues: list[ValidationIssue] = []

    try:
        manifest = PresetManifest.model_validate(payload)
    except ValidationError as exc:
        issues.extend(_pydantic_issues(exc))
        return ManifestValidationResult(
            path=path, kind="preset", manifest_id=manifest_id,
            schema_version=None, issues=issues,
        )

    # Soft warning: presets without [preset.scope] run in v1 full-replace
    # mode, which silently overwrites the user's whole config. v2 with
    # explicit scope is the recommended contract.
    if manifest.preset.scope is None:
        issues.append(ValidationIssue(
            severity="warning", code="warn_no_preset_scope",
            location="preset.scope",
            message=(
                "preset omits [preset.scope]; runs in v1 full-replace mode. "
                "Declare `[preset.scope]` with explicit `replaces` / "
                "`preserves` lists to opt into v2 partial-apply semantics."
            ),
        ))

    return ManifestValidationResult(
        path=path, kind="preset", manifest_id=manifest.preset.id,
        schema_version=None, issues=issues,
    )


def _pydantic_issues(exc: ValidationError) -> list[ValidationIssue]:
    """Flatten a Pydantic ``ValidationError`` into stable issue rows."""
    out: list[ValidationIssue] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()))
        out.append(ValidationIssue(
            severity="error", code="error_schema",
            location=loc, message=err.get("msg", "validation failed"),
        ))
    return out


def _safe_get_str(payload: dict, *path: str) -> str | None:
    """Walk nested dicts; return a string leaf or ``None`` on any miss."""
    cursor: object = payload
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    if isinstance(cursor, str):
        return cursor or None
    return None


__all__ = [
    "IssueCode",
    "ManifestKind",
    "ManifestValidationResult",
    "Severity",
    "ValidationIssue",
    "validate_manifest_dir",
    "validate_manifest_file",
]
