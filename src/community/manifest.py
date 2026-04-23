"""Pydantic models for community manifests (plugins and presets)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PluginKind = Literal["validation", "evaluation", "reward", "reports"]

Stability = Literal["stable", "beta", "experimental"]


class EntryPoint(BaseModel):
    """Entry point for plugin code imports."""

    model_config = ConfigDict(extra="forbid")

    module: str = Field(description="Python module name inside the plugin folder (without .py).")
    class_name: str = Field(
        alias="class",
        description="Fully-qualified class name within the module.",
    )


class PresetEntryPoint(BaseModel):
    """Entry point for a preset (points to YAML body)."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(description="Relative path to the preset YAML inside the preset folder.")


class SecretsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: list[str] = Field(default_factory=list)


class CompatSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_core_version: str = ""


class ReportsSpec(BaseModel):
    """Report-plugin metadata — section ordering inside the rendered report.

    Only meaningful for ``plugin.kind == "reports"``. ``order`` must be
    unique across all registered report plugins; the registry enforces
    this at catalog load time.
    """

    model_config = ConfigDict(extra="forbid")

    order: int = Field(
        description="Global section order in the rendered Markdown report.",
    )


class PluginSpec(BaseModel):
    """Body of [plugin] section in plugin manifest.toml."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: PluginKind
    name: str = ""
    version: str = "1.0.0"
    category: str = ""
    stability: Stability = "stable"
    description: str = ""
    entry_point: EntryPoint

    @model_validator(mode="after")
    def _fill_name(self) -> PluginSpec:
        if not self.name:
            object.__setattr__(self, "name", self.id)
        return self


class PluginManifest(BaseModel):
    """Full plugin manifest loaded from manifest.toml."""

    model_config = ConfigDict(extra="forbid")

    plugin: PluginSpec
    params_schema: dict[str, Any] = Field(default_factory=dict)
    thresholds_schema: dict[str, Any] = Field(default_factory=dict)
    suggested_params: dict[str, Any] = Field(default_factory=dict)
    suggested_thresholds: dict[str, Any] = Field(default_factory=dict)
    secrets: SecretsSpec = Field(default_factory=SecretsSpec)
    compat: CompatSpec = Field(default_factory=CompatSpec)
    reports: ReportsSpec | None = None

    @model_validator(mode="after")
    def _check_suggested_against_schema(self) -> PluginManifest:
        self._assert_keys_subset(
            self.suggested_params, self.params_schema, label="suggested_params"
        )
        self._assert_keys_subset(
            self.suggested_thresholds,
            self.thresholds_schema,
            label="suggested_thresholds",
        )
        return self

    @model_validator(mode="after")
    def _reports_block_matches_kind(self) -> PluginManifest:
        """``[reports]`` is required for kind=reports plugins and forbidden
        for every other kind — the block is dead weight elsewhere."""
        if self.plugin.kind == "reports" and self.reports is None:
            raise ValueError(
                "[reports] block is required when plugin.kind == 'reports' "
                "(must provide the `order` field)"
            )
        if self.plugin.kind != "reports" and self.reports is not None:
            raise ValueError(
                f"[reports] block is only valid for plugin.kind == 'reports'; "
                f"got kind={self.plugin.kind!r}"
            )
        return self

    @staticmethod
    def _assert_keys_subset(
        suggestions: dict[str, Any],
        schema: dict[str, Any],
        *,
        label: str,
    ) -> None:
        if not schema:
            return
        extra = set(suggestions) - set(schema)
        if extra:
            raise ValueError(
                f"{label} keys {sorted(extra)} are not declared in the corresponding schema"
            )

    def ui_manifest(self) -> dict[str, Any]:
        """Flatten into the shape consumed by the web UI / ``/plugins`` API."""
        out: dict[str, Any] = {
            "id": self.plugin.id,
            "name": self.plugin.name,
            "version": self.plugin.version,
            "description": self.plugin.description,
            "category": self.plugin.category,
            "stability": self.plugin.stability,
            "params_schema": dict(self.params_schema),
            "thresholds_schema": dict(self.thresholds_schema),
            "suggested_params": dict(self.suggested_params),
            "suggested_thresholds": dict(self.suggested_thresholds),
        }
        if self.reports is not None:
            out["order"] = self.reports.order
        return out


class PresetScope(BaseModel):
    """Which top-level config keys the preset owns vs leaves alone.

    Apply semantics (see :mod:`src.community.preset_apply`):

    - ``replaces``  — these top-level keys are fully overwritten from the
      preset YAML. Anything the user had under these keys is discarded.
    - ``preserves`` — these keys are kept untouched; values present in the
      preset YAML for these keys are ignored.
    - Keys in the preset YAML that are in neither list fall through to
      ``replaces`` semantics (safe default: preset is authoritative over
      whatever it bothered to write down).

    When the whole ``[preset.scope]`` block is omitted, the preset runs
    in v1 compatibility mode — full replacement of the user config.
    """

    model_config = ConfigDict(extra="forbid")

    replaces: list[str] = Field(default_factory=list)
    preserves: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_overlap(self) -> PresetScope:
        overlap = set(self.replaces) & set(self.preserves)
        if overlap:
            raise ValueError(
                f"scope.replaces and scope.preserves overlap on {sorted(overlap)}"
            )
        return self


class PresetRequirements(BaseModel):
    """What the user's environment must provide for the preset to make sense."""

    model_config = ConfigDict(extra="forbid")

    # HF Hub models referenced by the preset; gated ones imply HF_TOKEN.
    hub_models: list[str] = Field(default_factory=list)
    # Preferred provider kinds (informational). Empty → preset is neutral.
    provider_kind: list[str] = Field(default_factory=list)
    # Community plugin ids the preset expects to be loaded (``kind:id``).
    required_plugins: list[str] = Field(default_factory=list)
    # Informational hardware hint (displayed in UI; not enforced).
    min_vram_gb: int | None = None


class PresetSpec(BaseModel):
    """Body of [preset] section in preset manifest.toml."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = ""
    description: str = ""
    size_tier: str = ""
    version: str = "1.0.0"
    entry_point: PresetEntryPoint
    # v2 contract — all three blocks are optional; absent = v1 full replace.
    scope: PresetScope | None = None
    requirements: PresetRequirements | None = None
    # Dotted JSONPath → hint. Surfaced in the preview modal so the user
    # knows which fields *they* must fill after applying.
    placeholders: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fill_name(self) -> PresetSpec:
        if not self.name:
            object.__setattr__(self, "name", self.id)
        return self


class PresetManifest(BaseModel):
    """Full preset manifest loaded from manifest.toml."""

    model_config = ConfigDict(extra="forbid")

    preset: PresetSpec


__all__ = [
    "CompatSpec",
    "EntryPoint",
    "PluginKind",
    "PluginManifest",
    "PluginSpec",
    "PresetEntryPoint",
    "PresetManifest",
    "PresetRequirements",
    "PresetScope",
    "PresetSpec",
    "SecretsSpec",
    "Stability",
]
