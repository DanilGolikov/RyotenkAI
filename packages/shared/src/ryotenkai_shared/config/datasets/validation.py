from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ..base import StrictBaseModel


def _default_validation_apply_to() -> list[Literal["train", "eval"]]:
    return ["train", "eval"]


def _autogen_validator_id(plugin: str, params: dict[str, Any]) -> str:
    """Stable id from plugin name + params hash. Same shape as
    :func:`ryotenkai_shared.config.evaluation.schema._autogen_plugin_id`."""
    payload = json.dumps(params, sort_keys=True, default=str).encode()
    digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()[:8]
    return f"{plugin}_{digest}"


class DatasetValidationPluginConfig(StrictBaseModel):
    """
    Single validation plugin instance config.

    Contract:
    - `params` contains runtime and execution settings
    - `thresholds` contains pass/fail criteria

    Both blocks stay intentionally dynamic; concrete plugins validate their
    own required and optional keys.

    ``id`` is OPTIONAL — auto-generated as
    ``f"{plugin}_{md5(params)[:8]}"`` when not supplied. Override with an
    explicit string for human-readable names that survive across param
    changes. Auto-id collisions within a parent ``plugins`` list (rare:
    identical plugin+params) are resolved with ``_2`` / ``_3`` suffixes.
    """

    id: str | None = Field(
        default=None,
        description=(
            "Optional unique instance id used in metrics, artifacts, and "
            "reports. Auto-generated as ``f'{plugin}_{md5(params)[:8]}'`` "
            "when not supplied."
        ),
    )
    plugin: str = Field(..., description="Registered validation plugin name, e.g. 'min_samples'.")
    apply_to: list[Literal["train", "eval"]] = Field(
        default_factory=_default_validation_apply_to,
        description="Where to apply plugin. Default: ['train','eval'] (applied only to non-null datasets).",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime and execution settings for the plugin (dynamic).",
    )
    thresholds: dict[str, Any] = Field(
        default_factory=dict,
        description="Pass/fail criteria for the plugin (dynamic).",
    )

    @field_validator("apply_to")
    @classmethod
    def validate_apply_to_non_empty(cls, v: list[Literal["train", "eval"]]) -> list[Literal["train", "eval"]]:
        if not v:
            raise ValueError("apply_to must contain at least one of ['train','eval']")
        seen: set[Literal["train", "eval"]] = set()
        out: list[Literal["train", "eval"]] = []
        for item in v:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    @model_validator(mode="after")
    def _autofill_id(self) -> DatasetValidationPluginConfig:
        if not self.id:
            self.id = _autogen_validator_id(self.plugin, self.params)
        return self


class DatasetValidationsConfig(StrictBaseModel):
    """Dataset validations config."""

    critical_failures: int = Field(
        0,
        ge=0,
        description="Number of failed plugins to stop validation (0 = never stop).",
    )
    mode: Literal["fast", "full"] = Field(
        "fast",
        description="Validation mode: 'fast' = sample, 'full' = entire dataset.",
    )
    plugins: list[DatasetValidationPluginConfig] = Field(
        default_factory=list,
        description="Validation plugins list.",
    )

    @field_validator("plugins")
    @classmethod
    def validate_unique_plugin_ids(
        cls,
        plugins: list[DatasetValidationPluginConfig],
    ) -> list[DatasetValidationPluginConfig]:
        """Resolve duplicate auto-ids by suffixing _2/_3/...; explicit
        id collisions raise."""
        seen_explicit: set[str] = set()
        for plugin in plugins:
            assert plugin.id is not None  # filled by _autofill_id
            auto_id = _autogen_validator_id(plugin.plugin, plugin.params)
            if plugin.id != auto_id:
                if plugin.id in seen_explicit:
                    raise ValueError(
                        f"Duplicate explicit validation plugin id: {plugin.id!r}"
                    )
                seen_explicit.add(plugin.id)

        seen_count: dict[str, int] = {}
        for plugin in plugins:
            base = plugin.id or ""
            n = seen_count.get(base, 0)
            if n > 0:
                plugin.id = f"{base}_{n + 1}"
            seen_count[base] = n + 1

        return plugins


__all__ = [
    "DatasetValidationPluginConfig",
    "DatasetValidationsConfig",
]
