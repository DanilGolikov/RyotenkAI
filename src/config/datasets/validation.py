from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from ..base import StrictBaseModel


def _default_validation_apply_to() -> list[Literal["train", "eval"]]:
    return ["train", "eval"]


class DatasetValidationPluginConfig(StrictBaseModel):
    """
    Single validation plugin instance config.

    Contract:
    - `params` contains runtime and execution settings
    - `thresholds` contains pass/fail criteria

    Both blocks stay intentionally dynamic; concrete plugins validate their
    own required and optional keys.
    """

    id: str = Field(..., description="Unique instance id used in metrics, artifacts, and reports.")
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
        seen: set[str] = set()
        for plugin in plugins:
            if plugin.id in seen:
                raise ValueError(f"Duplicate validation plugin id: {plugin.id!r}")
            seen.add(plugin.id)
        return plugins


__all__ = [
    "DatasetValidationPluginConfig",
    "DatasetValidationsConfig",
]
