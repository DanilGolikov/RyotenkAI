"""
Base building blocks for configuration schema.

The goal is to keep this module dependency-light (no imports from higher-level config modules),
so it can be safely imported everywhere without circular dependencies.
"""

from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    """
    BaseModel with strict schema.

    We intentionally forbid unknown fields to prevent silent acceptance of legacy/typo config keys.
    """

    model_config = ConfigDict(
        extra="forbid",
        # Allow populating by field name even when aliases are defined.
        # This keeps Python-side construction ergonomic while preserving YAML aliases (e.g. deltaT).
        populate_by_name=True,
    )


__all__ = [
    "StrictBaseModel",
]
