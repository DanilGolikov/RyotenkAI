"""
Template provider configuration schema.

Copy and modify this for your provider.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class TemplateProviderConfig(BaseModel):
    """
    Configuration for Template provider.

    Modify this class for your provider's configuration.

    Required fields:
        - type: Must match your provider type string

    Add your provider-specific fields here.
    """

    # Provider type (CHANGE THIS to your type)
    type: str = Field("template", description="Provider type")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Ensure type matches your provider."""
        # CHANGE "template" to your provider type
        if v != "template":
            raise ValueError(f"TemplateProviderConfig type must be 'template', got '{v}'")
        return v

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateProviderConfig:
        """Create config from dictionary."""
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(exclude_none=True)


__all__ = [
    "TemplateProviderConfig",
]
