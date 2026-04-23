"""Integration schema registry.

Single source of truth mapping integration ``type`` string → Pydantic
schema class plus UI metadata. Mirrors ``src/config/providers/registry.py``.

Purpose:
- Give the web API a single place to enumerate the supported integration
  types (``GET /integrations/types``).
- Let the UI render an integration's Config form off the JSON schema
  generated from the Pydantic class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.config.integrations.huggingface_integration import HuggingFaceIntegrationConfig
from src.config.integrations.mlflow_integration import MLflowIntegrationConfig

if TYPE_CHECKING:
    from src.config.base import StrictBaseModel

INTEGRATION_TYPE_MLFLOW = "mlflow"
INTEGRATION_TYPE_HUGGINGFACE = "huggingface"


@dataclass(slots=True)
class IntegrationType:
    """Metadata for a single integration type supported in Settings."""

    id: str
    label: str
    schema: type
    schema_name: str
    requires_token: bool


INTEGRATION_TYPES: dict[str, IntegrationType] = {
    INTEGRATION_TYPE_MLFLOW: IntegrationType(
        id=INTEGRATION_TYPE_MLFLOW,
        label="MLflow tracking",
        schema=MLflowIntegrationConfig,
        schema_name=MLflowIntegrationConfig.__name__,
        requires_token=False,  # token optional for unauthenticated servers
    ),
    INTEGRATION_TYPE_HUGGINGFACE: IntegrationType(
        id=INTEGRATION_TYPE_HUGGINGFACE,
        label="HuggingFace Hub",
        schema=HuggingFaceIntegrationConfig,
        schema_name=HuggingFaceIntegrationConfig.__name__,
        requires_token=True,
    ),
}


def get_integration_type(type_id: str) -> IntegrationType | None:
    return INTEGRATION_TYPES.get(type_id)


def get_integration_schema(type_id: str) -> type[StrictBaseModel] | None:
    entry = INTEGRATION_TYPES.get(type_id)
    return entry.schema if entry else None  # type: ignore[return-value]


__all__ = [
    "INTEGRATION_TYPES",
    "INTEGRATION_TYPE_HUGGINGFACE",
    "INTEGRATION_TYPE_MLFLOW",
    "IntegrationType",
    "get_integration_schema",
    "get_integration_type",
]
