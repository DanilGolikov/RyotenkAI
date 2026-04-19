"""Provider schema registry.

Single source of truth mapping provider `type` string → Pydantic schema
class plus UI metadata (label) and error-code suffixes used by
cross-validators.

Purpose:
- Replace hardcoded if/elif chains in ``src/config/validators/cross.py``
  without changing any error codes — existing tests keep passing.
- Give the web API a single place to enumerate the supported types
  (``GET /providers/types``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.config.providers.runpod import RunPodProviderConfig
from src.config.providers.single_node import SingleNodeConfig
from src.constants import PROVIDER_RUNPOD, PROVIDER_SINGLE_NODE

if TYPE_CHECKING:
    from src.config.base import StrictBaseModel


@dataclass(slots=True)
class ProviderType:
    """Metadata for a single provider type supported by PipelineConfig."""

    id: str
    label: str
    schema: type
    schema_name: str
    training_error_code: str
    inference_error_code: str


PROVIDER_TYPES: dict[str, ProviderType] = {
    PROVIDER_SINGLE_NODE: ProviderType(
        id=PROVIDER_SINGLE_NODE,
        label="Single node (local GPU)",
        schema=SingleNodeConfig,
        schema_name=SingleNodeConfig.__name__,
        training_error_code="CONFIG_SINGLE_NODE_PROVIDER_INVALID",
        inference_error_code="CONFIG_INFERENCE_SINGLE_NODE_INVALID",
    ),
    PROVIDER_RUNPOD: ProviderType(
        id=PROVIDER_RUNPOD,
        label="RunPod (cloud GPU)",
        schema=RunPodProviderConfig,
        schema_name=RunPodProviderConfig.__name__,
        training_error_code="CONFIG_RUNPOD_PROVIDER_INVALID",
        inference_error_code="CONFIG_INFERENCE_RUNPOD_INVALID",
    ),
}


def get_provider_type(type_id: str) -> ProviderType | None:
    return PROVIDER_TYPES.get(type_id)


def get_provider_schema(type_id: str) -> type[StrictBaseModel] | None:
    entry = PROVIDER_TYPES.get(type_id)
    return entry.schema if entry else None


__all__ = [
    "PROVIDER_TYPES",
    "ProviderType",
    "get_provider_schema",
    "get_provider_type",
]
