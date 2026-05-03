"""Reusable integration configurations (HuggingFace, MLflow).

Mirrors ``src/pipeline/settings/providers/`` — same file-backed registry
contract, same atomic-write + snapshot-per-save semantics. Kept as a
sibling module instead of a generic abstraction because "Provider" and
"Integration" are user-visible domain terms with distinct URLs, tabs
and API shapes; premature generalisation would hurt readability.
"""

from __future__ import annotations

from .models import (
    IntegrationConfigVersion,
    IntegrationMetadata,
    IntegrationRegistryEntry,
)
from .registry import IntegrationRegistry, IntegrationRegistryError
from .store import IntegrationStore, IntegrationStoreError

__all__ = [
    "IntegrationConfigVersion",
    "IntegrationMetadata",
    "IntegrationRegistry",
    "IntegrationRegistryEntry",
    "IntegrationRegistryError",
    "IntegrationStore",
    "IntegrationStoreError",
]
