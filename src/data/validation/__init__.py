"""
Dataset Validation Plugin System.

Provides pluggable architecture for dataset validation with:
- Base plugin interface (ValidationPlugin)
- Plugin registry for auto-discovery
- MLflow event integration
- Support for streaming datasets
"""

from src.data.validation.base import ValidationPlugin, ValidationResult
from src.data.validation.discovery import ensure_validation_plugins_discovered
from src.data.validation.registry import ValidationPluginRegistry

__all__ = [
    "ValidationPlugin",
    "ValidationPluginRegistry",
    "ValidationResult",
    "ensure_validation_plugins_discovered",
]
