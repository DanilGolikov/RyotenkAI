"""
Dataset Validation Plugin System.

Provides pluggable architecture for dataset validation with:
- Base plugin interface (ValidationPlugin)
- Plugin registry populated by ``src.community.catalog``
- MLflow event integration
- Support for streaming datasets
"""

from src.data.validation.base import ValidationPlugin, ValidationResult
from src.data.validation.registry import ValidationPluginRegistry, validation_registry

__all__ = [
    "ValidationPlugin",
    "ValidationPluginRegistry",
    "ValidationResult",
    "validation_registry",
]
