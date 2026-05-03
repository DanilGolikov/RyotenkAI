"""
Dataset Validation Plugin System.

Provides pluggable architecture for dataset validation with:
- Base plugin interface (ValidationPlugin)
- Plugin registry populated by ``src.community.catalog``
- MLflow event integration
- Support for streaming datasets
"""

from ryotenkai_control.data.validation.base import ValidationPlugin, ValidationResult
from ryotenkai_control.data.validation.registry import ValidationPluginRegistry, validation_registry

__all__ = [
    "ValidationPlugin",
    "ValidationPluginRegistry",
    "ValidationResult",
    "validation_registry",
]
