"""
Centralized cross-field ("if X then Y") validators for config models.

Design:
- Each config model keeps a SINGLE `@model_validator(mode="after")` method with a common name
  (we use `_run_model_validators`), which enumerates all validators to run for that model.
- Validator implementations live here as small pure functions that accept a config instance and
  raise `ValueError` with a clear message.
- To avoid import cycles, validator modules MUST NOT import config models at runtime.
  Use `TYPE_CHECKING` + forward references for typing only.
"""

from . import cross, datasets, inference, pipeline, providers, training

__all__ = [
    "cross",
    "datasets",
    "inference",
    "pipeline",
    "providers",
    "training",
]
