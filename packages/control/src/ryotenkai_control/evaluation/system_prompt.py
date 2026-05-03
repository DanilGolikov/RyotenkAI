"""Backward-compat shim — moved to ``ryotenkai_shared.infrastructure.mlflow.system_prompt`` (ADR row 9)."""

from __future__ import annotations

from ryotenkai_shared.infrastructure.mlflow.system_prompt import (
    SystemPromptLoader,
    SystemPromptResult,
)

__all__ = ["SystemPromptLoader", "SystemPromptResult"]
