"""Backward-compat shim — moved to ``ryotenkai_shared.config.loader`` (ADR row 5)."""

from __future__ import annotations

from ryotenkai_shared.config.loader import load_pipeline_config

__all__ = ["load_pipeline_config"]
