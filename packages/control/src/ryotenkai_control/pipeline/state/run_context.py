"""Backward-compat shim — RunContext moved to ``ryotenkai_shared.pipeline_context`` (ADR row 9)."""

from __future__ import annotations

from ryotenkai_shared.pipeline_context import RunContext

__all__ = ["RunContext"]
