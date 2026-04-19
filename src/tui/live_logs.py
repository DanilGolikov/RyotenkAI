"""Backward-compatibility shim. Real module lives in src/pipeline/live_logs."""

from __future__ import annotations

from src.pipeline.live_logs import LiveLogTail

__all__ = ["LiveLogTail"]
