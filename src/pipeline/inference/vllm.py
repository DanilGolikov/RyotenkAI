"""Backward-compat shim — canonical home moved to ``src.providers.inference.vllm``.

Removed at the start of Phase B (monorepo packagization, see
``docs/plans/2026-05-03-monorepo-uv-workspace-packagization.md`` §B.4).
"""

from __future__ import annotations

from src.providers.inference.vllm.engine import VLLMEngine

__all__ = ["VLLMEngine"]
