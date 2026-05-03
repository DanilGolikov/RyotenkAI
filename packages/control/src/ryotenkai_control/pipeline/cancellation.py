"""Backward-compat shim — canonical home moved to ``src.utils.cancellation``.

Removed at the start of Phase B (monorepo packagization, see
``docs/plans/2026-05-03-monorepo-uv-workspace-packagization.md`` §B.4).
"""

from __future__ import annotations

from src.utils.cancellation import (
    PipelineCancelled,
    check_cancelled,
    get_active_orchestrator,
    install_handler,
    is_cancelled,
    reset_for_tests,
    set_active_orchestrator,
    sleep_cancellable,
)

__all__ = [
    "PipelineCancelled",
    "check_cancelled",
    "get_active_orchestrator",
    "install_handler",
    "is_cancelled",
    "reset_for_tests",
    "set_active_orchestrator",
    "sleep_cancellable",
]
