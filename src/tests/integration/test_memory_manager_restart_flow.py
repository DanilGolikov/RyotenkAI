"""
Integration tests: MemoryManager.safe_operation contract (OOM/restart-relevant behavior).

Goal:
- Make sure safe_operation translates "CUDA out of memory" RuntimeError into OOMRecoverableError
  and triggers cleanup + callbacks.
- Verify pre-operation critical memory path clears cache.

These tests do NOT require a real GPU: we patch get_memory_stats/is_memory_critical.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.utils.memory_manager import MemoryManager, MemoryStats, OOMRecoverableError


def test_safe_operation_oom_becomes_recoverable_error_and_triggers_cleanup_and_callback() -> None:
    events: list[tuple[str, object]] = []

    callbacks = MagicMock()
    callbacks.on_oom = lambda operation, free_mb: events.append((operation, free_mb))
    callbacks.on_memory_warning = None
    callbacks.on_cache_cleared = None
    callbacks.on_gpu_detected = None
    callbacks.on_oom_retry = None

    mm = MemoryManager(callbacks=callbacks)

    # Avoid any GPU dependency
    stats = MemoryStats(total_mb=8000, free_mb=123, used_mb=7877, utilization_percent=98.0, gpu_name="Test GPU")
    mm.get_memory_stats = MagicMock(return_value=stats)
    mm.is_memory_critical = MagicMock(return_value=False)
    mm.clear_cache = MagicMock()
    mm.checkpoint = MagicMock()
    mm.log_memory_status = MagicMock()
    mm.aggressive_cleanup = MagicMock()

    with pytest.raises(OOMRecoverableError) as exc_info:
        with mm.safe_operation("train_phase_0", context={"batch_size": 4}):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB...")

    assert exc_info.value.operation == "train_phase_0"
    mm.aggressive_cleanup.assert_called_once()
    assert events == [("train_phase_0", 123)]


def test_safe_operation_clears_cache_when_memory_is_critical() -> None:
    mm = MemoryManager()

    # Avoid any GPU dependency
    stats = MemoryStats(total_mb=8000, free_mb=100, used_mb=7900, utilization_percent=98.0, gpu_name="Test GPU")
    mm.get_memory_stats = MagicMock(return_value=stats)
    mm.is_memory_critical = MagicMock(return_value=True)
    mm.clear_cache = MagicMock()
    mm.checkpoint = MagicMock()

    with mm.safe_operation("create_trainer_phase_0"):
        pass

    mm.clear_cache.assert_called_once()


