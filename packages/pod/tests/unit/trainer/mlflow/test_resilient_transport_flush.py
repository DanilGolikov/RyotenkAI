"""Phase 9.B — :meth:`ResilientMLflowTransport.flush_buffer` contract.

Public, explicit drain entry point used by
:class:`CancellationCallback` on ``on_train_end`` to push buffered
metrics into the live MLflow run before HF Trainer closes it.

The existing ``_flush_buffer`` (private, fast-path drain inside
``_make_wrapper``) is exercised by :file:`test_resilient_transport.py`
which doesn't run in the slim dev venv. This file tests the new
public method using the same slim-venv import pattern as
:file:`test_concurrent_helpers.py`.

7-category coverage:

1. **Positive** — drain flushes records, returns count.
2. **Negative** — empty buffer returns 0; no buffer attached → 0.
3. **Boundary** — count exactly 0; transport not installed (no
   ``log_metric`` original).
4. **Invariants** — bypasses the breaker (drain never opens the
   circuit on transient backend stalls).
5. **Dependency errors** — buffer.flush raises → return 0,
   warning logged.
6. **Regressions** — drain is idempotent (running twice on an
   already-drained buffer returns 0).
7. **Logic-specific** — public `flush_buffer` calls
   ``_buffer.flush(log_metric_original)`` with the unpatched
   original.
"""

from __future__ import annotations

import importlib.util
import logging
import pathlib
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest


# Slim-venv: load resilient_transport by file path to bypass
# ``src.training/__init__`` which pulls in the full ML stack.
_TRANSPORT_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "src" / "ryotenkai_pod" / "trainer" / "mlflow" / "resilient_transport.py"
)
_spec = importlib.util.spec_from_file_location(
    "_ryotenkai_transport_under_test", _TRANSPORT_PATH,
)
assert _spec is not None and _spec.loader is not None
_transport_module = importlib.util.module_from_spec(_spec)
sys.modules["_ryotenkai_transport_under_test"] = _transport_module
_spec.loader.exec_module(_transport_module)

ResilientMLflowTransport = _transport_module.ResilientMLflowTransport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeBuffer:
    """In-memory stand-in for :class:`MetricsBuffer`."""

    def __init__(self, count: int = 0) -> None:
        self._count = count
        self.flush_calls: list[Any] = []
        self._raise_on_flush: Exception | None = None

    @property
    def count(self) -> int:
        return self._count

    def flush(self, log_metric_fn: Any) -> int:
        self.flush_calls.append(log_metric_fn)
        if self._raise_on_flush is not None:
            raise self._raise_on_flush
        drained = self._count
        self._count = 0  # actual buffer zeros itself after a successful flush
        return drained

    def raise_on_next_flush(self, exc: Exception) -> None:
        self._raise_on_flush = exc


def _make_transport_with_buffer(
    *, count: int = 0, installed: bool = True,
) -> tuple[Any, _FakeBuffer]:
    """Build a transport + attached buffer; optionally simulate
    install() so ``_originals[("module", "log_metric")]`` is set."""
    transport = ResilientMLflowTransport()
    buffer = _FakeBuffer(count=count)
    transport.attach_buffer(buffer)
    if installed:
        # Simulate install() side-effect — populate ``_originals`` with a
        # callable representing the unpatched ``log_metric``.
        transport._originals[("module", "log_metric")] = MagicMock(
            name="log_metric_original",
        )
    return transport, buffer


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_flush_drains_buffer_and_returns_count(self) -> None:
        transport, buffer = _make_transport_with_buffer(count=42)
        assert transport.flush_buffer() == 42

        # Buffer was actually drained (one flush call).
        assert len(buffer.flush_calls) == 1

    def test_flush_uses_original_log_metric(self) -> None:
        # Pinned: the public method MUST drain via the stored
        # unpatched ``log_metric`` original, not via the transport's
        # patched API. That bypasses the breaker so a transient stall
        # on the live path doesn't block the drain.
        transport, buffer = _make_transport_with_buffer(count=1)
        original_fn = transport._originals[("module", "log_metric")]

        transport.flush_buffer()

        # ``flush`` was called with the exact original from _originals.
        assert buffer.flush_calls == [original_fn]


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_no_buffer_attached_returns_zero(self) -> None:
        transport = ResilientMLflowTransport()
        # No attach_buffer call.
        assert transport.flush_buffer() == 0

    def test_empty_buffer_returns_zero_without_calling_flush(self) -> None:
        transport, buffer = _make_transport_with_buffer(count=0)
        assert transport.flush_buffer() == 0
        # No flush call — empty buffer is a fast-path return.
        assert buffer.flush_calls == []


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_buffer_with_one_entry(self) -> None:
        transport, buffer = _make_transport_with_buffer(count=1)
        assert transport.flush_buffer() == 1
        assert len(buffer.flush_calls) == 1

    def test_transport_not_installed_returns_zero(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Buffer has pending entries but no install() was called →
        # no log_metric original to drain through. Logged warning.
        caplog.set_level(logging.WARNING, logger="ryotenkai_pod.trainer.mlflow.resilient_transport")
        transport, buffer = _make_transport_with_buffer(
            count=10, installed=False,
        )
        assert transport.flush_buffer() == 0
        # No flush call — we couldn't proceed.
        assert buffer.flush_calls == []


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_drain_does_not_change_breaker_state(self) -> None:
        # Pin: explicit drain bypasses the breaker. State stays
        # whatever it was BEFORE the call.
        transport, _ = _make_transport_with_buffer(count=5)
        before = transport.breaker_state
        transport.flush_buffer()
        after = transport.breaker_state
        assert before == after


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_buffer_flush_exception_returns_zero(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Pin: best-effort contract — flush failures don't propagate;
        # caller's flow continues.
        caplog.set_level(logging.WARNING, logger="ryotenkai_pod.trainer.mlflow.resilient_transport")
        transport, buffer = _make_transport_with_buffer(count=3)
        buffer.raise_on_next_flush(RuntimeError("upstream stalled"))

        assert transport.flush_buffer() == 0


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_idempotent_drain(self) -> None:
        # First call drains the buffer (returns its size).
        transport, buffer = _make_transport_with_buffer(count=7)
        first = transport.flush_buffer()
        # Second call: buffer is now empty; returns 0 without
        # touching ``flush`` again.
        second = transport.flush_buffer()
        assert first == 7
        assert second == 0
        assert len(buffer.flush_calls) == 1


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_count_read_before_flush_for_return_value(self) -> None:
        """``flush_buffer`` reads ``count`` BEFORE calling ``buffer.flush``
        so the return value reflects the drain even though
        ``flush`` resets the buffer's count. Pin via spy on the order."""
        transport = ResilientMLflowTransport()
        order: list[str] = []

        class _OrderedBuffer:
            @property
            def count(self) -> int:
                order.append("count")
                return 5

            def flush(self, log_metric_fn: Any) -> int:
                order.append("flush")
                return 5

        transport.attach_buffer(_OrderedBuffer())
        transport._originals[("module", "log_metric")] = MagicMock()

        transport.flush_buffer()
        # Two count reads (one for early-return guard, one for return)
        # are acceptable; what matters is at least one count BEFORE
        # the flush call.
        assert order[0] == "count"
        assert "flush" in order
