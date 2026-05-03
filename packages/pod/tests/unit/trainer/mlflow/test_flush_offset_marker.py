"""Phase 12.A.1 — flush-offset marker contract.

After every successful drain (implicit fast-path drain inside
``_make_wrapper`` OR explicit ``flush_buffer``) the resilient
transport writes ``<workspace>/.runner/buffer.flush_offset`` so
Mac-side :class:`MetricsBufferRetriever` can fetch it for
forensics ("did the trainer ever drain?").

This file pins the marker shape and write conditions. The marker
is purely informational — ``BufferedMetricsReplay`` does NOT
consult it; correctness comes from the buffer-file invariant
(un-flushed entries only).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest


# Slim-venv: load resilient_transport directly to bypass
# ``src.training/__init__`` (which pulls the full ML stack).
_TRANSPORT_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "src" / "ryotenkai_pod" / "trainer" / "mlflow" / "resilient_transport.py"
)
_spec = importlib.util.spec_from_file_location(
    "_ryotenkai_transport_flush_offset", _TRANSPORT_PATH,
)
assert _spec is not None and _spec.loader is not None
_transport_module = importlib.util.module_from_spec(_spec)
sys.modules["_ryotenkai_transport_flush_offset"] = _transport_module
_spec.loader.exec_module(_transport_module)

ResilientMLflowTransport = _transport_module.ResilientMLflowTransport


class _FakeBuffer:
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
        self._count = 0
        return drained

    def raise_on_next_flush(self, exc: Exception) -> None:
        self._raise_on_flush = exc


def _make_installed_transport(
    *, count: int = 0,
) -> tuple[Any, _FakeBuffer]:
    transport = ResilientMLflowTransport()
    buffer = _FakeBuffer(count=count)
    transport.attach_buffer(buffer)
    transport._originals[("module", "log_metric")] = MagicMock()
    return transport, buffer


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_marker_written_after_explicit_flush(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WORKSPACE_PATH", str(tmp_path))
        transport, buffer = _make_installed_transport(count=12)

        transport.flush_buffer()

        marker = tmp_path / ".runner" / "buffer.flush_offset"
        assert marker.exists()

        payload = json.loads(marker.read_text())
        assert payload["v"] == 1
        assert payload["drained_count"] == 12
        assert isinstance(payload["drained_at_ms"], int)
        assert payload["drained_at_ms"] > 0

    def test_marker_dir_created_if_missing(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `.runner/` doesn't exist yet — write should mkdir.
        monkeypatch.setenv("WORKSPACE_PATH", str(tmp_path))
        transport, _ = _make_installed_transport(count=1)

        transport.flush_buffer()

        assert (tmp_path / ".runner").is_dir()


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_marker_NOT_written_when_buffer_empty(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Empty buffer → fast-path return without writing the marker.
        # Otherwise we'd see "drained_count: 0" markers spamming disk
        # on every cancellation hook hit when nothing was buffered.
        monkeypatch.setenv("WORKSPACE_PATH", str(tmp_path))
        transport, _ = _make_installed_transport(count=0)

        transport.flush_buffer()

        marker = tmp_path / ".runner" / "buffer.flush_offset"
        assert not marker.exists()

    def test_marker_NOT_written_when_flush_raises(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Failed drain → marker would be a lie. Don't write.
        monkeypatch.setenv("WORKSPACE_PATH", str(tmp_path))
        transport, buffer = _make_installed_transport(count=5)
        buffer.raise_on_next_flush(RuntimeError("upstream stalled"))

        transport.flush_buffer()

        marker = tmp_path / ".runner" / "buffer.flush_offset"
        assert not marker.exists()


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_workspace_env_default_when_unset(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # WORKSPACE_PATH unset → defaults to /workspace.
        # The actual write to /workspace/.runner will fail silently
        # (no perms in test env) — and that's OK by contract: marker
        # is best-effort. Verify no exception bubbles up.
        monkeypatch.delenv("WORKSPACE_PATH", raising=False)
        transport, _ = _make_installed_transport(count=1)

        # Must not raise.
        transport.flush_buffer()


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_marker_payload_is_compact_json(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Marker is tiny — pin to compact JSON to keep disk overhead
        # at < 100 bytes per write.
        monkeypatch.setenv("WORKSPACE_PATH", str(tmp_path))
        transport, _ = _make_installed_transport(count=1)
        transport.flush_buffer()

        marker = tmp_path / ".runner" / "buffer.flush_offset"
        text = marker.read_text()
        assert "\n" not in text  # no pretty-printing
        assert " " not in text   # no whitespace separators

    def test_atomic_replace_no_tmp_leftover(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Write should rename .tmp → final atomically.
        monkeypatch.setenv("WORKSPACE_PATH", str(tmp_path))
        transport, _ = _make_installed_transport(count=1)
        transport.flush_buffer()

        # No leftover tmp file.
        runner_dir = tmp_path / ".runner"
        leftovers = [p.name for p in runner_dir.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_marker_write_failure_does_not_break_flush(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Read-only "workspace" — marker write fails, but flush_buffer
        # MUST still return drained count and succeed semantically.
        ro_workspace = tmp_path / "ro"
        ro_workspace.mkdir()
        monkeypatch.setenv("WORKSPACE_PATH", str(ro_workspace))

        transport, _ = _make_installed_transport(count=3)

        # Force OSError on the marker write.
        original_replace = _transport_module.os.replace

        def _raise_on_replace(*args: Any, **kwargs: Any) -> Any:
            raise PermissionError("simulated EROFS")

        monkeypatch.setattr(_transport_module.os, "replace", _raise_on_replace)

        # Should NOT raise.
        result = transport.flush_buffer()
        assert result == 3

        # Restore (defensive — pytest does this anyway)
        monkeypatch.setattr(_transport_module.os, "replace", original_replace)


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_marker_overwritten_on_subsequent_flush(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two flush-buffer cycles → marker reflects the LAST drain,
        # not accumulated. Important so operator sees the most recent
        # drain timestamp.
        monkeypatch.setenv("WORKSPACE_PATH", str(tmp_path))

        transport, buffer = _make_installed_transport(count=5)
        transport.flush_buffer()
        first = json.loads(
            (tmp_path / ".runner" / "buffer.flush_offset").read_text()
        )

        # Re-pump the buffer and flush again with a different count.
        buffer._count = 17
        transport.flush_buffer()
        second = json.loads(
            (tmp_path / ".runner" / "buffer.flush_offset").read_text()
        )

        assert first["drained_count"] == 5
        assert second["drained_count"] == 17
        assert second["drained_at_ms"] >= first["drained_at_ms"]


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_drained_at_ms_within_realistic_range(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time
        monkeypatch.setenv("WORKSPACE_PATH", str(tmp_path))

        before = int(time.time() * 1000)
        transport, _ = _make_installed_transport(count=1)
        transport.flush_buffer()
        after = int(time.time() * 1000)

        marker = json.loads(
            (tmp_path / ".runner" / "buffer.flush_offset").read_text()
        )
        # Wall-clock ms timestamp: between "before" and "after".
        assert before <= marker["drained_at_ms"] <= after
