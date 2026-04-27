"""Phase 12.A.1 — :class:`BufferedMetricsReplay` contract.

7-category coverage for the Mac-side replay logic that ships
buffered MLflow metrics from a retrieved
``metrics_buffer.jsonl`` into an MLflow run.

Slim-venv compatible: replay module only uses stdlib + a small
Protocol stand-in for :class:`mlflow.tracking.MlflowClient`.
Imports through the standard Python path; no ML stack required.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from src.pipeline.stages.model_retriever.metrics_replay import (
    BufferedMetricsReplay,
    ReplayResult,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeMlflowClient:
    """Records every ``log_metric`` call.

    Optionally raises a configured exception every Nth call so we
    can exercise the partial-failure path.
    """

    def __init__(
        self,
        *,
        raise_after: int | None = None,
        raise_exc: type[Exception] = RuntimeError,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._raise_after = raise_after
        self._raise_exc = raise_exc

    def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        timestamp: int | None = None,
        step: int | None = None,
    ) -> None:
        if (
            self._raise_after is not None
            and len(self.calls) >= self._raise_after
        ):
            raise self._raise_exc("simulated transport error")
        self.calls.append(
            {
                "run_id": run_id,
                "key": key,
                "value": value,
                "step": step,
                "timestamp": timestamp,
            }
        )


def _write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_replays_all_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics_buffer.jsonl"
        _write_jsonl(
            path,
            [
                {"key": "loss", "value": 0.5, "step": 1, "timestamp": 1.0},
                {"key": "loss", "value": 0.4, "step": 2, "timestamp": 2.0},
                {"key": "loss", "value": 0.3, "step": 3, "timestamp": 3.0},
            ],
        )
        client = _FakeMlflowClient()
        replayer = BufferedMetricsReplay(client)

        result = replayer.replay(buffer_path=path, run_id="run-abc")

        assert result.replayed == 3
        assert result.failed == 0
        assert result.skipped == 0
        assert result.first_step == 1
        assert result.last_step == 3
        assert len(client.calls) == 3
        # Each call carries the run_id we asked for.
        for call in client.calls:
            assert call["run_id"] == "run-abc"

    def test_timestamp_converted_to_milliseconds(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "metrics_buffer.jsonl"
        # Trainer writes timestamps in seconds (float). MLflow expects
        # milliseconds. Replay must convert.
        _write_jsonl(
            path, [{"key": "loss", "value": 0.5, "step": 1, "timestamp": 1.5}]
        )
        client = _FakeMlflowClient()
        replayer = BufferedMetricsReplay(client)

        replayer.replay(buffer_path=path, run_id="run-abc")
        assert client.calls[0]["timestamp"] == 1500


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_missing_file_returns_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "does_not_exist.jsonl"
        client = _FakeMlflowClient()
        replayer = BufferedMetricsReplay(client)

        result = replayer.replay(buffer_path=path, run_id="run-abc")

        assert result == ReplayResult(replayed=0, duration_ms=0)
        assert client.calls == []

    def test_malformed_json_lines_are_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "metrics_buffer.jsonl"
        path.write_text(
            '{"key":"loss","value":0.5,"step":1,"timestamp":1.0}\n'
            "this-is-not-json\n"
            '{"key":"loss","value":0.4,"step":2,"timestamp":2.0}\n',
            encoding="utf-8",
        )
        client = _FakeMlflowClient()
        replayer = BufferedMetricsReplay(client)

        result = replayer.replay(buffer_path=path, run_id="run-abc")

        assert result.replayed == 2
        assert result.skipped == 1
        assert any("malformed JSON" in err for err in result.errors)

    def test_log_metric_failures_are_captured_and_continue(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "metrics_buffer.jsonl"
        _write_jsonl(
            path,
            [
                {"key": "loss", "value": 0.5, "step": 1, "timestamp": 1.0},
                {"key": "loss", "value": 0.4, "step": 2, "timestamp": 2.0},
                {"key": "loss", "value": 0.3, "step": 3, "timestamp": 3.0},
            ],
        )
        client = _FakeMlflowClient(raise_after=1)
        replayer = BufferedMetricsReplay(client)

        result = replayer.replay(buffer_path=path, run_id="run-abc")

        # First call succeeded, the rest raised — we should keep going
        # and end up with `failed >= 2`.
        assert result.replayed == 1
        assert result.failed == 2
        assert any("log_metric" in err for err in result.errors)


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_empty_file_returns_zero_no_calls(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "metrics_buffer.jsonl"
        path.write_text("", encoding="utf-8")
        client = _FakeMlflowClient()
        replayer = BufferedMetricsReplay(client)

        result = replayer.replay(buffer_path=path, run_id="run-abc")

        assert result.replayed == 0
        assert result.first_step == -1
        assert result.last_step == -1
        assert client.calls == []

    def test_single_entry(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics_buffer.jsonl"
        _write_jsonl(
            path, [{"key": "loss", "value": 0.5, "step": 7, "timestamp": 1.0}]
        )
        client = _FakeMlflowClient()
        replayer = BufferedMetricsReplay(client)

        result = replayer.replay(buffer_path=path, run_id="run-abc")

        assert result.replayed == 1
        assert result.first_step == 7
        assert result.last_step == 7

    def test_thousand_entries_perf(self, tmp_path: Path) -> None:
        # Replay 1000 entries — this is the upper end of what we'd
        # see on a real long sleep run with keep_all=true. Should be
        # well under the perf budget.
        path = tmp_path / "metrics_buffer.jsonl"
        _write_jsonl(
            path,
            [
                {"key": "loss", "value": 0.5, "step": i, "timestamp": float(i)}
                for i in range(1000)
            ],
        )
        client = _FakeMlflowClient()
        replayer = BufferedMetricsReplay(client)

        result = replayer.replay(buffer_path=path, run_id="run-abc")

        assert result.replayed == 1000
        assert result.duration_ms < 5000  # generous: 5s for 1000 records


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_entries_replayed_in_step_order(
        self, tmp_path: Path
    ) -> None:
        # Buffer file may be appended out of order across breaker
        # cycles. Replay MUST sort by (step, timestamp) before
        # shipping so MLflow UI shows a monotonic step axis.
        path = tmp_path / "metrics_buffer.jsonl"
        _write_jsonl(
            path,
            [
                {"key": "loss", "value": 0.3, "step": 3, "timestamp": 3.0},
                {"key": "loss", "value": 0.5, "step": 1, "timestamp": 1.0},
                {"key": "loss", "value": 0.4, "step": 2, "timestamp": 2.0},
            ],
        )
        client = _FakeMlflowClient()
        replayer = BufferedMetricsReplay(client)

        replayer.replay(buffer_path=path, run_id="run-abc")

        steps = [call["step"] for call in client.calls]
        assert steps == [1, 2, 3]

    def test_run_id_is_passed_unchanged(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics_buffer.jsonl"
        _write_jsonl(
            path, [{"key": "loss", "value": 0.5, "step": 1, "timestamp": 1.0}]
        )
        client = _FakeMlflowClient()
        replayer = BufferedMetricsReplay(client)

        replayer.replay(buffer_path=path, run_id="any-arbitrary-id")
        assert client.calls[0]["run_id"] == "any-arbitrary-id"


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_unreadable_file_returns_replayed_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "metrics_buffer.jsonl"
        path.write_text(
            '{"key":"loss","value":0.5,"step":1,"timestamp":1.0}\n',
            encoding="utf-8",
        )

        # Force a read error.
        original_open = Path.open

        def _raise_on_open(self: Path, *args: Any, **kwargs: Any) -> Any:
            if self == path:
                raise OSError("simulated read failure")
            return original_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", _raise_on_open)

        client = _FakeMlflowClient()
        replayer = BufferedMetricsReplay(client)

        result = replayer.replay(buffer_path=path, run_id="run-abc")

        assert result.replayed == 0
        assert any("read failed" in err for err in result.errors)


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_legacy_buffer_without_step_field_defaults_to_zero(
        self, tmp_path: Path
    ) -> None:
        # Defensive against older-format buffer files (no `step`).
        path = tmp_path / "metrics_buffer.jsonl"
        _write_jsonl(
            path, [{"key": "loss", "value": 0.5, "timestamp": 1.0}]
        )
        client = _FakeMlflowClient()
        replayer = BufferedMetricsReplay(client)

        result = replayer.replay(buffer_path=path, run_id="run-abc")

        assert result.replayed == 1
        assert client.calls[0]["step"] == 0
        assert result.first_step == 0

    def test_idempotent_when_called_twice_on_same_file(
        self, tmp_path: Path
    ) -> None:
        # Replay does NOT mutate the source file — the caller (
        # ModelRetriever) is responsible for archival. A second
        # replay returns the same outcome (no MLflow dedup needed
        # because in real production the buffer file is moved out
        # of the way by the retriever).
        path = tmp_path / "metrics_buffer.jsonl"
        _write_jsonl(
            path, [{"key": "loss", "value": 0.5, "step": 1, "timestamp": 1.0}]
        )
        client = _FakeMlflowClient()
        replayer = BufferedMetricsReplay(client)

        first = replayer.replay(buffer_path=path, run_id="run-abc")
        second = replayer.replay(buffer_path=path, run_id="run-abc")

        assert first.replayed == 1 == second.replayed
        # The source file is preserved on the Mac for forensics.
        assert path.exists()


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_zero_timestamp_converts_to_none(self, tmp_path: Path) -> None:
        # Timestamp `0` (or missing) => pass `None` to log_metric so
        # MLflow uses its server-side ingest time. This keeps the
        # MLflow UI ordering consistent.
        path = tmp_path / "metrics_buffer.jsonl"
        _write_jsonl(
            path, [{"key": "loss", "value": 0.5, "step": 1, "timestamp": 0.0}]
        )
        client = _FakeMlflowClient()
        replayer = BufferedMetricsReplay(client)

        replayer.replay(buffer_path=path, run_id="run-abc")
        assert client.calls[0]["timestamp"] is None

    def test_errors_capped_at_ten(self, tmp_path: Path) -> None:
        # 50 malformed lines → only first 10 errors retained.
        path = tmp_path / "metrics_buffer.jsonl"
        path.write_text("\n".join(["not-json"] * 50) + "\n", encoding="utf-8")
        client = _FakeMlflowClient()
        replayer = BufferedMetricsReplay(client)

        result = replayer.replay(buffer_path=path, run_id="run-abc")

        assert result.skipped == 50
        assert len(result.errors) == 10

    def test_completes_when_log_metric_partial_failure(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Even if every other call fails, replay continues to the end
        # — never raises into caller, captures errors, returns
        # `failed > 0`.
        path = tmp_path / "metrics_buffer.jsonl"
        _write_jsonl(
            path,
            [
                {"key": f"k{i}", "value": float(i), "step": i, "timestamp": 1.0}
                for i in range(20)
            ],
        )
        client = _FakeMlflowClient(raise_after=5)
        replayer = BufferedMetricsReplay(client)
        with caplog.at_level(logging.WARNING):
            result = replayer.replay(buffer_path=path, run_id="run-abc")

        assert result.replayed == 5
        assert result.failed == 15
        assert any(
            "metric writes failed" in record.message
            for record in caplog.records
        )
