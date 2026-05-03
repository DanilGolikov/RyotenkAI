"""Phase 9.C — :meth:`TrainingMonitor._reconcile_terminal_marker_if_present`.

When a trainer SIGKILLs before finishing its on_train_end MLflow flush
the callback writes ``attempts/<n>/cancelled.marker`` to the workspace
on the pod. The runtime keeps that file alive on the SSH-mounted
attempt dir; the Mac-side TrainingMonitor reads it after the WS event
stream returns and reconciles the upstream MLflow run state if needed.

The reconciliation is **always best-effort**: any I/O / MLflow error
is logged and swallowed. The only contract is that the marker must
be acted upon when (a) it exists, (b) carries a valid ``run_id``, and
(c) the upstream MLflow run is still ``RUNNING``. In every other
shape (missing marker, missing run_id, MLflow already terminal,
MLflow unreachable) it must be a silent no-op.

7-category coverage:

1. Positive       — marker + RUNNING → set_terminated(KILLED) called.
2. Negative       — no marker → no MLflow client touched at all.
3. Boundary       — marker without ``run_id`` → log + skip.
                   — marker with corrupt JSON → log + skip.
4. Invariants     — function NEVER raises, even on MLflow exceptions.
5. Dependency     — MlflowClient.get_run raising → swallowed.
6. Regressions    — already-terminal upstream → no double-set.
7. Logic-specific — context without attempt_directory → silent skip.

Tests bypass :mod:`src.pipeline.stages.__init__` heavy package init by
loading the monitor module directly via importlib (same pattern as
:file:`test_training_monitor_v2.py`).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch


def _load_monitor():
    """Load the monitor module directly so we don't drag the whole
    pipeline.stages package init."""
    if "ryotenkai_monitor_test" in sys.modules:
        return sys.modules["ryotenkai_monitor_test"]
    repo_root = Path(__file__).resolve().parents[4]
    src_path = repo_root / "src" / "pipeline" / "stages" / "training_monitor.py"
    spec = importlib.util.spec_from_file_location(
        "ryotenkai_monitor_test", str(src_path),
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ryotenkai_monitor_test"] = mod
    spec.loader.exec_module(mod)
    return mod


_monitor_mod = _load_monitor()
TrainingMonitor = _monitor_mod.TrainingMonitor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_monitor() -> TrainingMonitor:
    """Construct a bare TrainingMonitor — bypasses parent __init__
    because reconciliation only touches ``self`` for the logger
    surface, which is module-level."""
    monitor = TrainingMonitor.__new__(TrainingMonitor)
    return monitor


def _write_marker(
    attempt_dir: Path, *, run_id: str | None = "abc123",
    flushed_count: int = 0,
) -> Path:
    """Drop a valid cancelled.marker in attempt_dir."""
    payload: dict[str, Any] = {
        "flushed_count": flushed_count,
        "ts_ms": 1700000000000,
        "reason": "flush_budget_exceeded",
    }
    if run_id is not None:
        payload["run_id"] = run_id
    marker = attempt_dir / "cancelled.marker"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    return marker


def _ctx_with_attempt(attempt_dir: Path) -> dict[str, Any]:
    """Build a minimal context dict the reconciliation method reads.

    The method imports
    :data:`src.pipeline.stages.constants.PipelineContextKeys` lazily
    inside the function body — that import is heavy because the
    constants module triggers a chunk of the pipeline package import
    chain. We pin the canonical key value here.
    """
    # PipelineContextKeys.ATTEMPT_DIRECTORY value — confirmed by the
    # test fixtures elsewhere in this directory using the same string.
    return {"attempt_directory": str(attempt_dir)}


def _patch_pipeline_context_keys() -> Any:
    """Stub the lazy ``from ryotenkai_control.pipeline.stages.constants import
    PipelineContextKeys`` so the test doesn't need the heavy package.

    The monitor reads ``PipelineContextKeys.ATTEMPT_DIRECTORY`` to look
    up the attempt directory in the context dict. We replace that
    attribute with the canonical string.
    """
    fake = SimpleNamespace(ATTEMPT_DIRECTORY="attempt_directory")
    fake_module = SimpleNamespace(PipelineContextKeys=fake)
    return patch.dict(
        sys.modules,
        {"ryotenkai_control.pipeline.stages.constants": fake_module},
    )


# ---------------------------------------------------------------------------
# 1. Positive — marker + RUNNING → set_terminated KILLED
# ---------------------------------------------------------------------------


class TestPositive:
    def test_marker_running_calls_set_terminated_killed(
        self, tmp_path: Path,
    ) -> None:
        _write_marker(tmp_path, run_id="run-001")
        ctx = _ctx_with_attempt(tmp_path)

        # Stub out MlflowClient at import time inside the method.
        fake_client = MagicMock()
        fake_client.get_run.return_value = SimpleNamespace(
            info=SimpleNamespace(status="RUNNING"),
        )
        # set_terminated is a void method on real MlflowClient.
        fake_client.set_terminated = MagicMock(return_value=None)

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                monitor = _make_monitor()
                # Must not raise; must call set_terminated exactly once
                # with the expected args.
                monitor._reconcile_terminal_marker_if_present(ctx)

        fake_client.get_run.assert_called_once_with("run-001")
        fake_client.set_terminated.assert_called_once_with(
            run_id="run-001", status="KILLED",
        )


# ---------------------------------------------------------------------------
# 2. Negative — no marker, no client touched
# ---------------------------------------------------------------------------


class TestNegative:
    def test_no_marker_does_not_touch_mlflow(
        self, tmp_path: Path,
    ) -> None:
        # Empty attempt dir → no marker.
        ctx = _ctx_with_attempt(tmp_path)
        fake_client = MagicMock()

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)

        # Nothing on the MLflow surface should have fired — the
        # function bails before importing MlflowClient.
        fake_client.get_run.assert_not_called()
        fake_client.set_terminated.assert_not_called()

    def test_missing_attempt_dir_in_context(self, tmp_path: Path) -> None:
        # Context has no ``attempt_dir`` key → silent skip.
        ctx: dict[str, Any] = {}
        fake_client = MagicMock()

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)

        fake_client.get_run.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Boundary — marker without run_id; corrupt marker
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_marker_without_run_id_skips(self, tmp_path: Path) -> None:
        _write_marker(tmp_path, run_id=None)
        ctx = _ctx_with_attempt(tmp_path)
        fake_client = MagicMock()

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)

        fake_client.get_run.assert_not_called()

    def test_corrupt_marker_json_is_swallowed(self, tmp_path: Path) -> None:
        marker = tmp_path / "cancelled.marker"
        marker.write_text("not valid json {{{", encoding="utf-8")
        ctx = _ctx_with_attempt(tmp_path)
        fake_client = MagicMock()

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                # Must not raise.
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)
        fake_client.get_run.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Invariants — never raises
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_does_not_raise_when_mlflow_get_run_explodes(
        self, tmp_path: Path,
    ) -> None:
        _write_marker(tmp_path, run_id="x")
        ctx = _ctx_with_attempt(tmp_path)

        fake_client = MagicMock()
        fake_client.get_run.side_effect = RuntimeError(
            "tracking server unreachable",
        )

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                # MUST NOT raise.
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)

        # set_terminated never called because get_run blew up.
        fake_client.set_terminated.assert_not_called()

    def test_does_not_raise_when_mlflow_set_terminated_explodes(
        self, tmp_path: Path,
    ) -> None:
        _write_marker(tmp_path, run_id="y")
        ctx = _ctx_with_attempt(tmp_path)

        fake_client = MagicMock()
        fake_client.get_run.return_value = SimpleNamespace(
            info=SimpleNamespace(status="RUNNING"),
        )
        fake_client.set_terminated.side_effect = RuntimeError("HTTP 500")

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)


# ---------------------------------------------------------------------------
# 5. Dependency errors — MLflow client constructor raising
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_constructor_failure_swallowed(self, tmp_path: Path) -> None:
        _write_marker(tmp_path, run_id="z")
        ctx = _ctx_with_attempt(tmp_path)

        def _broken_constructor():
            raise RuntimeError("env not set")

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=_broken_constructor,
                )},
            ):
                # Must not raise.
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)


# ---------------------------------------------------------------------------
# 6. Regressions — already-terminal upstream
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_already_killed_no_double_set(self, tmp_path: Path) -> None:
        _write_marker(tmp_path, run_id="run-already")
        ctx = _ctx_with_attempt(tmp_path)

        fake_client = MagicMock()
        fake_client.get_run.return_value = SimpleNamespace(
            info=SimpleNamespace(status="KILLED"),
        )

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)

        # Already terminal → no double set.
        fake_client.set_terminated.assert_not_called()

    def test_already_finished_no_set(self, tmp_path: Path) -> None:
        _write_marker(tmp_path, run_id="run-finished")
        ctx = _ctx_with_attempt(tmp_path)

        fake_client = MagicMock()
        fake_client.get_run.return_value = SimpleNamespace(
            info=SimpleNamespace(status="FINISHED"),
        )

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)

        fake_client.set_terminated.assert_not_called()

    def test_already_failed_no_set(self, tmp_path: Path) -> None:
        # Scenario: trainer crashed with FAILED status; reconciliation
        # must NOT override that — only RUNNING gets corrected.
        _write_marker(tmp_path, run_id="run-failed")
        ctx = _ctx_with_attempt(tmp_path)

        fake_client = MagicMock()
        fake_client.get_run.return_value = SimpleNamespace(
            info=SimpleNamespace(status="FAILED"),
        )

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)

        fake_client.set_terminated.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Logic-specific — empty / non-string attempt_dir
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_non_string_attempt_dir_skips(self, tmp_path: Path) -> None:
        # Defensive — caller could put a Path object in the context
        # rather than a string; the type check rejects it cleanly.
        ctx = {"attempt_directory": tmp_path}  # Path, not str
        fake_client = MagicMock()

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)

        fake_client.get_run.assert_not_called()

    def test_empty_string_attempt_dir_skips(self, tmp_path: Path) -> None:
        ctx = {"attempt_directory": ""}
        fake_client = MagicMock()

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)

        fake_client.get_run.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 11.A — completion.marker reconciliation
# ---------------------------------------------------------------------------


def _write_completion_marker(
    attempt_dir: Path,
    *,
    run_id: str | None = "abc123",
    flushed_count: int = 5,
    flush_timed_out: bool = False,
    reason: str = "natural_completion",
) -> Path:
    """Drop a valid completion.marker in attempt_dir."""
    payload: dict[str, Any] = {
        "flushed_count": flushed_count,
        "flush_timed_out": flush_timed_out,
        "ts_ms": 1700000000000,
        "reason": reason,
    }
    if run_id is not None:
        payload["run_id"] = run_id
    marker = attempt_dir / "completion.marker"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    return marker


class TestCompletionMarkerReconciliation:
    """Phase 11.A — Mac reads completion.marker after WS terminal,
    forces MLflow run RUNNING → FINISHED if Mac was asleep when HF
    MLflow callback ran end_run (and timed out).
    """

    def test_completion_marker_running_calls_set_terminated_finished(
        self, tmp_path: Path,
    ) -> None:
        _write_completion_marker(tmp_path, run_id="run-c-001")
        ctx = _ctx_with_attempt(tmp_path)

        fake_client = MagicMock()
        fake_client.get_run.return_value = SimpleNamespace(
            info=SimpleNamespace(status="RUNNING"),
        )
        fake_client.set_terminated = MagicMock(return_value=None)

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)

        # Forced to FINISHED, not KILLED.
        fake_client.set_terminated.assert_called_once_with(
            run_id="run-c-001", status="FINISHED",
        )

    def test_completion_marker_timed_out_still_finishes(
        self, tmp_path: Path,
    ) -> None:
        # Even if flush timed out (partial data), the run is still
        # naturally completed — reconcile to FINISHED. Operator sees
        # warning about partial data via log message; Mac UI shows
        # FINISHED status.
        _write_completion_marker(
            tmp_path,
            run_id="run-c-tt",
            flush_timed_out=True,
            reason="flush_budget_exceeded",
        )
        ctx = _ctx_with_attempt(tmp_path)

        fake_client = MagicMock()
        fake_client.get_run.return_value = SimpleNamespace(
            info=SimpleNamespace(status="RUNNING"),
        )

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)

        fake_client.set_terminated.assert_called_once_with(
            run_id="run-c-tt", status="FINISHED",
        )

    def test_completion_marker_already_finished_no_op(
        self, tmp_path: Path,
    ) -> None:
        # MLflow already FINISHED (HF closed it before Mac slept) →
        # reconciliation no-op.
        _write_completion_marker(tmp_path, run_id="run-c-ok")
        ctx = _ctx_with_attempt(tmp_path)

        fake_client = MagicMock()
        fake_client.get_run.return_value = SimpleNamespace(
            info=SimpleNamespace(status="FINISHED"),
        )

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)

        fake_client.set_terminated.assert_not_called()

    def test_both_markers_cancellation_wins(
        self, tmp_path: Path,
    ) -> None:
        # Race condition: cancellation hit at the very end of natural
        # completion → both markers exist. Cancellation wins (priority
        # list) → KILLED, not FINISHED.
        _write_marker(tmp_path, run_id="run-cancel-wins")
        _write_completion_marker(
            tmp_path, run_id="run-cancel-wins",  # same run_id
        )
        ctx = _ctx_with_attempt(tmp_path)

        fake_client = MagicMock()
        fake_client.get_run.return_value = SimpleNamespace(
            info=SimpleNamespace(status="RUNNING"),
        )

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)

        # KILLED (cancellation wins), exactly one set_terminated call.
        fake_client.set_terminated.assert_called_once_with(
            run_id="run-cancel-wins", status="KILLED",
        )

    def test_completion_marker_corrupt_json_swallowed(
        self, tmp_path: Path,
    ) -> None:
        # Corrupt completion.marker → log + skip without trying
        # cancelled.marker (defensive: disk read failure means attempt
        # dir likely in a bad shape).
        marker = tmp_path / "completion.marker"
        marker.write_text("not valid json {{{", encoding="utf-8")
        ctx = _ctx_with_attempt(tmp_path)
        fake_client = MagicMock()

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)

        fake_client.get_run.assert_not_called()

    def test_completion_marker_without_run_id_skips(
        self, tmp_path: Path,
    ) -> None:
        _write_completion_marker(tmp_path, run_id=None)
        ctx = _ctx_with_attempt(tmp_path)
        fake_client = MagicMock()

        with _patch_pipeline_context_keys():
            with patch.dict(
                sys.modules,
                {"mlflow.tracking": SimpleNamespace(
                    MlflowClient=lambda: fake_client,
                )},
            ):
                _make_monitor()._reconcile_terminal_marker_if_present(ctx)

        fake_client.get_run.assert_not_called()
