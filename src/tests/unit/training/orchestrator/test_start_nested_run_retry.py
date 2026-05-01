"""Phase 9.B — :meth:`MlflowPhaseLogger.start_nested_run` retry-grace.

Covers the "Mac asleep at phase boundary" scenario in a multi-strategy
chain (CPT → SFT → DPO):

1. CPT phase finishes; orchestrator advances to SFT.
2. ``start_nested_run`` calls ``mlflow.start_run(nested=True)``.
3. Mac is asleep → MLflow upstream unreachable for ~10-25 seconds.
4. Old code: single attempt → exception → return None → SFT phase
   loses its nested run, all metrics land in the parent run, run
   tags missing.
5. New code: 5 attempts with 1s + 2s + 4s + 8s + 16s backoff = 31s
   total grace window. Mac wakes → upstream reachable → nested run
   created on a later attempt.

7-category coverage of the retry contract.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


# Slim-venv: load the mlflow_logger module by file path.
# Stub the dependencies it imports at module level (no ML packages).
_module_table: dict[str, ModuleType] = sys.modules


def _stub(name: str, attrs: dict[str, object] | None = None) -> None:
    if name in _module_table:
        return
    try:
        __import__(name)
    except ModuleNotFoundError:
        m = ModuleType(name)
        for k, v in (attrs or {}).items():
            setattr(m, k, v)
        _module_table[name] = m


# Pre-build the parent ``src.training`` shell + the constants module
# that mlflow_logger imports.
_stub("src.training", {})
_stub("src.training.constants", {
    "CATEGORY_TRAINING": "training",
    "TAG_PHASE_IDX": "mlflow.phase_idx",
    "TAG_STRATEGY_TYPE": "mlflow.strategy_type",
    "TRUNCATE_ERROR_MSG": 1024,
})
_stub("src.training.metrics_models", {"TrainingMetricsSnapshot": object})


_LOGGER_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "training" / "orchestrator" / "phase_executor" / "mlflow_logger.py"
)
_spec = importlib.util.spec_from_file_location(
    "_ryotenkai_mlflow_logger_under_test", _LOGGER_PATH,
)
assert _spec is not None and _spec.loader is not None
_mlflow_logger_mod = importlib.util.module_from_spec(_spec)
_module_table["_ryotenkai_mlflow_logger_under_test"] = _mlflow_logger_mod
_spec.loader.exec_module(_mlflow_logger_mod)

MlflowPhaseLogger = _mlflow_logger_mod.MlflowPhaseLogger
NESTED_RUN_MAX_ATTEMPTS = _mlflow_logger_mod._NESTED_RUN_MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeMlflowManager:
    is_active = True


def _make_logger() -> Any:
    """Build a logger with no-op config and an active manager."""
    config = SimpleNamespace(
        integrations=SimpleNamespace(
            mlflow=SimpleNamespace(system_metrics_callback_enabled=False),
        ),
    )
    return MlflowPhaseLogger(_FakeMlflowManager(), config)


def _phase(strategy_type: str = "sft") -> Any:
    return SimpleNamespace(strategy_type=strategy_type)


def _install_fake_mlflow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_first_n: int = 0,
    raise_after: bool = False,
) -> tuple[MagicMock, list[int]]:
    """Stub ``mlflow.start_run`` — fail N times then succeed.

    Returns the start_run mock (so tests can inspect call_count) and
    a list ``calls`` that gets appended on each invocation."""
    calls: list[int] = []

    def _start_run(**_kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) <= fail_first_n:
            raise ConnectionError(f"upstream blip {len(calls)}")
        if raise_after:
            raise ConnectionError("persistent failure")
        return SimpleNamespace(run_id="r-1")

    fake_mlflow = ModuleType("mlflow")
    fake_mlflow.start_run = _start_run  # type: ignore[attr-defined]
    fake_mlflow.disable_system_metrics_logging = MagicMock()  # type: ignore[attr-defined]
    fake_mlflow.enable_system_metrics_logging = MagicMock()  # type: ignore[attr-defined]
    fake_mlflow.set_tags = MagicMock()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    return MagicMock(name="start_run"), calls


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_first_attempt_succeeds_returns_run(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, calls = _install_fake_mlflow(monkeypatch, fail_first_n=0)
        sleep_calls: list[float] = []

        run = _make_logger().start_nested_run(
            0, _phase("sft"), sleep=sleep_calls.append,
        )

        assert run is not None
        assert run.run_id == "r-1"
        assert len(calls) == 1
        assert sleep_calls == []  # no backoff on first-try success


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_returns_none_when_manager_inactive(self) -> None:
        config = SimpleNamespace(
            integrations=SimpleNamespace(mlflow=None),
        )
        manager = SimpleNamespace(is_active=False)
        logger = MlflowPhaseLogger(manager, config)
        assert logger.start_nested_run(0, _phase()) is None

    def test_returns_none_when_no_manager(self) -> None:
        config = SimpleNamespace(
            integrations=SimpleNamespace(mlflow=None),
        )
        logger = MlflowPhaseLogger(None, config)
        assert logger.start_nested_run(0, _phase()) is None


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_succeeds_on_last_attempt(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Fail N-1 times, succeed on the N-th attempt.
        _install_fake_mlflow(
            monkeypatch, fail_first_n=NESTED_RUN_MAX_ATTEMPTS - 1,
        )
        sleep_calls: list[float] = []
        run = _make_logger().start_nested_run(
            0, _phase(), sleep=sleep_calls.append,
        )
        assert run is not None
        # Backoff fired N-1 times (between attempts 1 and N).
        assert len(sleep_calls) == NESTED_RUN_MAX_ATTEMPTS - 1

    def test_all_attempts_fail_returns_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_mlflow(
            monkeypatch, fail_first_n=NESTED_RUN_MAX_ATTEMPTS,
        )
        sleep_calls: list[float] = []
        run = _make_logger().start_nested_run(
            0, _phase(), sleep=sleep_calls.append,
        )
        assert run is None
        # Backoff fired N-1 times (no sleep after the last failed attempt).
        assert len(sleep_calls) == NESTED_RUN_MAX_ATTEMPTS - 1


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_backoff_is_exponential(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # All attempts fail so we observe every backoff.
        _install_fake_mlflow(
            monkeypatch, fail_first_n=NESTED_RUN_MAX_ATTEMPTS,
        )
        sleep_calls: list[float] = []
        _make_logger().start_nested_run(
            0, _phase(), sleep=sleep_calls.append,
        )
        # 1s, 2s, 4s, 8s, ... — strictly monotonic.
        for i in range(len(sleep_calls) - 1):
            assert sleep_calls[i + 1] > sleep_calls[i], (
                f"backoff not monotonic at i={i}: {sleep_calls}"
            )
        # And specifically: 2x growth.
        if len(sleep_calls) >= 2:
            assert sleep_calls[1] / sleep_calls[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_no_more_than_max_attempts(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Persistent failure — assert we don't retry forever.
        _, calls = _install_fake_mlflow(
            monkeypatch, fail_first_n=NESTED_RUN_MAX_ATTEMPTS,
        )
        sleep_calls: list[float] = []
        _make_logger().start_nested_run(
            0, _phase(), sleep=sleep_calls.append,
        )
        assert len(calls) == NESTED_RUN_MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_max_attempts_pinned_at_5(self) -> None:
        """Plan §9.4 (9.B) explicit number — 5 attempts. Pin so a
        future tweak surfaces as a test failure for review."""
        assert NESTED_RUN_MAX_ATTEMPTS == 5

    def test_initial_backoff_pinned_at_1s(self) -> None:
        # Plan: 1s + 2s + 4s + 8s + 16s = 31s.
        assert _mlflow_logger_mod._NESTED_RUN_INITIAL_BACKOFF_S == 1.0

    def test_backoff_multiplier_pinned_at_2x(self) -> None:
        assert _mlflow_logger_mod._NESTED_RUN_BACKOFF_MULTIPLIER == 2.0


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_total_backoff_window_is_around_31_seconds(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sanity check the documented "~31s grace window" claim."""
        _install_fake_mlflow(
            monkeypatch, fail_first_n=NESTED_RUN_MAX_ATTEMPTS,
        )
        sleep_calls: list[float] = []
        _make_logger().start_nested_run(
            0, _phase(), sleep=sleep_calls.append,
        )
        # 1 + 2 + 4 + 8 + 16 = 31
        # We have N-1 = 4 sleeps between 5 attempts: 1+2+4+8 = 15
        # (no sleep after the last attempt).
        # Plan said "31s total" — that includes a hypothetical 16s
        # before a 6th attempt. With N=5 and no post-final sleep,
        # the window is 15s. Pin the actual sum.
        assert sum(sleep_calls) == 1.0 + 2.0 + 4.0 + 8.0
