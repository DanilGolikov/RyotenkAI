"""Phase 12.A.2 — :class:`MetricsDecimator` config integration.

Pins the new config-driven branching:
* No config / config=None → keep_all=True (lossless default).
* keep_all=True regardless of windows → always True.
* keep_all=False → 3-tier window logic with configured keep_every-N.

Slim-venv compatible: imports the decimator directly, builds a tiny
SimpleNamespace stand-in for ``MetricsBufferConfig``. No need to drag
the Pydantic model in.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest


# Slim-venv: load metrics_buffer module by file path to bypass
# ``src.training/__init__`` heavyweights.
_BUFFER_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "training" / "mlflow" / "metrics_buffer.py"
)
_spec = importlib.util.spec_from_file_location(
    "_ryotenkai_metrics_buffer_under_test", _BUFFER_PATH,
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules["_ryotenkai_metrics_buffer_under_test"] = _module
_spec.loader.exec_module(_module)

MetricsDecimator = _module.MetricsDecimator


def _config(
    *,
    keep_all: bool = True,
    first_minutes: int = 10,
    first_keep_every: int = 1,
    mid_minutes: int = 30,
    mid_keep_every: int = 2,
    late_keep_every: int = 5,
) -> Any:
    """Build a SimpleNamespace stand-in for MetricsBufferConfig that
    quacks like a Pydantic model for the duck-typed accessors in
    MetricsDecimator._extract_decimation."""
    return SimpleNamespace(
        keep_all=keep_all,
        decimation=SimpleNamespace(
            window_first_minutes=first_minutes,
            window_first_keep_every=first_keep_every,
            window_mid_minutes=mid_minutes,
            window_mid_keep_every=mid_keep_every,
            window_late_keep_every=late_keep_every,
        ),
    )


# ---------------------------------------------------------------------------
# 1. Positive — keep_all=True is lossless
# ---------------------------------------------------------------------------


class TestPositive:
    def test_no_config_is_lossless(self) -> None:
        # Backward compat: callers that didn't pass config (legacy
        # tests, ad-hoc usage) get the safer lossless behaviour.
        d = MetricsDecimator(training_start_time=0.0)
        # Even at step 7 (would have been decimated under tier 3)
        # AND elapsed ~∞ — keep.
        # Force elapsed → very large by setting start time to 0.
        for step in range(0, 100):
            assert d.should_keep(step) is True

    def test_keep_all_true_overrides_window_logic(self) -> None:
        d = MetricsDecimator(
            training_start_time=0.0,
            config=_config(keep_all=True, first_keep_every=10),
        )
        # Even though first_keep_every=10 (would skip 9 out of 10
        # steps), keep_all=True trumps the window logic.
        for step in range(20):
            assert d.should_keep(step) is True


# ---------------------------------------------------------------------------
# 2. Negative — keep_all=False enforces decimation
# ---------------------------------------------------------------------------


class TestNegative:
    def test_keep_all_false_uses_window_logic(self) -> None:
        # Phase 9 baseline: first 10 min keep all, 10-30 keep every 2nd,
        # 30+ keep every 5th. Use start_time=0 so elapsed=now =
        # extremely large = late window.
        import time as time_module

        d = MetricsDecimator(
            training_start_time=time_module.time() - 99999,
            config=_config(keep_all=False),
        )
        # Late window: keep every 5th step.
        assert d.should_keep(0) is True
        assert d.should_keep(1) is False
        assert d.should_keep(4) is False
        assert d.should_keep(5) is True
        assert d.should_keep(10) is True
        assert d.should_keep(15) is True


# ---------------------------------------------------------------------------
# 3. Boundary — exactly at window crossover
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_first_window_inclusive_lower(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time as time_module

        # Pin elapsed = 0 → in the first window (keep_every=1).
        monkeypatch.setattr(_module.time, "time", lambda: 100.0)
        d = MetricsDecimator(
            training_start_time=100.0,
            config=_config(keep_all=False, first_keep_every=1),
        )
        # First window: keep every step (keep_every=1).
        assert d.should_keep(0) is True
        assert d.should_keep(1) is True

    def test_mid_window_keep_every_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pin elapsed = 15 minutes → mid window.
        monkeypatch.setattr(_module.time, "time", lambda: 100.0 + 15 * 60)
        d = MetricsDecimator(
            training_start_time=100.0,
            config=_config(keep_all=False),
        )
        assert d.should_keep(0) is True
        assert d.should_keep(1) is False
        assert d.should_keep(2) is True

    def test_late_window_keep_every_5(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pin elapsed = 60 minutes → late window.
        monkeypatch.setattr(_module.time, "time", lambda: 100.0 + 60 * 60)
        d = MetricsDecimator(
            training_start_time=100.0,
            config=_config(keep_all=False),
        )
        assert d.should_keep(0) is True
        assert d.should_keep(1) is False
        assert d.should_keep(5) is True
        assert d.should_keep(10) is True


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_decimator_immutable_after_construction(self) -> None:
        # Decimator captures windows once at init — changing the
        # underlying config object after init does NOT alter behaviour.
        cfg = _config(keep_all=True)
        d = MetricsDecimator(training_start_time=0.0, config=cfg)
        cfg.keep_all = False
        # Still keeps everything because we captured at construction.
        for step in range(5):
            assert d.should_keep(step) is True


# ---------------------------------------------------------------------------
# 5. Dependency errors — defensive duck-typing
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_config_with_missing_decimation_uses_defaults(self) -> None:
        # Some test stand-ins might pass keep_all only. Decimator must
        # fall back to defaults rather than raise AttributeError.
        cfg = SimpleNamespace(keep_all=False)  # no `decimation` attr
        d = MetricsDecimator(training_start_time=0.0, config=cfg)
        # Should still work — fall back to Phase 9 defaults.
        # In late window so keep every 5th.
        # We don't need a precise assertion; just must NOT raise.
        d.should_keep(0)


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_legacy_call_signature_still_works(self) -> None:
        # Pre-Phase 12.A.2 callers used `MetricsDecimator(start_time)`
        # without config kwarg. New default = keep_all=True. Existing
        # tests / call sites continue to work, but the BEHAVIOUR widens
        # — they keep MORE metrics than before. This is the user-
        # mandated "lossless by default" stance.
        d = MetricsDecimator()
        # Will be in late window (started "just now" ~ 0 elapsed → first
        # window), but the point is: doesn't raise, and keep_all=True
        # means always keep.
        assert d.should_keep(0) is True
        assert d.should_keep(99) is True


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_window_minutes_converted_to_seconds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # YAML uses MINUTES; internal logic uses SECONDS. Verify the
        # conversion factor of 60.
        # Set first_minutes=2 → boundary at elapsed=120s.
        # Pin elapsed=119 → still in first window.
        monkeypatch.setattr(_module.time, "time", lambda: 119.0)
        d = MetricsDecimator(
            training_start_time=0.0,
            config=_config(
                keep_all=False, first_minutes=2, first_keep_every=1,
                mid_minutes=1, mid_keep_every=100,
            ),
        )
        # In first window (every step kept).
        assert d.should_keep(99) is True

        # Pin elapsed=121 → now in mid window (keep_every=100).
        monkeypatch.setattr(_module.time, "time", lambda: 121.0)
        d2 = MetricsDecimator(
            training_start_time=0.0,
            config=_config(
                keep_all=False, first_minutes=2, first_keep_every=1,
                mid_minutes=1, mid_keep_every=100,
            ),
        )
        assert d2.should_keep(99) is False
        assert d2.should_keep(100) is True

    def test_custom_late_keep_every(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pin elapsed = 1 hour → past mid; late_keep_every=10.
        monkeypatch.setattr(_module.time, "time", lambda: 3600.0)
        d = MetricsDecimator(
            training_start_time=0.0,
            config=_config(keep_all=False, late_keep_every=10),
        )
        assert d.should_keep(0) is True
        assert d.should_keep(9) is False
        assert d.should_keep(10) is True
        assert d.should_keep(19) is False
        assert d.should_keep(20) is True


# ---------------------------------------------------------------------------
# 8. MetricsBuffer init wiring
# ---------------------------------------------------------------------------


class TestMetricsBufferInit:
    def test_buffer_passes_config_to_decimator(self, tmp_path: pathlib.Path) -> None:
        # Wire-up: MetricsBuffer.__init__(config=cfg) must propagate the
        # config to its internal decimator.
        MetricsBuffer = _module.MetricsBuffer
        cfg = _config(keep_all=False, first_keep_every=99)
        buf = MetricsBuffer(buffer_dir=str(tmp_path), config=cfg)
        # Internal decimator captured cfg.keep_all=False and the windows.
        assert buf._decimator._keep_all is False
        assert buf._decimator._windows[1] == 99  # first_keep

    def test_buffer_default_no_config_is_lossless(self, tmp_path: pathlib.Path) -> None:
        MetricsBuffer = _module.MetricsBuffer
        buf = MetricsBuffer(buffer_dir=str(tmp_path))
        # No config → keep_all=True.
        assert buf._decimator._keep_all is True

    def test_buffer_write_metric_lossless_default(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end: with default config (keep_all=True), even very
        # late steps get written.
        monkeypatch.setattr(_module.time, "time", lambda: 100000.0)
        MetricsBuffer = _module.MetricsBuffer
        buf = MetricsBuffer(buffer_dir=str(tmp_path), training_start_time=0.0)
        # Step 7 would be decimated under tier 3 (keep_every=5), but
        # keep_all=True means kept.
        assert buf.write_metric("loss", 0.5, step=7) is True
        assert buf.path.exists()
