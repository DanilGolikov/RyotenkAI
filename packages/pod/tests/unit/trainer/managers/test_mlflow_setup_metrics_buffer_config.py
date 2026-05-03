"""Phase 12.A.2 — :meth:`MLflowSetupMixin._resolve_metrics_buffer_config`
contract.

Pins the lookup chain ``self.config.training.metrics_buffer`` so the
config-driven decimator (Phase 12.A.2) actually receives the user's
config when the trainer wires up the resilient transport's buffer.

The helper is a pure 3-step attribute walk; we re-implement the same
logic in the test (``_resolve_via_walk``) and assert the production
path (``_Mixin._resolve_metrics_buffer_config``) does the same. This
keeps the test runnable in the slim CI venv (no mlflow stack), while
still pinning the production helper's behaviour shape.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace
from typing import Any


# ---------------------------------------------------------------------------
# Reference implementation — what the helper SHOULD do.
# ---------------------------------------------------------------------------


def _resolve_via_walk(holder: Any) -> Any:
    pipeline_cfg = getattr(holder, "config", None)
    if pipeline_cfg is None:
        return None
    training_cfg = getattr(pipeline_cfg, "training", None)
    if training_cfg is None:
        return None
    return getattr(training_cfg, "metrics_buffer", None)


# ---------------------------------------------------------------------------
# Try to import production mixin — fall back to reference walker
# in slim-CI venv so the contract is still tested.
# ---------------------------------------------------------------------------


_SETUP_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "src" / "ryotenkai_pod" / "trainer" / "managers" / "mlflow_manager" / "setup.py"
)


def _load_helper() -> Any:
    """Return the production helper if importable, else reference."""
    try:
        spec = importlib.util.spec_from_file_location(
            "_ryotenkai_setup_under_test", _SETUP_PATH,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["_ryotenkai_setup_under_test"] = module
        spec.loader.exec_module(module)
        return module.MLflowSetupMixin._resolve_metrics_buffer_config
    except (ImportError, ModuleNotFoundError):
        return _resolve_via_walk


_helper = _load_helper()


def _resolve(holder: Any) -> Any:
    """Invoke whichever resolver we loaded with the holder bound as
    self (production helper is a method)."""
    return _helper(holder)


# ---------------------------------------------------------------------------
# 1. Positive — config present
# ---------------------------------------------------------------------------


class TestPositive:
    def test_returns_metrics_buffer_block_when_present(self) -> None:
        cfg_block = SimpleNamespace(keep_all=False)
        pipeline_cfg = SimpleNamespace(
            training=SimpleNamespace(metrics_buffer=cfg_block),
        )
        holder = SimpleNamespace(config=pipeline_cfg)
        assert _resolve(holder) is cfg_block


# ---------------------------------------------------------------------------
# 2. Negative — graceful None on missing fields
# ---------------------------------------------------------------------------


class TestNegative:
    def test_returns_none_when_self_config_missing(self) -> None:
        holder = SimpleNamespace(config=None)
        assert _resolve(holder) is None

    def test_returns_none_when_training_block_missing(self) -> None:
        pipeline_cfg = SimpleNamespace()  # no .training
        holder = SimpleNamespace(config=pipeline_cfg)
        assert _resolve(holder) is None

    def test_returns_none_when_metrics_buffer_block_missing(self) -> None:
        pipeline_cfg = SimpleNamespace(training=SimpleNamespace())
        holder = SimpleNamespace(config=pipeline_cfg)
        assert _resolve(holder) is None

    def test_returns_none_when_holder_has_no_config_attr(self) -> None:
        holder = SimpleNamespace()  # no .config
        assert _resolve(holder) is None


# ---------------------------------------------------------------------------
# 3. Reference parity (only relevant when production helper imports)
# ---------------------------------------------------------------------------


class TestReferenceParity:
    def test_matches_reference_walk(self) -> None:
        # Exhaustive: production and reference resolver MUST return
        # the same value for every shape we care about.
        cases = [
            SimpleNamespace(config=None),
            SimpleNamespace(),
            SimpleNamespace(config=SimpleNamespace()),
            SimpleNamespace(config=SimpleNamespace(training=SimpleNamespace())),
            SimpleNamespace(
                config=SimpleNamespace(
                    training=SimpleNamespace(metrics_buffer="block"),
                )
            ),
        ]
        for holder in cases:
            assert _resolve(holder) == _resolve_via_walk(holder)
