from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.training.strategies.factory import StrategyFactory
from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig


def test_create_from_phase_does_not_access_legacy_flat_hyperparams_fields() -> None:
    """
    Regression (log-derived):
    StrategyFactory.create_from_phase must NOT access legacy fields like `phase.epochs`
    (they don't exist in the strict schema; hyperparams live under `phase.hyperparams.*`).

    This test would fail with:
        AttributeError: 'StrategyPhaseConfig' object has no attribute 'epochs'
    """
    phase = StrategyPhaseConfig(
        strategy_type="cpt",
        dataset="default",
        hyperparams=PhaseHyperparametersConfig(epochs=1, learning_rate=1e-5),
    )

    # Prove the schema is strict / legacy flat fields are absent.
    with pytest.raises(AttributeError):
        _ = phase.epochs  # type: ignore[attr-defined]

    sf = StrategyFactory()
    sf.create = MagicMock(return_value="sentinel_strategy")  # type: ignore[method-assign]

    cfg = MagicMock(name="pipeline_config")
    out = sf.create_from_phase(phase=phase, config=cfg)

    assert out == "sentinel_strategy"
    sf.create.assert_called_once_with("cpt", cfg)


