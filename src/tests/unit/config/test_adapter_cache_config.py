"""
Tests for AdapterCacheConfig, StrategyPhaseConfig.adapter_cache,
phase-level validators, and cross-validators related to adapter cache.

Coverage matrix
───────────────
AdapterCacheConfig           defaults / enabled / validation
StrategyPhaseConfig          adapter_cache field present & defaulted
validate_strategy_phase_config  positive / negative / boundary / invariants
validate_pipeline_adapter_cache_hf_config  positive / negative / boundary
validate_strategy_chain       cache-enabled phases excluded from dataset uniqueness
Regression                   SimpleNamespace mocks without adapter_cache field
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from src.config.training.adapter_cache import AdapterCacheConfig
from src.config.training.strategies.phase import StrategyPhaseConfig
from src.config.training.strategies.transitions import validate_strategy_chain
from src.config.validators.cross import validate_pipeline_adapter_cache_hf_config
from src.config.validators.training import validate_strategy_phase_config

pytestmark = pytest.mark.unit


def _assert_ok(result) -> None:
    assert result.is_success()
    assert result.unwrap() is None


def _assert_err(result, *, code: str | None = None) -> str:
    assert result.is_failure()
    err = result.unwrap_err()
    if code is not None:
        assert err.code == code
    return str(err)


# ─────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────


def _cache(*, enabled: bool = False, repo_id: str | None = None, private: bool = True) -> AdapterCacheConfig:
    return AdapterCacheConfig(enabled=enabled, repo_id=repo_id, private=private)


def _phase(
    strategy_type: str = "sft",
    dataset: str | None = "sft_data",
    *,
    cache: AdapterCacheConfig | None = None,
) -> StrategyPhaseConfig:
    kwargs: dict[str, Any] = {"strategy_type": strategy_type, "dataset": dataset}
    if cache is not None:
        kwargs["adapter_cache"] = cache
    return StrategyPhaseConfig(**kwargs)


@dataclass
class _AdapterCacheStub:
    enabled: bool = False
    repo_id: str | None = None


@dataclass
class _HFCfg:
    enabled: bool
    repo_id: str


@dataclass
class _ETCfg:
    huggingface: _HFCfg | None


@dataclass
class _Training:
    strategies: list[Any]


@dataclass
class _Pipeline:
    training: _Training
    experiment_tracking: _ETCfg


def _pipeline(
    strategies: list[Any],
    hf_enabled: bool = True,
    hf_repo: str = "org/final-model",
) -> _Pipeline:
    return _Pipeline(
        training=_Training(strategies=strategies),
        experiment_tracking=_ETCfg(huggingface=_HFCfg(enabled=hf_enabled, repo_id=hf_repo)),
    )


def _pipeline_no_hf(strategies: list[Any]) -> _Pipeline:
    return _Pipeline(
        training=_Training(strategies=strategies),
        experiment_tracking=_ETCfg(huggingface=None),
    )


# ─────────────────────────────────────────────
# AdapterCacheConfig — unit
# ─────────────────────────────────────────────


class TestAdapterCacheConfigDefaults:
    def test_positive_defaults_disabled(self) -> None:
        cfg = AdapterCacheConfig()
        assert cfg.enabled is False
        assert cfg.repo_id is None
        assert cfg.private is True

    def test_positive_explicit_enabled_with_repo_id(self) -> None:
        cfg = AdapterCacheConfig(enabled=True, repo_id="org/adapters", private=False)
        assert cfg.enabled is True
        assert cfg.repo_id == "org/adapters"
        assert cfg.private is False

    def test_positive_enabled_false_repo_id_none_allowed(self) -> None:
        cfg = AdapterCacheConfig(enabled=False, repo_id=None)
        assert cfg.repo_id is None

    def test_invariant_enabled_defaults_to_false(self) -> None:
        """Invariant: adapter_cache is always disabled unless explicitly enabled."""
        cfg = AdapterCacheConfig()
        assert cfg.enabled is False

    def test_invariant_private_defaults_to_true(self) -> None:
        cfg = AdapterCacheConfig()
        assert cfg.private is True


# ─────────────────────────────────────────────
# StrategyPhaseConfig — adapter_cache field
# ─────────────────────────────────────────────


class TestStrategyPhaseConfigAdapterCache:
    def test_positive_adapter_cache_present_with_defaults(self) -> None:
        phase = StrategyPhaseConfig(strategy_type="sft", dataset="d")
        assert hasattr(phase, "adapter_cache")
        assert isinstance(phase.adapter_cache, AdapterCacheConfig)
        assert phase.adapter_cache.enabled is False

    def test_positive_adapter_cache_enabled_with_repo_id(self) -> None:
        phase = _phase(cache=_cache(enabled=True, repo_id="org/sft-cache"))
        assert phase.adapter_cache.enabled is True
        assert phase.adapter_cache.repo_id == "org/sft-cache"

    def test_invariant_missing_adapter_cache_in_input_uses_default(self) -> None:
        """StrategyPhaseConfig built without adapter_cache key should still have defaults."""
        phase = StrategyPhaseConfig(strategy_type="dpo", dataset="pref_data")
        assert phase.adapter_cache.enabled is False
        assert phase.adapter_cache.repo_id is None


# ─────────────────────────────────────────────
# validate_strategy_phase_config
# ─────────────────────────────────────────────


class TestValidateStrategyPhaseConfigAdapterCache:
    # ── Positive ──────────────────────────────

    def test_positive_cache_disabled_no_extra_requirements(self) -> None:
        phase = _phase(cache=_cache(enabled=False))
        validate_strategy_phase_config(phase)  # must not raise

    def test_positive_cache_enabled_with_repo_and_dataset(self) -> None:
        phase = _phase(dataset="sft_data", cache=_cache(enabled=True, repo_id="org/cache"))
        validate_strategy_phase_config(phase)  # must not raise

    def test_positive_cache_enabled_private_false(self) -> None:
        phase = _phase(dataset="sft_data", cache=_cache(enabled=True, repo_id="org/cache", private=False))
        validate_strategy_phase_config(phase)

    # ── Negative ──────────────────────────────

    def test_negative_cache_enabled_without_repo_id(self) -> None:
        # Pydantic model rejects construction
        with pytest.raises(ValidationError, match="repo_id is required"):
            StrategyPhaseConfig(
                strategy_type="sft",
                dataset="sft_data",
                adapter_cache=AdapterCacheConfig(enabled=True, repo_id=None),
            )
        # Standalone validator also rejects
        ns = SimpleNamespace(strategy_type="sft", dataset="d", adapter_cache=_AdapterCacheStub(enabled=True, repo_id=None))
        with pytest.raises(ValueError, match="repo_id is required"):
            validate_strategy_phase_config(ns)  # type: ignore[arg-type]

    def test_negative_cache_enabled_without_dataset(self) -> None:
        ns = SimpleNamespace(
            strategy_type="sft",
            dataset=None,
            adapter_cache=_AdapterCacheStub(enabled=True, repo_id="org/cache"),
        )
        with pytest.raises(ValueError, match="requires dataset"):
            validate_strategy_phase_config(ns)  # type: ignore[arg-type]

    def test_negative_cache_enabled_empty_dataset(self) -> None:
        """dataset='' is falsy → same as None."""
        ns = SimpleNamespace(
            strategy_type="sft",
            dataset="",
            adapter_cache=_AdapterCacheStub(enabled=True, repo_id="org/cache"),
        )
        with pytest.raises(ValueError, match="requires dataset"):
            validate_strategy_phase_config(ns)  # type: ignore[arg-type]

    # ── Boundary ──────────────────────────────

    def test_boundary_cache_enabled_default_false_no_check(self) -> None:
        """When enabled=False, repo_id and dataset are not required."""
        ns = SimpleNamespace(
            strategy_type="dpo",
            dataset=None,
            adapter_cache=_AdapterCacheStub(enabled=False, repo_id=None),
        )
        validate_strategy_phase_config(ns)  # type: ignore[arg-type]

    def test_boundary_no_adapter_cache_attr_on_mock_does_not_crash(self) -> None:
        """Regression: mocks without adapter_cache attribute should not crash."""
        ns = SimpleNamespace(strategy_type="sft", hyperparams=SimpleNamespace())
        validate_strategy_phase_config(ns)  # type: ignore[arg-type]

    # ── Regression: wiring ─────────────────────

    def test_regression_pydantic_wiring_cache_enabled_no_repo_raises_validation_error(self) -> None:
        """StrategyPhaseConfig model_validator must reject enabled=True with no repo_id."""
        with pytest.raises(ValidationError, match="repo_id is required"):
            StrategyPhaseConfig(
                strategy_type="sft",
                dataset="sft_data",
                adapter_cache=AdapterCacheConfig(enabled=True, repo_id=None),
            )

    def test_regression_pydantic_wiring_cache_enabled_no_dataset_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="requires dataset"):
            StrategyPhaseConfig(
                strategy_type="sft",
                dataset=None,
                adapter_cache=AdapterCacheConfig(enabled=True, repo_id="org/cache"),
            )


# ─────────────────────────────────────────────
# validate_pipeline_adapter_cache_hf_config
# ─────────────────────────────────────────────


@dataclass
class _StrategyWithCache:
    strategy_type: str
    adapter_cache: _AdapterCacheStub = field(default_factory=_AdapterCacheStub)


class TestValidatePipelineAdapterCacheHFConfig:
    # ── Positive ──────────────────────────────

    def test_positive_no_phases_with_cache_enabled(self) -> None:
        cfg = _pipeline(strategies=[_StrategyWithCache("sft")])
        _assert_ok(validate_pipeline_adapter_cache_hf_config(cfg))  # type: ignore[arg-type]

    def test_positive_cache_enabled_hf_configured_different_repo(self) -> None:
        s = _StrategyWithCache("sft", _AdapterCacheStub(enabled=True, repo_id="org/sft-cache"))
        cfg = _pipeline(strategies=[s], hf_enabled=True, hf_repo="org/final-model")
        _assert_ok(validate_pipeline_adapter_cache_hf_config(cfg))  # type: ignore[arg-type]

    def test_positive_multiple_phases_different_repo_ids(self) -> None:
        phases = [
            _StrategyWithCache("sft", _AdapterCacheStub(enabled=True, repo_id="org/sft-cache")),
            _StrategyWithCache("dpo", _AdapterCacheStub(enabled=True, repo_id="org/dpo-cache")),
        ]
        cfg = _pipeline(strategies=phases, hf_enabled=True, hf_repo="org/final")
        _assert_ok(validate_pipeline_adapter_cache_hf_config(cfg))  # type: ignore[arg-type]

    # ── Negative ──────────────────────────────

    def test_negative_cache_enabled_hf_not_configured(self) -> None:
        s = _StrategyWithCache("sft", _AdapterCacheStub(enabled=True, repo_id="org/sft-cache"))
        cfg = _pipeline_no_hf(strategies=[s])
        err = _assert_err(validate_pipeline_adapter_cache_hf_config(cfg), code="CONFIG_ADAPTER_CACHE_HF_REQUIRED")  # type: ignore[arg-type]
        assert "experiment_tracking.huggingface" in err

    def test_negative_cache_enabled_hf_not_enabled(self) -> None:
        s = _StrategyWithCache("sft", _AdapterCacheStub(enabled=True, repo_id="org/sft-cache"))
        cfg = _pipeline(strategies=[s], hf_enabled=False, hf_repo="org/final")
        err = _assert_err(validate_pipeline_adapter_cache_hf_config(cfg), code="CONFIG_ADAPTER_CACHE_HF_REQUIRED")  # type: ignore[arg-type]
        assert "huggingface" in err.lower()

    def test_negative_adapter_cache_repo_equals_final_model_repo(self) -> None:
        s = _StrategyWithCache("sft", _AdapterCacheStub(enabled=True, repo_id="org/final-model"))
        cfg = _pipeline(strategies=[s], hf_enabled=True, hf_repo="org/final-model")
        err = _assert_err(validate_pipeline_adapter_cache_hf_config(cfg), code="CONFIG_ADAPTER_CACHE_REPO_CONFLICT")  # type: ignore[arg-type]
        assert "must differ" in err

    def test_negative_second_phase_repo_equals_final_model_repo(self) -> None:
        phases = [
            _StrategyWithCache("sft", _AdapterCacheStub(enabled=True, repo_id="org/sft-ok")),
            _StrategyWithCache("dpo", _AdapterCacheStub(enabled=True, repo_id="org/final")),
        ]
        cfg = _pipeline(strategies=phases, hf_enabled=True, hf_repo="org/final")
        err = _assert_err(validate_pipeline_adapter_cache_hf_config(cfg), code="CONFIG_ADAPTER_CACHE_REPO_CONFLICT")  # type: ignore[arg-type]
        assert "must differ" in err

    # ── Boundary ──────────────────────────────

    def test_boundary_empty_strategies_list(self) -> None:
        cfg = _pipeline(strategies=[], hf_enabled=True, hf_repo="org/final")
        _assert_ok(validate_pipeline_adapter_cache_hf_config(cfg))  # type: ignore[arg-type]

    def test_boundary_all_phases_disabled_hf_not_configured(self) -> None:
        """If no phase has cache enabled, HF config is not required."""
        phases = [_StrategyWithCache("sft"), _StrategyWithCache("dpo")]
        cfg = _pipeline_no_hf(strategies=phases)
        _assert_ok(validate_pipeline_adapter_cache_hf_config(cfg))  # type: ignore[arg-type]

    def test_boundary_strategy_without_adapter_cache_attr_skipped(self) -> None:
        """Strategies without adapter_cache attribute should be treated as disabled."""
        plain_strategy = SimpleNamespace(strategy_type="sft", dataset="d")
        cfg = _pipeline_no_hf(strategies=[plain_strategy])
        _assert_ok(validate_pipeline_adapter_cache_hf_config(cfg))  # type: ignore[arg-type]


# ─────────────────────────────────────────────
# validate_strategy_chain — adapter_cache interaction
# ─────────────────────────────────────────────


@dataclass
class _FullPhase:
    """Phase stub with all fields needed by validate_strategy_chain."""
    strategy_type: str
    dataset: str | None = None
    adapter_cache: _AdapterCacheStub = field(default_factory=_AdapterCacheStub)


class TestValidateStrategyChainAdapterCache:
    # ── Positive ──────────────────────────────

    def test_positive_cache_enabled_phases_excluded_from_dataset_uniqueness(self) -> None:
        """Two phases using the same dataset — one has cache enabled → no uniqueness error."""
        phases = [
            _FullPhase("sft", dataset="shared_data", adapter_cache=_AdapterCacheStub(enabled=True)),
            _FullPhase("dpo", dataset="shared_data"),
        ]
        _assert_ok(validate_strategy_chain(phases))  # type: ignore[arg-type]

    def test_positive_all_cache_enabled_phases_skip_uniqueness_check(self) -> None:
        phases = [
            _FullPhase("sft", dataset="d1", adapter_cache=_AdapterCacheStub(enabled=True)),
            _FullPhase("dpo", dataset="d1", adapter_cache=_AdapterCacheStub(enabled=True)),
        ]
        _assert_ok(validate_strategy_chain(phases))  # type: ignore[arg-type]

    def test_positive_normal_phases_with_unique_datasets(self) -> None:
        phases = [
            _FullPhase("sft", dataset="sft_data"),
            _FullPhase("dpo", dataset="dpo_data"),
        ]
        _assert_ok(validate_strategy_chain(phases))  # type: ignore[arg-type]

    # ── Negative ──────────────────────────────

    def test_negative_two_non_cache_phases_same_dataset_still_fails(self) -> None:
        """Cache-disabled phases must still pass the uniqueness check."""
        phases = [
            _FullPhase("sft", dataset="shared"),
            _FullPhase("dpo", dataset="shared"),
        ]
        err = _assert_err(validate_strategy_chain(phases), code="STRATEGY_CHAIN_DUPLICATE_DATASET")  # type: ignore[arg-type]
        assert "shared" in err

    def test_negative_mixed_only_non_cache_duplicate_fails(self) -> None:
        """Cache phase uses same dataset → excluded. But two non-cache use same → still fails."""
        phases = [
            _FullPhase("sft", dataset="sft_data", adapter_cache=_AdapterCacheStub(enabled=True)),
            _FullPhase("dpo", dataset="dup"),
            _FullPhase("orpo", dataset="dup"),
        ]
        # sft → dpo → orpo is not a valid chain (orpo not valid from dpo)
        # Use a single-phase chain to test uniqueness in isolation
        phases2 = [
            _FullPhase("sft", dataset="dup"),
            _FullPhase("dpo", dataset="dup"),
        ]
        err = _assert_err(validate_strategy_chain(phases2), code="STRATEGY_CHAIN_DUPLICATE_DATASET")  # type: ignore[arg-type]
        assert "dup" in err

    # ── Boundary ──────────────────────────────

    def test_boundary_single_cache_enabled_phase_is_valid(self) -> None:
        phases = [_FullPhase("sft", dataset="d", adapter_cache=_AdapterCacheStub(enabled=True))]
        _assert_ok(validate_strategy_chain(phases))  # type: ignore[arg-type]

    def test_boundary_phase_without_adapter_cache_attr_uses_default_false(self) -> None:
        """Plain dataclass without adapter_cache field is treated as cache-disabled."""
        @dataclass
        class MinPhase:
            strategy_type: str
            dataset: str | None = None

        phases = [MinPhase("sft", "shared"), MinPhase("dpo", "shared")]
        _assert_err(validate_strategy_chain(phases), code="STRATEGY_CHAIN_DUPLICATE_DATASET")  # type: ignore[arg-type]

    # ── Combinatorial ─────────────────────────

    def test_combinatorial_three_phases_mixed_cache_uniqueness(self) -> None:
        """
        sft: cache=True, dataset=shared   → excluded from uniqueness check
        dpo: cache=False, dataset=shared  → included; no previous non-cache phase used 'shared' → ok
        Only fails if TWO non-cache phases use same dataset.
        """
        phases = [
            _FullPhase("sft", dataset="shared", adapter_cache=_AdapterCacheStub(enabled=True)),
            _FullPhase("dpo", dataset="shared"),
        ]
        _assert_ok(validate_strategy_chain(phases))  # type: ignore[arg-type]
