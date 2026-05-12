"""Tests for :func:`make_pipeline_config`."""

from __future__ import annotations

from pathlib import Path

from ryotenkai_shared.config import PipelineConfig

from tests._factories.pipeline_config import make_pipeline_config


def test_make_pipeline_config_returns_real_pipeline_config() -> None:
    """Factory returns a real :class:`PipelineConfig`, not a mock."""
    cfg = make_pipeline_config()
    assert isinstance(cfg, PipelineConfig)


def test_make_pipeline_config_stamps_source_path() -> None:
    """``_source_path`` is set so :class:`PipelineBootstrap` won't reject it."""
    cfg = make_pipeline_config()
    assert cfg._source_path is not None
    assert isinstance(cfg._source_path, Path)


def test_make_pipeline_config_source_path_override() -> None:
    """Caller-supplied ``source_path`` reaches the private attribute."""
    explicit = Path("/tmp/explicit-config.yaml")
    cfg = make_pipeline_config(source_path=explicit)
    assert cfg._source_path == explicit


def test_make_pipeline_config_defaults_validate() -> None:
    """The default config passes every cross-block validator.

    The :class:`PipelineConfig` ``model_validator(mode="after")`` runs:

      * ``validate_pipeline_active_provider_is_registered``
      * ``validate_pipeline_strategy_dataset_references``
      * ``validate_pipeline_inference_provider_config``
      * ``validate_pipeline_evaluation_requires_inference``
      * ``validate_pipeline_adapter_cache_hf_config``

    Construction here is the validation — if any of these fail
    :class:`pydantic.ValidationError` is raised before the assertion.
    """
    cfg = make_pipeline_config()
    # Single SFT strategy by default — short-circuits the cross-validator
    # that requires referenced datasets to exist.
    assert len(cfg.training.strategies) == 1
    assert cfg.training.strategies[0].strategy_type == "sft"
    assert "default" in cfg.datasets


def test_make_pipeline_config_accepts_overrides() -> None:
    """Caller can override individual blocks without rebuilding the rest."""
    cfg = make_pipeline_config(providers={"my_provider": {"type": "custom"}})
    # Override took effect.
    assert cfg.providers == {"my_provider": {"type": "custom"}}
    # Unchanged blocks still carry the default shape.
    assert cfg.training.adapter.kind == "qlora"
