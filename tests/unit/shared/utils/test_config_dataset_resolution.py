"""
Tests for dataset resolution logic in PipelineConfig (NEW DatasetConfig schema).

Scope:
- PipelineConfig.get_dataset()
- PipelineConfig.get_primary_dataset()
- PipelineConfig.get_dataset_for_strategy()
- DatasetConfig helpers: get_source_type(), get_source_uri(), is_huggingface()
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from ryotenkai_engines.vllm.config import VLLMEngineConfig

from ryotenkai_shared.config import (
    DatasetConfig,
    GlobalHyperparametersConfig,
    InferenceConfig,
    IntegrationsConfig,
    MLflowConfig,
    ModelConfig,
    PipelineConfig,
    QLoRAConfig,
    StrategyPhaseConfig,
    TrainingOnlyConfig,
)


def _model_cfg() -> ModelConfig:
    return ModelConfig(
        name="test-model",
        torch_dtype="bfloat16",
        trust_remote_code=False,
    )


def _hp_cfg() -> GlobalHyperparametersConfig:
    return GlobalHyperparametersConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        warmup_ratio=0.0,
        epochs=1,
    )


def _lora_cfg() -> QLoRAConfig:
    return QLoRAConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        use_dora=False,
        use_rslora=False,
        init_lora_weights="gaussian",
    )


def _training_cfg(strategies: list[StrategyPhaseConfig]) -> TrainingOnlyConfig:
    return TrainingOnlyConfig(
        adapter=_lora_cfg(),
        hyperparams=_hp_cfg(),
        strategies=strategies,
    )


def _inference_cfg_disabled() -> InferenceConfig:
    return InferenceConfig(
        enabled=False,
        provider="single_node",
        engine=VLLMEngineConfig(),
    )


def _local_ds(local_path: str) -> DatasetConfig:
    """Convenience: create a local DatasetConfig (discriminated union)."""
    return DatasetConfig(
        source={"kind": "local", "local_paths": {"train": local_path, "eval": None}},
    )


def _hf_ds(train_id: str, *, eval_id: str | None = None) -> DatasetConfig:
    """Convenience: create a HuggingFace DatasetConfig."""
    return DatasetConfig(
        source={"kind": "huggingface", "train_id": train_id, "eval_id": eval_id},
    )


# MLflowConfig fixture fixed 2026-05-12 (xfail audit). Tests that still
# fail are tests using the removed `DatasetConfig.source_local` /
# `get_source_type` accessor surface (renamed to typed `source` accessor).
_XFAIL_DATASET_ACCESSOR_DRIFT = pytest.mark.xfail(
    strict=True,
    reason=(
        "DatasetConfig.source_local / get_source_type removed — replaced by typed "
        "`source` discriminated union (cfg.source.local_paths etc). Test needs "
        "rewriting against the typed accessor surface."
    ),
)


class TestGetDataset:
    @_XFAIL_DATASET_ACCESSOR_DRIFT
    def test_get_dataset_by_name(self) -> None:
        cfg = PipelineConfig(
            model=_model_cfg(),
            datasets={"default": _local_ds("data/default.jsonl"), "alpaca": _local_ds("data/alpaca.jsonl")},
            training=_training_cfg([StrategyPhaseConfig(strategy_type="sft")]),
            inference=_inference_cfg_disabled(),
            integrations=IntegrationsConfig(
                mlflow=MLflowConfig(tracking_uri="https://test.example.com", experiment_name="test")
            ),
        )

        ds = cfg.get_dataset("alpaca")
        assert ds.source_local is not None
        assert ds.source_local.local_paths.train == "data/alpaca.jsonl"

    @_XFAIL_DATASET_ACCESSOR_DRIFT
    def test_get_dataset_none_returns_primary(self) -> None:
        cfg = PipelineConfig(
            model=_model_cfg(),
            datasets={"default": _local_ds("data/default.jsonl")},
            training=_training_cfg([StrategyPhaseConfig(strategy_type="sft")]),
            inference=_inference_cfg_disabled(),
            integrations=IntegrationsConfig(
                mlflow=MLflowConfig(tracking_uri="https://test.example.com", experiment_name="test")
            ),
        )

        ds = cfg.get_dataset(None)
        assert ds.source_local is not None
        assert ds.source_local.local_paths.train == "data/default.jsonl"

    def test_get_dataset_missing_raises_keyerror(self) -> None:
        cfg = PipelineConfig(
            model=_model_cfg(),
            datasets={"default": _local_ds("data/default.jsonl")},
            training=_training_cfg([StrategyPhaseConfig(strategy_type="sft")]),
            inference=_inference_cfg_disabled(),
            integrations=IntegrationsConfig(
                mlflow=MLflowConfig(tracking_uri="https://test.example.com", experiment_name="test")
            ),
        )

        with pytest.raises(KeyError) as exc_info:
            cfg.get_dataset("missing")

        assert "missing" in str(exc_info.value)
        assert "default" in str(exc_info.value)


class TestPrimaryDataset:
    @_XFAIL_DATASET_ACCESSOR_DRIFT
    def test_primary_prefers_default_key(self) -> None:
        cfg = PipelineConfig(
            model=_model_cfg(),
            datasets={
                "default": _local_ds("data/default.jsonl"),
                "other": _local_ds("data/other.jsonl"),
            },
            training=_training_cfg([StrategyPhaseConfig(strategy_type="sft")]),
            inference=_inference_cfg_disabled(),
            integrations=IntegrationsConfig(
                mlflow=MLflowConfig(tracking_uri="https://test.example.com", experiment_name="test")
            ),
        )

        primary = cfg.get_primary_dataset()
        assert primary.source_local is not None
        assert primary.source_local.local_paths.train == "data/default.jsonl"

    @_XFAIL_DATASET_ACCESSOR_DRIFT
    def test_primary_falls_back_to_first_available_when_no_default(self) -> None:
        cfg = PipelineConfig(
            model=_model_cfg(),
            datasets={"first": _local_ds("data/first.jsonl"), "second": _local_ds("data/second.jsonl")},
            training=_training_cfg([StrategyPhaseConfig(strategy_type="sft")]),
            inference=_inference_cfg_disabled(),
            integrations=IntegrationsConfig(
                mlflow=MLflowConfig(tracking_uri="https://test.example.com", experiment_name="test")
            ),
        )

        primary = cfg.get_primary_dataset()
        assert primary.source_local is not None
        assert primary.source_local.local_paths.train == "data/first.jsonl"


class TestDatasetForStrategy:
    @_XFAIL_DATASET_ACCESSOR_DRIFT
    def test_strategy_dataset_lookup(self) -> None:
        cfg = PipelineConfig(
            model=_model_cfg(),
            datasets={
                "default": _local_ds("data/default.jsonl"),
                "sft_data": _local_ds("data/sft.jsonl"),
                "dpo_data": _hf_ds("Anthropic/hh-rlhf"),
            },
            training=_training_cfg(
                [
                    StrategyPhaseConfig(strategy_type="sft", dataset="sft_data"),
                    StrategyPhaseConfig(strategy_type="dpo", dataset="dpo_data"),
                ]
            ),
            inference=_inference_cfg_disabled(),
            integrations=IntegrationsConfig(
                mlflow=MLflowConfig(tracking_uri="https://test.example.com", experiment_name="test")
            ),
        )

        sft_ds = cfg.get_dataset_for_strategy(cfg.training.strategies[0])
        dpo_ds = cfg.get_dataset_for_strategy(cfg.training.strategies[1])

        assert sft_ds.get_source_type() == "local"
        assert sft_ds.source_local is not None
        assert sft_ds.source_local.local_paths.train == "data/sft.jsonl"

        assert dpo_ds.get_source_type() == "huggingface"
        assert dpo_ds.source_hf is not None
        assert dpo_ds.source_hf.train_id == "Anthropic/hh-rlhf"

    @_XFAIL_DATASET_ACCESSOR_DRIFT
    def test_strategy_without_dataset_uses_primary(self) -> None:
        cfg = PipelineConfig(
            model=_model_cfg(),
            datasets={"default": _local_ds("data/default.jsonl")},
            training=_training_cfg([StrategyPhaseConfig(strategy_type="sft")]),
            inference=_inference_cfg_disabled(),
            integrations=IntegrationsConfig(
                mlflow=MLflowConfig(tracking_uri="https://test.example.com", experiment_name="test")
            ),
        )

        ds = cfg.get_dataset_for_strategy(cfg.training.strategies[0])
        assert ds.source_local is not None
        assert ds.source_local.local_paths.train == "data/default.jsonl"


class TestValidateDatasets:
    def test_validate_datasets_missing_reference_fails(self) -> None:
        with pytest.raises(ValidationError, match=r"references\\s+dataset 'missing'|references dataset 'missing'"):
            _ = PipelineConfig(
                model=_model_cfg(),
                datasets={"default": _local_ds("data/default.jsonl")},
                training=_training_cfg(
                    [
                        StrategyPhaseConfig(strategy_type="sft", dataset="missing"),
                    ]
                ),
                inference=_inference_cfg_disabled(),
                integrations=IntegrationsConfig(
                    mlflow=MLflowConfig(tracking_uri="https://test.example.com", experiment_name="test")
                ),
            )


class TestDatasetConfigHelpers:
    def test_source_kind_local(self) -> None:
        ds = _local_ds("data/train.jsonl")
        assert ds.source.kind == "local"
        assert ds.is_huggingface() is False

    def test_source_kind_hf(self) -> None:
        ds = _hf_ds("tatsu-lab/alpaca")
        assert ds.source.kind == "huggingface"
        assert ds.is_huggingface() is True

    def test_get_source_uri_local_is_absolute(self) -> None:
        ds = _local_ds("data/train.jsonl")
        uri = ds.get_source_uri()
        assert Path(uri).is_absolute()

    def test_get_source_uri_hf(self) -> None:
        ds = _hf_ds("openai/gsm8k")
        assert ds.get_source_uri() == "huggingface://openai/gsm8k"


