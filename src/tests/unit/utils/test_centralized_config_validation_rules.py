import pytest
from pydantic import ValidationError

from src.utils.config import (
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceLocal,
    ExperimentTrackingConfig,
    GlobalHyperparametersConfig,
    InferenceConfig,
    InferenceEnginesConfig,
    InferenceVLLMEngineConfig,
    LoraConfig,
    MLflowConfig,
    ModelConfig,
    PipelineConfig,
    StrategyPhaseConfig,
    TrainingOnlyConfig,
)


def _model_cfg() -> ModelConfig:
    return ModelConfig(
        name="test/model",
        torch_dtype="bfloat16",
        trust_remote_code=False,
    )


def _lora_cfg() -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        use_dora=False,
        use_rslora=False,
        init_lora_weights="gaussian",
    )


def _hp_cfg() -> GlobalHyperparametersConfig:
    return GlobalHyperparametersConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        warmup_ratio=0.0,
        epochs=1,
    )


def _dataset_cfg_local() -> DatasetConfig:
    return DatasetConfig(
        source_type="local",
        source_local=DatasetSourceLocal(
            local_paths=DatasetLocalPaths(
                train="data/train.jsonl",
                eval=None,
            )
        ),
    )


def _inference_cfg_disabled() -> InferenceConfig:
    return InferenceConfig(
        enabled=False,
        provider="single_node",
        engine="vllm",
        engines=InferenceEnginesConfig(
            vllm=InferenceVLLMEngineConfig(
                merge_image="test/merge:latest",
                serve_image="test/vllm:latest",
            )
        ),
    )


def _experiment_tracking_cfg() -> ExperimentTrackingConfig:
    return ExperimentTrackingConfig(
        mlflow=MLflowConfig(
            tracking_uri="http://127.0.0.1:5002",
            experiment_name="test-exp",
            log_artifacts=False,
            log_model=False,
        )
    )


def _pipeline_cfg(*, training: TrainingOnlyConfig, providers: dict, datasets: dict[str, DatasetConfig]) -> PipelineConfig:
    return PipelineConfig(
        model=_model_cfg(),
        training=training,
        providers=providers,
        datasets=datasets,
        inference=_inference_cfg_disabled(),
        experiment_tracking=_experiment_tracking_cfg(),
    )


def test_rule_1_strategies_chain_invalid_transition_fails_fast() -> None:
    with pytest.raises(ValidationError, match="Invalid transition"):
        TrainingOnlyConfig(
            type="qlora",
            qlora=_lora_cfg(),
            hyperparams=_hp_cfg(),
            strategies=[
                StrategyPhaseConfig(strategy_type="sft"),
                StrategyPhaseConfig(strategy_type="cpt"),
            ],
        )


def test_rule_2_dataset_reference_must_exist_in_registry() -> None:
    with pytest.raises(ValidationError, match=r"references dataset 'missing'"):
        _pipeline_cfg(
            training=TrainingOnlyConfig(
                provider=None,
                type="qlora",
                qlora=_lora_cfg(),
                hyperparams=_hp_cfg(),
                strategies=[StrategyPhaseConfig(strategy_type="sft", dataset="missing")],
            ),
            providers={},
            datasets={"default": _dataset_cfg_local()},
        )


def test_rule_3_training_provider_must_exist_in_providers_registry_if_set() -> None:
    with pytest.raises(ValidationError, match=r"training\.provider='runpod' not found"):
        _pipeline_cfg(
            training=TrainingOnlyConfig(
                provider="runpod",
                type="qlora",
                qlora=_lora_cfg(),
                hyperparams=_hp_cfg(),
                strategies=[StrategyPhaseConfig(strategy_type="sft", dataset="default")],
            ),
            providers={"single_node": {"type": "ssh"}},
            datasets={"default": _dataset_cfg_local()},
        )


def test_rule_4_adalora_requires_adalora_block() -> None:
    with pytest.raises(ValidationError, match=r"requires 'training\.adalora:'"):
        TrainingOnlyConfig(
            type="adalora",
            lora=_lora_cfg(),  # optional field, but adalora: block is missing → fails
            hyperparams=_hp_cfg(),
            strategies=[StrategyPhaseConfig(strategy_type="sft")],
        )


def test_rule_5_inference_enabled_requires_supported_provider_engine() -> None:
    with pytest.raises(ValidationError) as e:
        _ = InferenceConfig(
            enabled=True,
            provider="unknown_provider",
            engine="vllm",
            engines=InferenceEnginesConfig(
                vllm=InferenceVLLMEngineConfig(
                    merge_image="test/merge:latest",
                    serve_image="test/vllm:latest",
                )
            ),
        )
    assert any((err.get("loc") or ("",))[0] == "provider" for err in e.value.errors())

