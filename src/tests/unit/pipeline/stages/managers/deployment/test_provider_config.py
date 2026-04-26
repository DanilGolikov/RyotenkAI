"""Unit tests for src.pipeline.stages.managers.deployment.provider_config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.constants import PROVIDER_SINGLE_NODE
from src.pipeline.stages.managers.deployment.provider_config import (
    get_active_provider_name,
    get_cloud_training_cfg,
    get_provider_training_cfg,
    get_single_node_training_cfg,
    is_single_node_provider,
)
from src.utils.config import (
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceLocal,
    GlobalHyperparametersConfig,
    InferenceConfig,
    InferenceEnginesConfig,
    InferenceVLLMEngineConfig,
    ModelConfig,
    PipelineConfig,
    QLoRAConfig,
    TrainingOnlyConfig,
)

pytestmark = pytest.mark.unit

DATASET_CHAT_FIXTURE = "src/tests/fixtures/datasets/test_chat.jsonl"

SINGLE_NODE_PROVIDER_CFG: dict[str, Any] = {
    "connect": {"ssh": {"alias": "pc"}},
    "training": {"workspace_path": "/tmp/workspace", "docker_image": "test/training-runtime:latest"},
}


@dataclass(frozen=True)
class DummySecrets:
    hf_token: str = "hf_test_token"


def _make_config(provider: str = "single_node") -> PipelineConfig:
    return PipelineConfig(
        model=ModelConfig(name="gpt2", torch_dtype="bfloat16", trust_remote_code=False),
        providers={"single_node": SINGLE_NODE_PROVIDER_CFG},
        training=TrainingOnlyConfig(
            provider=provider,
            type="qlora",
            qlora=QLoRAConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules="all-linear",
                use_dora=False,
                use_rslora=False,
                init_lora_weights="gaussian",
            ),
            hyperparams=GlobalHyperparametersConfig(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=2e-4,
                warmup_ratio=0.0,
                epochs=1,
            ),
        ),
        datasets={
            "default": DatasetConfig(
                source_type="local",
                source_local=DatasetSourceLocal(local_paths=DatasetLocalPaths(train=DATASET_CHAT_FIXTURE, eval=None)),
            )
        },
        inference=InferenceConfig(
            enabled=False,
            provider="single_node",
            engine="vllm",
            engines=InferenceEnginesConfig(
                vllm=InferenceVLLMEngineConfig(
                    merge_image="test/merge:latest",
                    serve_image="test/vllm:latest",
                )
            ),
        ),
    )


def test_get_active_provider_name_returns_real_value():
    cfg = _make_config()
    assert get_active_provider_name(cfg) == PROVIDER_SINGLE_NODE


def test_get_active_provider_name_falls_back_when_accessor_raises():
    cfg = MagicMock()
    cfg.get_active_provider_name.side_effect = RuntimeError("boom")
    cfg.training.provider = "my_custom_provider"
    assert get_active_provider_name(cfg) == "my_custom_provider"


def test_get_active_provider_name_non_string_provider_returns_single_node():
    cfg = MagicMock()
    cfg.get_active_provider_name.side_effect = RuntimeError("boom")
    cfg.training.provider = 12345  # not a string
    assert get_active_provider_name(cfg) == PROVIDER_SINGLE_NODE


def test_get_active_provider_name_no_training_attr_returns_single_node():
    cfg = MagicMock(spec=["get_active_provider_name"])  # no training attr
    cfg.get_active_provider_name.side_effect = RuntimeError("boom")
    assert get_active_provider_name(cfg) == PROVIDER_SINGLE_NODE


def test_is_single_node_provider_true():
    cfg = _make_config()
    assert is_single_node_provider(cfg) is True


def test_is_single_node_provider_false():
    cfg = MagicMock()
    cfg.get_active_provider_name.return_value = "runpod"
    assert is_single_node_provider(cfg) is False


def test_get_provider_training_cfg_known_provider_returns_dict():
    cfg = _make_config()
    training = get_provider_training_cfg(cfg, "single_node")
    assert isinstance(training, dict)
    assert training["docker_image"] == "test/training-runtime:latest"


def test_get_provider_training_cfg_missing_provider_falls_back_to_default():
    cfg = MagicMock()

    def side_effect(*args, **kwargs):
        if args and args[0] == "missing":
            raise KeyError("missing")
        return {"training": {"workspace_path": "/default"}}

    cfg.get_provider_config.side_effect = side_effect
    training = get_provider_training_cfg(cfg, "missing")
    assert training == {"workspace_path": "/default"}


def test_get_provider_training_cfg_returns_empty_when_all_accessors_fail():
    cfg = MagicMock()
    cfg.get_provider_config.side_effect = ValueError("nope")
    assert get_provider_training_cfg(cfg, "anything") == {}


def test_get_single_node_training_cfg_returns_single_node_training():
    cfg = _make_config()
    training = get_single_node_training_cfg(cfg)
    assert training["docker_image"] == "test/training-runtime:latest"


def test_get_cloud_training_cfg_returns_active_provider_training():
    cfg = MagicMock()
    cfg.get_active_provider_name.return_value = "runpod"
    cfg.get_provider_config.return_value = {"training": {"image_name": "runpod/img"}}
    training = get_cloud_training_cfg(cfg)
    assert training == {"image_name": "runpod/img"}
