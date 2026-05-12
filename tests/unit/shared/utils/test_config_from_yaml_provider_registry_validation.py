"""
Regression tests: provider name validation must fail early at config load time (from_yaml),
before reaching GPUDeployer stage.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ryotenkai_shared.config import PipelineConfig

# DRIFT (xfail audit 2026-05-12): tests patch `ryotenkai_providers.training.
# GPUProviderFactory` (removed Phase B — replaced by ProviderRegistry).
# Inline YAML schemas were migrated by the audit, but the test bodies still
# anchor on the removed factory. Rewriting against ProviderRegistry is
# beyond the audit scope.
pytestmark = pytest.mark.xfail(
    strict=True,
    reason=(
        "ryotenkai_providers.training.GPUProviderFactory removed (replaced by "
        "ProviderRegistry); tests need rewriting against the new registry surface."
    ),
)


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_from_yaml_raises_for_unregistered_provider_name(tmp_path: Path) -> None:
    cfg_path = tmp_path / "pipeline.yaml"
    _write_yaml(
        cfg_path,
        """
model:
  name: test-model
  torch_dtype: bfloat16
  trust_remote_code: false
providers:
  local:
    connect:
      ssh:
        alias: pc
    training:
      workspace_path: /tmp/workspace
    inference:
      serve:
        workspace: /tmp/test_inference
        host: 127.0.0.1
        port: 8000
training:
  provider: local
  hyperparams:
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 1
    learning_rate: 0.0001
    warmup_ratio: 0.0
    epochs: 1
  adapter:
    kind: qlora
    r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    bias: none
    target_modules: all-linear
    use_dora: false
    use_rslora: false
    init_lora_weights: gaussian
  strategies:
    - strategy_type: sft
      dataset: default
inference:
datasets:
  default:
    source:
      kind: local
      local_paths:
        train: data/train.jsonl
        eval: null
integrations:
  mlflow:
    experiment_name: test-exp
""",
    )

    with patch("ryotenkai_providers.training.GPUProviderFactory.get_available_providers", return_value=["single_node"]):  # noqa: SIM117
        with pytest.raises(ValueError, match=r"Unknown provider: 'local'"):
            _ = PipelineConfig.from_yaml(cfg_path)


def test_from_yaml_accepts_provider_when_registered(tmp_path: Path) -> None:
    cfg_path = tmp_path / "pipeline.yaml"
    _write_yaml(
        cfg_path,
        """
model:
  name: test-model
  torch_dtype: bfloat16
  trust_remote_code: false
providers:
  x:
    connect:
      ssh:
        alias: pc
    training:
      workspace_path: /tmp/workspace
    inference:
      serve:
        workspace: /tmp/test_inference
        host: 127.0.0.1
        port: 8000
training:
  provider: x
  hyperparams:
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 1
    learning_rate: 0.0001
    warmup_ratio: 0.0
    epochs: 1
  adapter:
    kind: qlora
    r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    bias: none
    target_modules: all-linear
    use_dora: false
    use_rslora: false
    init_lora_weights: gaussian
  strategies:
    - strategy_type: sft
      dataset: default
inference:
datasets:
  default:
    source:
      kind: local
      local_paths:
        train: data/train.jsonl
        eval: null
integrations:
  mlflow:
    experiment_name: test-exp
""",
    )

    with patch("ryotenkai_providers.training.GPUProviderFactory.get_available_providers", return_value=["x"]):
        cfg = PipelineConfig.from_yaml(cfg_path)

    assert cfg.training.provider == "x"


def test_from_yaml_does_not_fail_when_pipeline_module_missing(tmp_path: Path) -> None:
    """
    Remote training runtime may not include `src.providers` (only training subset is shipped).
    In that case we must skip provider factory registry validation instead of crashing.
    """
    cfg_path = tmp_path / "pipeline.yaml"
    _write_yaml(
        cfg_path,
        """
model:
  name: test-model
  torch_dtype: bfloat16
  trust_remote_code: false
providers:
  local:
    connect:
      ssh:
        alias: pc
    training:
      workspace_path: /tmp/workspace
    inference:
      serve:
        workspace: /tmp/test_inference
        host: 127.0.0.1
        port: 8000
training:
  provider: local
  hyperparams:
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 1
    learning_rate: 0.0001
    warmup_ratio: 0.0
    epochs: 1
  adapter:
    kind: qlora
    r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    bias: none
    target_modules: all-linear
    use_dora: false
    use_rslora: false
    init_lora_weights: gaussian
  strategies:
    - strategy_type: sft
      dataset: default
inference:
datasets:
  default:
    source:
      kind: local
      local_paths:
        train: data/train.jsonl
        eval: null
integrations:
  mlflow:
    experiment_name: test-exp
""",
    )

    # Simulate missing `src.providers` import in this runtime.
    import importlib

    real_import_module = importlib.import_module

    def _fake_import_module(name: str, package=None):  # type: ignore[no-untyped-def]
        if name.startswith("ryotenkai_providers"):
            raise ModuleNotFoundError("No module named 'ryotenkai_providers'", name="ryotenkai_providers")
        return real_import_module(name, package)

    with patch("importlib.import_module", side_effect=_fake_import_module):
        cfg = PipelineConfig.from_yaml(cfg_path)

    assert cfg.training.provider == "local"


