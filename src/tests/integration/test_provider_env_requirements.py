"""
Integration tests for provider-specific env requirements.

Regression:
- RUNPOD_API_KEY must NOT be required when active provider is not runpod.
- RUNPOD_API_KEY must be required when active provider is runpod.

NOTE: file name intentionally avoids '*secrets*' / '*api_key*' substrings because
the repo .gitignore contains broad patterns that would ignore such filenames.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.orchestrator import PipelineOrchestrator


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _patch_stage_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid stage construction + other heavy deps during __init__
    monkeypatch.setattr("src.pipeline.orchestrator.DatasetValidator", lambda *a, **k: None)
    monkeypatch.setattr("src.pipeline.orchestrator.GPUDeployer", lambda *a, **k: None)
    monkeypatch.setattr("src.pipeline.orchestrator.TrainingMonitor", lambda *a, **k: None)
    monkeypatch.setattr("src.pipeline.orchestrator.ModelRetriever", lambda *a, **k: None)
    monkeypatch.setattr("src.pipeline.orchestrator.ModelEvaluator", lambda *a, **k: None)
    monkeypatch.setattr("src.pipeline.orchestrator.InferenceDeployer", lambda *a, **k: None)


def test_runpod_key_optional_for_ssh_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Positive: config with active provider type=ssh must not require RUNPOD_API_KEY.
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
  single_node:
    connect:
      ssh:
        alias: pc
    training:
      workspace_path: /tmp/workspace
      docker_image: test/training-runtime:latest
    inference:
      serve:
        workspace: /tmp/test_inference
        host: 127.0.0.1
        port: 8000

training:
  provider: single_node
  type: qlora
  hyperparams:
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 1
    learning_rate: 2.0e-4
    warmup_ratio: 0.0
    epochs: 1
  qlora:
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
  enabled: false
  provider: single_node
  engine: vllm
  engines:
    vllm:
      merge_image: test/merge:latest
      serve_image: test/vllm:latest

datasets:
  default:
    source_type: local
    source_local:
      local_paths:
        train: data/does_not_matter.jsonl
        eval: null

experiment_tracking:
  mlflow:
    tracking_uri: http://127.0.0.1:5002
    experiment_name: test-exp
""",
    )

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    _patch_stage_construction(monkeypatch)
    monkeypatch.setattr(
        "src.pipeline.orchestrator.load_secrets",
        lambda: type("S", (), {"hf_token": "hf_test_token", "runpod_api_key": None})(),
    )

    # Should not raise on env requirements
    _ = PipelineOrchestrator(cfg_path)


def test_runpod_key_required_for_runpod_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Negative: config with active provider type=runpod must require RUNPOD_API_KEY.
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
  runpod:
    connect:
      ssh:
        key_path: /tmp/id_ed25519
    cleanup: {}
    training:
      gpu_type: NVIDIA A40
      cloud_type: ALL
      image_name: test/training-runtime:latest
    inference: {}

training:
  provider: runpod
  type: qlora
  hyperparams:
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 1
    learning_rate: 2.0e-4
    warmup_ratio: 0.0
    epochs: 1
  qlora:
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
  enabled: false
  provider: single_node
  engine: vllm
  engines:
    vllm:
      merge_image: test/merge:latest
      serve_image: test/vllm:latest

datasets:
  default:
    source_type: local
    source_local:
      local_paths:
        train: data/does_not_matter.jsonl
        eval: null

experiment_tracking:
  mlflow:
    tracking_uri: http://127.0.0.1:5002
    experiment_name: test-exp
""",
    )

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    _patch_stage_construction(monkeypatch)
    monkeypatch.setattr(
        "src.pipeline.orchestrator.load_secrets",
        lambda: type("S", (), {"hf_token": "hf_test_token", "runpod_api_key": None})(),
    )

    with pytest.raises(ValueError, match="RUNPOD_API_KEY is required"):
        _ = PipelineOrchestrator(cfg_path)


def test_runpod_key_required_for_runpod_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Negative: inference.provider=runpod must require RUNPOD_API_KEY,
    even when training.provider is not runpod.
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
  single_node:
    connect:
      ssh:
        alias: pc
    training:
      workspace_path: /tmp/workspace
      docker_image: test/training-runtime:latest
    inference:
      serve:
        workspace: /tmp/test_inference
        host: 127.0.0.1
        port: 8000
  runpod:
    connect:
      ssh:
        key_path: /tmp/id_ed25519
    cleanup: {}
    training:
      # Required by RunPodProviderConfig schema (even if training.provider is different)
      image_name: test/training-runtime:latest
      gpu_type: "NVIDIA A40"
    inference:
      volume:
        id: nv_test_123
        name: helix-test-volume
        size_gb: 50
      pod:
        image_name: test/inference-vllm:v0.1.1
      serve:
        port: 8000

training:
  provider: single_node
  type: qlora
  hyperparams:
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 1
    learning_rate: 2.0e-4
    warmup_ratio: 0.0
    epochs: 1
  qlora:
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
  enabled: true
  provider: runpod
  engine: vllm
  engines:
    vllm:
      merge_image: test/merge:latest
      serve_image: test/vllm:latest

datasets:
  default:
    source_type: local
    source_local:
      local_paths:
        train: data/does_not_matter.jsonl
        eval: null

experiment_tracking:
  mlflow:
    tracking_uri: http://127.0.0.1:5002
    experiment_name: test-exp
""",
    )

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    _patch_stage_construction(monkeypatch)
    monkeypatch.setattr(
        "src.pipeline.orchestrator.load_secrets",
        lambda: type("S", (), {"hf_token": "hf_test_token", "runpod_api_key": None})(),
    )

    with pytest.raises(ValueError, match="inference\\.provider='runpod'"):
        _ = PipelineOrchestrator(cfg_path)


