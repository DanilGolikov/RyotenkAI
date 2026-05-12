"""
E2E (thick) test for NEW dataset schema:

Chain covered (mostly real code, heavy parts mocked):
- YAML -> PipelineConfig.from_yaml (source_root)
- TrainingDeploymentManager.deploy_files() builds local_abs -> remote_rel mapping (no SSH upload, mocked)
- Simulate provider workspace materialization using that mapping
- Training DatasetLoader.load_for_phase() loads from auto-generated data/{strategy_type}/{basename} (runtime view)
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ryotenkai_control.pipeline.stages.managers.deployment_manager import TrainingDeploymentManager
from ryotenkai_shared.config import PipelineConfig
from ryotenkai_pod.trainer.container import TrainingContainer
from ryotenkai_shared.utils.result import Ok


@dataclass(frozen=True)
class _DummySecrets:
    hf_token: str = "hf_test_token"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "xfail-debt:file-uploader-async-refactor — YAML schema migrated "
        "to the typed `source` / `adapter` / `engine` discriminators "
        "(was the original block); the remaining failure is that "
        "`FileUploader._upload_files_batch` was replaced by async "
        "`_upload_all`. Rewrite needs async patching + signature update. "
        "Tracked in xfail_debt.md."
    ),
)
def test_yaml_to_deploy_mapping_to_runtime_load_chain(tmp_path: Path, monkeypatch) -> None:
    # ---------------------------------------------------------------------
    # Arrange: emulate llm_pipeline_datas-like layout
    # ---------------------------------------------------------------------
    root = tmp_path / "llm_pipeline_datas"
    cfg_dir = root / "config"
    data_dir = root / "data"
    cfg_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    train_local = data_dir / "train.jsonl"
    eval_local = data_dir / "eval.jsonl"
    train_local.write_text('{"text":"train"}\n', encoding="utf-8")
    eval_local.write_text('{"text":"eval"}\n', encoding="utf-8")

    cfg_path = cfg_dir / "pipeline_config.yaml"
    cfg_path.write_text(
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
    inference:
      serve:
        workspace: /tmp/test_inference
        host: 127.0.0.1
        port: 8000

training:
  provider: single_node
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
  hyperparams:
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 1
    learning_rate: 2.0e-4
    warmup_ratio: 0.0
    epochs: 1
    eval_steps: 7
  strategies:
    - strategy_type: sft
      dataset: default

inference:
  enabled: false
  provider: single_node
  engine:
    kind: vllm

datasets:
  default:
    source:
      kind: local
      local_paths:
        train: data/train.jsonl
        eval: data/eval.jsonl
""".strip()
        + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------------------
    # Act 1: YAML -> PipelineConfig (defaulting + source_root)
    # ---------------------------------------------------------------------
    cfg = PipelineConfig.from_yaml(cfg_path)

    assert cfg.get_source_root() == root
    ds = cfg.datasets["default"]
    # Typed discriminated union: ds.source.kind == "local" with local_paths
    assert ds.source.kind == "local"
    assert ds.source.local_paths.train == "data/train.jsonl"
    assert ds.source.local_paths.eval == "data/eval.jsonl"

    # ---------------------------------------------------------------------
    # Act 2: DeploymentManager builds local_abs -> remote_rel mapping
    # ---------------------------------------------------------------------
    deployment = TrainingDeploymentManager(config=cfg, secrets=_DummySecrets())
    deployment.set_workspace(workspace_path=str(tmp_path / "remote_workspace"))

    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")

    with (
        patch.object(deployment._file_uploader, "_upload_files_batch", return_value=Ok(None)) as mock_batch,
        patch.object(deployment._code_syncer, "sync", return_value=Ok(None)),
    ):
        res = deployment.deploy_files(ssh_client, {"config_path": str(cfg_path)})

    assert res.is_ok()
    files_to_upload: list[tuple[str, str]] = mock_batch.call_args[0][1]

    # Extract only dataset uploads (remote_rel under data/)
    dataset_pairs = [(Path(local), remote) for (local, remote) in files_to_upload if remote.startswith("data/")]
    assert (Path(train_local).resolve(), "data/sft/train.jsonl") in dataset_pairs
    assert (Path(eval_local).resolve(), "data/sft/eval.jsonl") in dataset_pairs

    # ---------------------------------------------------------------------
    # Act 3: simulate provider workspace materialization and runtime load
    # ---------------------------------------------------------------------
    remote_root = tmp_path / "provider_run"
    remote_root.mkdir()
    for local_abs, remote_rel in dataset_pairs:
        dst = remote_root / remote_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_abs, dst)

    # Training runtime: working dir = workspace/run root
    monkeypatch.chdir(remote_root)

    # Patch datasets.load_dataset used inside training DatasetLoader module
    train_ds = MagicMock()
    train_ds.__len__.return_value = 1
    eval_ds = MagicMock()
    eval_ds.__len__.return_value = 1

    with patch("ryotenkai_pod.trainer.orchestrator.dataset_loader.load_dataset", side_effect=[train_ds, eval_ds]) as mock_ld:
        loader = TrainingContainer(config=cfg).dataset_loader
        phase = cfg.training.strategies[0]
        out = loader.load_for_phase(phase)

    assert out.is_success()
    loaded_train, loaded_eval = out.unwrap()
    assert loaded_train is train_ds
    assert loaded_eval is eval_ds

    # Ensure load_dataset was called with training_paths (runtime view)
    assert mock_ld.call_count == 2
    first_call_kwargs = mock_ld.call_args_list[0].kwargs
    second_call_kwargs = mock_ld.call_args_list[1].kwargs
    assert Path(first_call_kwargs["data_files"]).as_posix().endswith("data/sft/train.jsonl")
    assert Path(second_call_kwargs["data_files"]).as_posix().endswith("data/sft/eval.jsonl")


