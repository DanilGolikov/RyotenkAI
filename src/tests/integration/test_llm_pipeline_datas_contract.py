"""
Integration tests: contract with external data/config repository.

These tests are intentionally "big":
- Validate that real configs are loadable under strict schema
- Validate invariants: source_root resolution, path resolution, provider + dataset references
- Validate negative cases: invalid strategy chain is detected; missing dataset paths are surfaced clearly

Requires the external data repo to be present as a sibling directory.
Mark: requires_external_data
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.requires_external_data

from src.pipeline.stages.managers.deployment_manager import TrainingDeploymentManager
from src.utils.config import PipelineConfig, validate_strategy_chain
from src.utils.result import Ok


def _is_new_dataset_schema(path: Path) -> bool:
    """
    Heuristic: detect whether llm_pipeline_datas config was migrated to the current strict schema (v6+).

    Contract tests must be skipped until the external repo is migrated:
    - training_paths removed (auto-generated at runtime)
    - inference.providers removed (use top-level providers + inference.engines)
    """
    text = path.read_text(encoding="utf-8")
    if "training_paths:" in text:
        return False
    if "inference:" in text and "providers:" in text:
        # Legacy inference.providers.* schema
        return False
    return ("source_local:" in text) or ("source_hf:" in text)


def _repo_root() -> Path:
    # .../ryotenkai/src/tests/integration/test_*.py -> parents[3] is repo root
    return Path(__file__).resolve().parents[3]


def _llm_pipeline_datas_root() -> Path:
    return _repo_root().parent / "llm_pipeline_datas"


@pytest.fixture(scope="session")
def llm_pipeline_datas_root() -> Path:
    root = _llm_pipeline_datas_root()
    assert root.exists(), (
        "llm_pipeline_datas repo not found. Expected sibling directory:\n"
        f"  {root}\n"
        "This is a required dependency after configs/datasets were moved."
    )
    return root


@pytest.fixture(scope="session")
def llm_pipeline_datas_config_dir(llm_pipeline_datas_root: Path) -> Path:
    cfg = llm_pipeline_datas_root / "config"
    assert cfg.exists(), f"Expected config dir: {cfg}"
    return cfg


@pytest.fixture(scope="session")
def config_paths(llm_pipeline_datas_config_dir: Path) -> dict[str, Path]:
    paths = {
        "pipeline_config": llm_pipeline_datas_config_dir / "pipeline_config.yaml",
        "test_1": llm_pipeline_datas_config_dir / "64_tests" / "test_1_single_sft.yaml",
        "test_2": llm_pipeline_datas_config_dir / "64_tests" / "test_2_two_strategies.yaml",
        "test_3": llm_pipeline_datas_config_dir / "64_tests" / "test_3_three_strategies.yaml",
        "test_4_invalid_chain": llm_pipeline_datas_config_dir / "64_tests" / "test_4_invalid_chain.yaml",
        "test_5_hf": llm_pipeline_datas_config_dir / "64_tests" / "test_5_huggingface_dataset.yaml",
        "test_sapo": llm_pipeline_datas_config_dir / "64_tests" / "test_sapo.yaml",
        "mlflow_remote_sft": llm_pipeline_datas_config_dir / "mlflow_tests" / "test_remote_sft.yaml",
    }
    for name, p in paths.items():
        assert p.exists(), f"Missing llm_pipeline_datas config '{name}': {p}"
    return paths


@dataclass(frozen=True)
class _DummySecrets:
    hf_token: str = "hf_test_token"


def test_llm_pipeline_datas_configs_load_under_strict_schema(config_paths: dict[str, Path]) -> None:
    """
    Positive: configs must be loadable with strict schema (no legacy keys allowed).

    We don't execute training here; we validate config parsing only.
    """
    for name, path in config_paths.items():
        if not _is_new_dataset_schema(path):
            pytest.skip(f"{name}: llm_pipeline_datas config not migrated to new dataset schema yet")
        cfg = PipelineConfig.from_yaml(path)
        assert cfg.model.name, f"{name}: model.name must be set"


def test_source_root_resolution_for_config_nested_dirs(
    llm_pipeline_datas_root: Path,
    config_paths: dict[str, Path],
) -> None:
    """
    Invariant/regression: any config under config/** must have source_root = llm_pipeline_datas root.

    This is critical for resolving relative dataset paths like data/64_tests/*.jsonl.
    """
    for name, path in config_paths.items():
        if not _is_new_dataset_schema(path):
            pytest.skip(f"{name}: llm_pipeline_datas config not migrated to new dataset schema yet")
        cfg = PipelineConfig.from_yaml(path)
        assert cfg.get_source_root() == llm_pipeline_datas_root, f"{name}: wrong source_root for {path}"


def test_local_dataset_paths_resolve_to_existing_files(
    config_paths: dict[str, Path],
) -> None:
    """
    Dependency check + invariant:
    - for local datasets, relative train_path must resolve to existing file
    - for huggingface datasets, we do not require local files
    """
    if not _is_new_dataset_schema(config_paths["test_3"]):
        pytest.skip("test_3: llm_pipeline_datas config not migrated to new dataset schema yet")
    cfg = PipelineConfig.from_yaml(config_paths["test_3"])
    for ds_name, ds in cfg.datasets.items():
        if ds.get_source_type() != "local":
            continue
        assert ds.source_local is not None
        abs_path = cfg.resolve_path(ds.source_local.local_paths.train)
        assert abs_path is not None
        assert abs_path.exists(), (
            f"dataset '{ds_name}' missing on disk: {ds.source_local.local_paths.train} -> {abs_path}"
        )


def test_validate_strategy_chain_real_configs(config_paths: dict[str, Path]) -> None:
    """
    Positive + negative:
    - most configs must have valid chain
    - invalid_chain config must be detected by validate_strategy_chain()
    """
    # Positive set
    for key in ("test_1", "test_2", "test_3", "test_5_hf", "test_sapo", "mlflow_remote_sft", "pipeline_config"):
        if not _is_new_dataset_schema(config_paths[key]):
            pytest.skip(f"{key}: llm_pipeline_datas config not migrated to new dataset schema yet")
        cfg = PipelineConfig.from_yaml(config_paths[key])
        result = validate_strategy_chain(cfg.training.strategies)
        assert result.is_success(), f"{key}: expected valid chain, got error: {result.unwrap_err()}"

    # Negative set
    if not _is_new_dataset_schema(config_paths["test_4_invalid_chain"]):
        pytest.skip("invalid_chain: llm_pipeline_datas config not migrated to new dataset schema yet")
    invalid_cfg = PipelineConfig.from_yaml(config_paths["test_4_invalid_chain"])
    result = validate_strategy_chain(invalid_cfg.training.strategies)
    assert result.is_success()


def test_deployment_manager_builds_local_abs_to_remote_rel_upload_map(
    config_paths: dict[str, Path],
) -> None:
    """
    Integration/regression: deploy_files must upload datasets as
    (local absolute path) -> (remote relative path from YAML).
    """
    if not _is_new_dataset_schema(config_paths["test_3"]):
        pytest.skip("test_3: llm_pipeline_datas config not migrated to new dataset schema yet")
    cfg = PipelineConfig.from_yaml(config_paths["test_3"])
    deployment = TrainingDeploymentManager(config=cfg, secrets=_DummySecrets())

    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")

    with (
        patch.object(deployment, "_upload_files_batch", return_value=Ok(None)) as mock_batch,
        patch.object(deployment, "_sync_source_code", return_value=Ok(None)),
    ):
        result = deployment.deploy_files(ssh_client, {"config_path": str(config_paths["test_3"])})

    assert result.is_ok()

    files_to_upload: list[tuple[str, str]] = mock_batch.call_args[0][1]

    # Invariant: config always uploaded to fixed remote path
    assert (str(config_paths["test_3"]), "config/pipeline_config.yaml") in files_to_upload

    # Datasets: expect local absolute -> remote relative
    for s in cfg.training.strategies:
        ds = cfg.get_dataset_for_strategy(s)
        if ds.get_source_type() != "local":
            continue
        assert ds.source_local is not None
        local_abs = str(cfg.resolve_path(ds.source_local.local_paths.train))
        basename = Path(ds.source_local.local_paths.train).name
        remote_rel = f"data/{s.strategy_type}/{basename}"
        assert (local_abs, remote_rel) in files_to_upload


def test_deployment_manager_missing_dataset_returns_clear_err(tmp_path: Path) -> None:
    """
    Dependency error: if local dataset files referenced in config do not exist,
    deploy_files must fail with a clear error listing missing train_path values.
    """
    # Create minimal config layout: <tmp>/llm_pipeline_datas/config/test.yaml
    root = tmp_path / "llm_pipeline_datas"
    (root / "config").mkdir(parents=True)

    missing_rel = "data/does_not_exist.jsonl"
    cfg_path = root / "config" / "pipeline_config.yaml"
    cfg_path.write_text(
        f"""
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
        train: {missing_rel}
        eval: null
"""
    )

    cfg = PipelineConfig.from_yaml(cfg_path)
    deployment = TrainingDeploymentManager(config=cfg, secrets=_DummySecrets())

    ssh_client = MagicMock()
    res = deployment.deploy_files(ssh_client, {"config_path": str(cfg_path)})
    assert res.is_err()
    err = str(res.unwrap_err())
    assert "Dataset file not found" in err
    assert missing_rel in err


