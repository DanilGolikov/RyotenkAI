"""
Lightweight restart-point query — no orchestrator, no stage init, no secrets.

Only needs:
  - load_config()     to compute config hashes
  - PipelineStateStore.load()  to read persisted state
  - hash_payload()    to reproduce the same hashes the orchestrator stores
  - urlopen           to probe the inference health endpoint
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.request import urlopen

from src.pipeline._types import CANONICAL_STAGE_ORDER, StageNames
from src.pipeline.state import PipelineStateStore, hash_payload

if TYPE_CHECKING:
    from pathlib import Path

    from src.utils.config import PipelineConfig

_HTTP_OK_MIN = 200
_HTTP_ERROR_MIN = 400


def compute_config_hashes(config: PipelineConfig) -> dict[str, str]:
    model_dataset_payload = {
        "model": config.model.model_dump(mode="json"),
        "training": config.training.model_dump(mode="json"),
        "datasets": {name: cfg.model_dump(mode="json") for name, cfg in config.datasets.items()},
    }
    training_payload = {
        **model_dataset_payload,
        "provider_name": config.get_active_provider_name(),
        "provider": config.get_provider_config(),
    }
    late_payload = {
        "inference": config.inference.model_dump(mode="json"),
        "evaluation": config.evaluation.model_dump(mode="json"),
    }
    return {
        "training_critical": hash_payload(training_payload),
        "late_stage": hash_payload(late_payload),
        "model_dataset": hash_payload(model_dataset_payload),
    }


def is_inference_runtime_healthy(inference_ctx: dict[str, Any]) -> bool:
    endpoint_info = inference_ctx.get("endpoint_info")
    if not isinstance(endpoint_info, dict):
        endpoint_info = {}
    health_url = endpoint_info.get("health_url") or inference_ctx.get("endpoint_url")
    if not isinstance(health_url, str) or not health_url:
        return False
    try:
        with urlopen(health_url, timeout=5) as response:
            return _HTTP_OK_MIN <= int(getattr(response, "status", _HTTP_OK_MIN)) < _HTTP_ERROR_MIN
    except Exception:
        return False


def list_restart_points(run_dir: Path, config: PipelineConfig) -> list[dict[str, Any]]:
    """
    Read pipeline_state.json and return restart availability for each stage.
    Does NOT require PipelineOrchestrator — safe to call with only a parsed config.
    """
    store = PipelineStateStore(run_dir.expanduser().resolve())
    state = store.load()
    config_hashes = compute_config_hashes(config)

    points: list[dict[str, Any]] = []

    for stage_name in CANONICAL_STAGE_ORDER:
        available = True
        reason = "restart_allowed"
        mode = "fresh_only"

        if stage_name == StageNames.TRAINING_MONITOR:
            mode = "reconnect_only"
            ref = state.current_output_lineage.get(StageNames.GPU_DEPLOYER)
            gpu_outputs = ref.outputs if ref else {}
            if not all(gpu_outputs.get(k) for k in ("ssh_host", "ssh_port", "workspace_path")):
                available = False
                reason = "missing_gpu_deployer_outputs"

        elif stage_name == StageNames.MODEL_RETRIEVER:
            mode = "fresh_or_resume"
            ref = state.current_output_lineage.get(StageNames.GPU_DEPLOYER)
            if ref is None:
                available = False
                reason = "missing_gpu_deployer_outputs"

        elif stage_name == StageNames.INFERENCE_DEPLOYER:
            mode = "fresh_or_resume"
            ref = state.current_output_lineage.get(StageNames.MODEL_RETRIEVER)
            outputs = ref.outputs if ref else {}
            if not (outputs.get("hf_repo_id") or outputs.get("local_model_path")):
                available = False
                reason = "missing_model_retriever_outputs"

        elif stage_name == StageNames.MODEL_EVALUATOR:
            mode = "live_runtime_only"
            ref = state.current_output_lineage.get(StageNames.INFERENCE_DEPLOYER)
            if ref is None:
                available = False
                reason = "missing_inference_outputs"
            # Health probe is intentionally omitted here — it's a network call with timeout
            # and belongs at runtime (fail-fast in _validate_stage_prerequisites).

        if state.model_dataset_config_hash:
            # Fine-grained check: only model/training/datasets matter, provider change is allowed
            if state.model_dataset_config_hash != config_hashes["model_dataset"]:
                available = False
                reason = "training_critical_config_changed"
        elif state.training_critical_config_hash != config_hashes["training_critical"]:
            # Legacy check for states without model_dataset_config_hash
            available = False
            reason = "training_critical_config_changed"

        if state.late_stage_config_hash != config_hashes["late_stage"] and stage_name not in {
            StageNames.INFERENCE_DEPLOYER,
            StageNames.MODEL_EVALUATOR,
        }:
            available = False
            reason = "late_stage_config_changed"

        points.append({"stage": stage_name, "available": available, "mode": mode, "reason": reason})

    return points
