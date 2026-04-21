"""
Lightweight restart-point query — no orchestrator, no stage init, no secrets.

Callers that only have ``(config, run_dir)`` and don't want the network
health probe use this function. It shares
:func:`src.pipeline.restart_rules.compute_restart_points` with
:class:`RestartPointsInspector` so the decision rules stay in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.pipeline._types import CANONICAL_STAGE_ORDER
from src.pipeline.restart_rules import compute_restart_points
from src.pipeline.state import PipelineStateStore, hash_payload

if TYPE_CHECKING:
    from pathlib import Path

    from src.utils.config import PipelineConfig


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


def list_restart_points(run_dir: Path, config: PipelineConfig) -> list[dict[str, Any]]:
    """Return per-stage restart availability without performing a health probe.

    The health probe is a network call with timeout — callers that just want
    a quick "can I restart from here?" answer skip it. The orchestrator's
    :class:`RestartPointsInspector` performs the probe when actually
    preparing a run.
    """
    store = PipelineStateStore(run_dir.expanduser().resolve())
    state = store.load()
    config_hashes = compute_config_hashes(config)
    return compute_restart_points(
        state=state,
        stage_names=CANONICAL_STAGE_ORDER,
        config_hashes=config_hashes,
        # Health probe intentionally omitted — this is the "fast" path.
        # It belongs at runtime (fail-fast in _validate_stage_prerequisites).
        inference_health_checker=lambda _ctx: True,
    )
