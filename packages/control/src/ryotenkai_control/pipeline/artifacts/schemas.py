"""TypedDict schemas for the `data` field of each stage's StageArtifactEnvelope.

Each TypedDict defines the exact structure of the `data` payload written by
the corresponding pipeline stage. The schemas serve as a contract between:
  - the orchestrator / stage (writer side)
  - the MLflowAdapter + ReportBuilder (reader side)
"""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Dataset Validator
# ---------------------------------------------------------------------------


class ValidationPluginData(TypedDict):
    """Result of a single validation plugin for one dataset split."""

    id: str
    plugin_name: str
    passed: bool
    duration_ms: float
    description: str
    metrics: dict[str, Any]
    params: dict[str, Any]
    thresholds: dict[str, Any]
    errors: list[str]
    recommendations: list[str]


class ValidationDatasetData(TypedDict):
    """Aggregated result for a single dataset file (split)."""

    name: str
    path: str
    sample_count: int | None
    status: str  # "passed" | "failed"
    critical_failures: int
    plugins: list[ValidationPluginData]


class ValidationArtifactData(TypedDict):
    """Top-level data payload for dataset_validator_results.json."""

    datasets: list[ValidationDatasetData]


# ---------------------------------------------------------------------------
# Model Evaluator
# ---------------------------------------------------------------------------


class EvalPluginData(TypedDict):
    """Result of a single evaluation plugin."""

    plugin_name: str
    passed: bool
    description: str
    params: dict[str, Any]
    thresholds: dict[str, Any]
    metrics: dict[str, float]
    errors: list[str]
    recommendations: list[str]
    sample_count: int
    failed_samples: int


class EvalArtifactData(TypedDict):
    """Top-level data payload for evaluation_results.json."""

    overall_passed: bool
    sample_count: int
    duration_seconds: float
    skipped_plugins: list[str]
    errors: list[str]
    plugins: dict[str, EvalPluginData]


# ---------------------------------------------------------------------------
# GPU Deployer
# ---------------------------------------------------------------------------


class DeploymentArtifactData(TypedDict):
    """Data payload for gpu_deployer_results.json."""

    upload_duration_seconds: float | None
    deps_duration_seconds: float | None
    provider_name: str | None
    provider_type: str | None
    gpu_type: str | None
    resource_id: str | None


# ---------------------------------------------------------------------------
# Training Monitor
# ---------------------------------------------------------------------------


class TrainingArtifactData(TypedDict):
    """Data payload for training_monitor_results.json."""

    training_duration_seconds: float | None


# ---------------------------------------------------------------------------
# Model Retriever
# ---------------------------------------------------------------------------


class ModelArtifactData(TypedDict):
    """Data payload for model_retriever_results.json."""

    model_size_mb: float | None
    hf_repo_id: str | None
    upload_duration_seconds: float | None


# ---------------------------------------------------------------------------
# Inference Deployer
# ---------------------------------------------------------------------------


class InferenceArtifactData(TypedDict):
    """Data payload for inference_deployer_results.json."""

    endpoint_url: str | None
    model_name: str | None
    provider: str | None
