"""Schema contract for PR3: experiment_tracking.* as references to Settings integrations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.integrations.experiment_tracking import ExperimentTrackingConfig
from src.config.integrations.huggingface import HuggingFaceHubConfig
from src.config.integrations.mlflow import MLflowTrackingRef


def test_mlflow_ref_requires_experiment_name_when_integration_set() -> None:
    with pytest.raises(ValidationError) as exc_info:
        MLflowTrackingRef(integration="mlflow-prod")
    assert "experiment_name is required" in str(exc_info.value)


def test_mlflow_ref_empty_is_valid() -> None:
    # No integration selected = block inactive, no experiment_name needed.
    ref = MLflowTrackingRef()
    assert ref.integration is None


def test_mlflow_ref_full_shape() -> None:
    ref = MLflowTrackingRef(
        integration="mlflow-prod",
        experiment_name="qwen-7b",
        run_description_file="./desc.md",
    )
    assert ref.integration == "mlflow-prod"
    assert ref.experiment_name == "qwen-7b"


def test_hf_ref_requires_repo_id_when_integration_set() -> None:
    with pytest.raises(ValidationError) as exc_info:
        HuggingFaceHubConfig(integration="hf-prod")
    assert "repo_id is required" in str(exc_info.value)


def test_hf_ref_enabled_derives_from_integration() -> None:
    empty = HuggingFaceHubConfig()
    assert empty.enabled is False
    filled = HuggingFaceHubConfig(integration="hf-prod", repo_id="me/model")
    assert filled.enabled is True


def test_legacy_mlflow_tracking_uri_rejected_with_hint() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ExperimentTrackingConfig(
            mlflow={
                "tracking_uri": "http://localhost:5002",
                "experiment_name": "exp",
            }
        )
    msg = str(exc_info.value)
    assert "tracking_uri" in msg
    assert "Settings" in msg or "Integrations" in msg


def test_legacy_mlflow_ca_bundle_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ExperimentTrackingConfig(
            mlflow={
                "ca_bundle_path": "/etc/certs/mlflow.pem",
                "integration": "x",
                "experiment_name": "y",
            }
        )
    assert "ca_bundle_path" in str(exc_info.value)


def test_legacy_hf_enabled_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ExperimentTrackingConfig(
            huggingface={
                "enabled": True,
                "integration": "hf-prod",
                "repo_id": "me/model",
            }
        )
    assert "enabled" in str(exc_info.value)


def test_new_shape_passes() -> None:
    cfg = ExperimentTrackingConfig(
        mlflow={"integration": "mlflow-prod", "experiment_name": "qwen-7b"},
        huggingface={"integration": "hf-prod", "repo_id": "me/qwen-7b", "private": True},
    )
    assert cfg.mlflow is not None
    assert cfg.mlflow.integration == "mlflow-prod"
    assert cfg.huggingface is not None
    assert cfg.huggingface.repo_id == "me/qwen-7b"


def test_get_report_to_active_when_integration_set() -> None:
    cfg = ExperimentTrackingConfig(
        mlflow={"integration": "m", "experiment_name": "e"},
    )
    assert cfg.get_report_to() == ["mlflow"]


def test_get_report_to_none_when_integration_empty() -> None:
    cfg = ExperimentTrackingConfig(mlflow=None)
    assert cfg.get_report_to() == ["none"]


# ─── PR5b: tracking_uri on integration is now required ────────────────────


def test_mlflow_integration_rejects_missing_tracking_uri() -> None:
    """A reusable MLflow integration without a tracker is a dead ref;
    the schema must say so up-front rather than letting it round-trip
    silently and blow up later at connection time."""
    from src.config.integrations.mlflow_integration import MLflowIntegrationConfig

    with pytest.raises(ValidationError) as exc_info:
        MLflowIntegrationConfig()
    assert "tracking_uri" in str(exc_info.value)


def test_mlflow_integration_rejects_empty_tracking_uri() -> None:
    from src.config.integrations.mlflow_integration import MLflowIntegrationConfig

    with pytest.raises(ValidationError) as exc_info:
        MLflowIntegrationConfig(tracking_uri="   ")
    assert "tracking_uri" in str(exc_info.value)


def test_mlflow_integration_accepts_valid_tracking_uri() -> None:
    from src.config.integrations.mlflow_integration import MLflowIntegrationConfig

    cfg = MLflowIntegrationConfig(tracking_uri="http://localhost:5002")
    assert cfg.tracking_uri == "http://localhost:5002"
