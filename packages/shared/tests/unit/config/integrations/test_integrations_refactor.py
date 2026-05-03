"""Schema contract tests for ``integrations.*``.

Project YAMLs configure ``mlflow`` and ``huggingface`` inline.
``MLflowIntegrationConfig`` is the separate Settings-side schema for
a reusable integration entry's own ``current.yaml`` (the file the Web
UI writes when the user creates a Settings → Integrations → MLflow
entry). It stays even though project YAMLs no longer reference it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_shared.config.integrations.root import IntegrationsConfig
from ryotenkai_shared.config.integrations.huggingface import HuggingFaceHubConfig
from ryotenkai_shared.config.integrations.mlflow import MLflowConfig
from ryotenkai_shared.config.integrations.mlflow_integration import MLflowIntegrationConfig

# ---------------------------------------------------------------------------
# MLflowConfig — inline project shape
# ---------------------------------------------------------------------------


class TestMLflowConfig:
    def test_inline_tracking_uri_accepted(self) -> None:
        cfg = MLflowConfig(
            tracking_uri="https://mlflow.example.com",
            experiment_name="exp",
        )
        assert cfg.tracking_uri == "https://mlflow.example.com"

    def test_only_local_tracking_uri_accepted(self) -> None:
        cfg = MLflowConfig(
            local_tracking_uri="http://localhost:5002",
            experiment_name="exp",
        )
        assert cfg.local_tracking_uri == "http://localhost:5002"

    def test_neither_uri_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            MLflowConfig(experiment_name="exp")
        assert "tracking_uri" in str(exc_info.value)

    def test_experiment_name_required(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            MLflowConfig(tracking_uri="https://x")
        assert "experiment_name" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# HuggingFaceHubConfig — repo_id-only schema
# ---------------------------------------------------------------------------


class TestHuggingFaceConfig:
    def test_empty_block_accepted(self) -> None:
        cfg = HuggingFaceHubConfig()
        assert cfg.repo_id is None
        assert cfg.enabled is False

    def test_repo_id_only_block_accepted(self) -> None:
        cfg = HuggingFaceHubConfig(repo_id="me/model")
        assert cfg.repo_id == "me/model"
        assert cfg.enabled is True

    def test_enabled_derives_from_repo_id(self) -> None:
        empty = HuggingFaceHubConfig()
        assert empty.enabled is False
        filled = HuggingFaceHubConfig(repo_id="me/model")
        assert filled.enabled is True

    def test_repo_id_whitespace_normalized_to_none(self) -> None:
        cfg = HuggingFaceHubConfig(repo_id="   ")
        assert cfg.repo_id is None
        assert cfg.enabled is False


# ---------------------------------------------------------------------------
# IntegrationsConfig — assembled shape + legacy guards
# ---------------------------------------------------------------------------


class TestExperimentTracking:
    def test_full_shape_assembles(self) -> None:
        cfg = IntegrationsConfig(
            mlflow={
                "tracking_uri": "https://mlflow.example.com",
                "experiment_name": "exp",
            },
            huggingface={
                "repo_id": "me/model",
                "private": True,
            },
        )
        assert cfg.mlflow is not None
        assert cfg.mlflow.tracking_uri == "https://mlflow.example.com"
        assert cfg.huggingface is not None
        assert cfg.huggingface.repo_id == "me/model"

    def test_get_report_to_active_when_uri_set(self) -> None:
        cfg = IntegrationsConfig(
            mlflow={
                "tracking_uri": "https://x",
                "experiment_name": "exp",
            },
        )
        assert cfg.get_report_to() == ["mlflow"]

    def test_get_report_to_none_when_mlflow_absent(self) -> None:
        cfg = IntegrationsConfig(mlflow=None)
        assert cfg.get_report_to() == ["none"]

    def test_removed_system_metrics_field_rejected_with_hint(self) -> None:
        """Migration guard for the second refactor pass — fields
        deleted from the nested ``system_metrics:`` block surface a
        targeted hint instead of the generic ``extra_forbidden``."""
        with pytest.raises(ValidationError) as exc_info:
            IntegrationsConfig(
                mlflow={
                    "tracking_uri": "https://x",
                    "experiment_name": "exp",
                    "system_metrics": {"sampling_interval": 5},
                },
            )
        msg = str(exc_info.value)
        assert "sampling_interval" in msg
        assert "system_metrics" in msg


# ---------------------------------------------------------------------------
# MLflowIntegrationConfig — schema for the integration's own current.yaml
# ---------------------------------------------------------------------------


class TestMLflowIntegrationConfig:
    def test_rejects_missing_tracking_uri(self) -> None:
        """A reusable MLflow integration without a tracker is a dead
        ref; the schema must say so up-front rather than letting it
        round-trip silently and blow up later at connection time."""
        with pytest.raises(ValidationError) as exc_info:
            MLflowIntegrationConfig()
        assert "tracking_uri" in str(exc_info.value)

    def test_rejects_empty_tracking_uri(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            MLflowIntegrationConfig(tracking_uri="   ")
        assert "tracking_uri" in str(exc_info.value)

    def test_accepts_valid_tracking_uri(self) -> None:
        cfg = MLflowIntegrationConfig(tracking_uri="http://localhost:5002")
        assert cfg.tracking_uri == "http://localhost:5002"
