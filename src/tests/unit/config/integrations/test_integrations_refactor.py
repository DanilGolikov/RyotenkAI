"""Schema contract tests for ``integrations.*``.

After the boundary refactor (Variant 1 follow-up), core's
``MLflowConfig`` accepts inline values directly — there's no
``MLflowTrackingRef`` and no ``Union`` shape on
``IntegrationsConfig.mlflow``. Project YAMLs that use the
``integration: <id>`` shorthand have it expanded by the UX-layer
resolver (``src.workspace.integrations.resolver``) BEFORE this
schema is validated.

These tests pin the schema as it is post-resolver; the resolver
itself has its own test suite at
``src/tests/unit/workspace/integrations/test_resolver.py``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.integrations.root import IntegrationsConfig
from src.config.integrations.huggingface import HuggingFaceHubConfig
from src.config.integrations.mlflow import MLflowConfig
from src.config.integrations.mlflow_integration import MLflowIntegrationConfig

# ---------------------------------------------------------------------------
# MLflowConfig — post-resolver shape
# ---------------------------------------------------------------------------


class TestMLflowConfig:
    def test_inline_tracking_uri_accepted(self) -> None:
        """Inline ``tracking_uri`` is valid (no integration shortcut
        used). This is what the resolver produces and also a path
        users can take directly."""
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

    def test_neither_uri_rejected_with_hint(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            MLflowConfig(experiment_name="exp")
        msg = str(exc_info.value)
        assert "tracking_uri" in msg
        # Tip about the shortcut should be in the error.
        assert "integration" in msg

    def test_integration_secrets_tag_round_trips(self) -> None:
        """The ``integration`` field is preserved as a secrets-tag —
        runtime code passes it to ``secrets.get_provider_token``."""
        cfg = MLflowConfig(
            tracking_uri="https://mlflow.example.com",
            experiment_name="exp",
            integration="prod-mlflow",
        )
        assert cfg.integration == "prod-mlflow"

    def test_experiment_name_required(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            MLflowConfig(tracking_uri="https://x")
        assert "experiment_name" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# HuggingFaceHubConfig — unchanged (still ref-style)
# ---------------------------------------------------------------------------


class TestHuggingFaceConfig:
    def test_repo_id_required_when_integration_set(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            HuggingFaceHubConfig(integration="hf-prod")
        assert "repo_id is required" in str(exc_info.value)

    def test_integration_required_when_repo_id_set(self) -> None:
        # Half-configured block — repo_id without integration. Schema
        # used to silently accept this (→ "HF Hub upload disabled" at
        # runtime); now it fails at config-load time so the operator
        # sees the typo before the run starts.
        with pytest.raises(ValidationError) as exc_info:
            HuggingFaceHubConfig(repo_id="me/model")
        assert "integration is required" in str(exc_info.value)

    def test_empty_block_accepted(self) -> None:
        # Empty block = HF disabled. No fields → no validation → fine.
        cfg = HuggingFaceHubConfig()
        assert cfg.integration is None
        assert cfg.repo_id is None

    def test_enabled_derives_from_integration(self) -> None:
        empty = HuggingFaceHubConfig()
        assert empty.enabled is False
        filled = HuggingFaceHubConfig(integration="hf-prod", repo_id="me/model")
        assert filled.enabled is True


# ---------------------------------------------------------------------------
# IntegrationsConfig — assembled shape + legacy guards
# ---------------------------------------------------------------------------


class TestExperimentTracking:
    def test_full_shape_assembles(self) -> None:
        cfg = IntegrationsConfig(
            mlflow={
                "tracking_uri": "https://mlflow.example.com",
                "experiment_name": "exp",
                "integration": "prod-mlflow",
            },
            huggingface={
                "integration": "hf-prod",
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
