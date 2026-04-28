"""Tests for the integration resolver.

Covers all categories required by the plan: positive, negative, boundary,
invariants, dependency errors, regressions, logic-specific.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.integrations.exceptions import (
    IntegrationNotFoundError,
    IntegrationUnresolvedError,
)
from src.config.integrations.experiment_tracking import ExperimentTrackingConfig
from src.config.integrations.mlflow import MLflowConfig, MLflowTrackingRef
from src.config.integrations.resolver import resolve_pipeline_config
from src.config.integrations.system_metrics import SystemMetricsConfig
from src.workspace.integrations.registry import IntegrationRegistry
from src.workspace.integrations.store import IntegrationStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_integration(
    workspace_root: Path,
    *,
    integration_id: str,
    integration_type: str,
    current_yaml: str,
) -> None:
    """Lay down a registered integration under ``workspace_root``."""
    registry = IntegrationRegistry(root=workspace_root)
    integration_path = registry.default_integration_path(integration_id)
    store = IntegrationStore(integration_path)
    store.create(
        id=integration_id,
        name=integration_id,
        type=integration_type,
    )
    # Overwrite empty current.yaml with given content.
    (integration_path / "current.yaml").write_text(current_yaml, encoding="utf-8")
    registry.register(
        integration_id=integration_id,
        name=integration_id,
        type=integration_type,
        path=integration_path,
    )


def _mk_pipeline_config(*, mlflow_ref=None, hf_ref=None):
    """Build a minimal PipelineConfig stub for resolver tests.

    We only populate ``experiment_tracking`` since the resolver only
    inspects that block; other fields are not required.
    """
    # Use a SimpleNamespace-like object instead of full PipelineConfig
    # to keep the test surface tiny — resolver only touches
    # ``cfg.experiment_tracking``.
    et = ExperimentTrackingConfig(mlflow=mlflow_ref, huggingface=hf_ref)

    class _Cfg:
        experiment_tracking = et

    return _Cfg()


_MLFLOW_INTEGRATION_YAML = """
tracking_uri: "https://mlflow.example.com"
local_tracking_uri: "http://localhost:5002"
ca_bundle_path: null
system_metrics:
  callback_enabled: true
"""


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_mlflow_ref_resolves_to_config(self, tmp_path: Path) -> None:
        _create_integration(
            tmp_path,
            integration_id="my-mlflow",
            integration_type="mlflow",
            current_yaml=_MLFLOW_INTEGRATION_YAML,
        )
        ref = MLflowTrackingRef(integration="my-mlflow", experiment_name="my-exp")
        cfg = _mk_pipeline_config(mlflow_ref=ref)
        registry = IntegrationRegistry(root=tmp_path)

        resolve_pipeline_config(cfg, registry=registry)

        resolved = cfg.experiment_tracking.mlflow
        assert isinstance(resolved, MLflowConfig)
        assert resolved.tracking_uri == "https://mlflow.example.com"
        assert resolved.local_tracking_uri == "http://localhost:5002"
        assert resolved.ca_bundle_path is None
        assert resolved.experiment_name == "my-exp"
        assert resolved.system_metrics.callback_enabled is True

    def test_resolved_carries_run_description_file_from_ref(self, tmp_path: Path) -> None:
        _create_integration(
            tmp_path,
            integration_id="my-mlflow",
            integration_type="mlflow",
            current_yaml=_MLFLOW_INTEGRATION_YAML,
        )
        ref = MLflowTrackingRef(
            integration="my-mlflow",
            experiment_name="exp",
            run_description_file="runs/desc.md",
        )
        cfg = _mk_pipeline_config(mlflow_ref=ref)
        registry = IntegrationRegistry(root=tmp_path)

        resolve_pipeline_config(cfg, registry=registry)

        resolved = cfg.experiment_tracking.mlflow
        assert isinstance(resolved, MLflowConfig)
        assert resolved.run_description_file == "runs/desc.md"

    def test_huggingface_ref_existence_check_passes(self, tmp_path: Path) -> None:
        from src.config.integrations.huggingface import HuggingFaceHubConfig

        _create_integration(
            tmp_path,
            integration_id="my-hf",
            integration_type="huggingface",
            current_yaml="{}",  # empty schema; HF integration is token-only
        )
        hf_ref = HuggingFaceHubConfig(integration="my-hf", repo_id="user/repo")
        cfg = _mk_pipeline_config(hf_ref=hf_ref)
        registry = IntegrationRegistry(root=tmp_path)

        # Should not raise — and ref stays as HuggingFaceHubConfig (no body merge).
        resolve_pipeline_config(cfg, registry=registry)
        assert cfg.experiment_tracking.huggingface is hf_ref


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_missing_mlflow_integration_raises_not_found(self, tmp_path: Path) -> None:
        ref = MLflowTrackingRef(integration="nonexistent", experiment_name="x")
        cfg = _mk_pipeline_config(mlflow_ref=ref)
        registry = IntegrationRegistry(root=tmp_path)

        with pytest.raises(IntegrationNotFoundError) as exc_info:
            resolve_pipeline_config(cfg, registry=registry)

        err = exc_info.value
        assert err.integration_id == "nonexistent"
        assert err.integration_type == "mlflow"
        assert "nonexistent" in str(err)
        assert "Settings → Integrations" in str(err)

    def test_empty_current_yaml_raises_unresolved(self, tmp_path: Path) -> None:
        _create_integration(
            tmp_path,
            integration_id="empty",
            integration_type="mlflow",
            current_yaml="",  # the typical "stub" case
        )
        ref = MLflowTrackingRef(integration="empty", experiment_name="x")
        cfg = _mk_pipeline_config(mlflow_ref=ref)
        registry = IntegrationRegistry(root=tmp_path)

        with pytest.raises(IntegrationUnresolvedError) as exc_info:
            resolve_pipeline_config(cfg, registry=registry)

        assert "current.yaml is empty" in str(exc_info.value)

    def test_invalid_yaml_raises_unresolved(self, tmp_path: Path) -> None:
        _create_integration(
            tmp_path,
            integration_id="bad-yaml",
            integration_type="mlflow",
            current_yaml=":\n:",  # syntactically invalid YAML
        )
        ref = MLflowTrackingRef(integration="bad-yaml", experiment_name="x")
        cfg = _mk_pipeline_config(mlflow_ref=ref)
        registry = IntegrationRegistry(root=tmp_path)

        with pytest.raises(IntegrationUnresolvedError) as exc_info:
            resolve_pipeline_config(cfg, registry=registry)

        assert "not valid YAML" in str(exc_info.value)

    def test_missing_required_field_raises_unresolved(self, tmp_path: Path) -> None:
        _create_integration(
            tmp_path,
            integration_id="incomplete",
            integration_type="mlflow",
            current_yaml="local_tracking_uri: 'http://localhost:5002'\n",  # no tracking_uri
        )
        ref = MLflowTrackingRef(integration="incomplete", experiment_name="x")
        cfg = _mk_pipeline_config(mlflow_ref=ref)
        registry = IntegrationRegistry(root=tmp_path)

        with pytest.raises(IntegrationUnresolvedError) as exc_info:
            resolve_pipeline_config(cfg, registry=registry)

        assert "schema validation" in str(exc_info.value)

    def test_type_mismatch_raises_unresolved(self, tmp_path: Path) -> None:
        # Register a HuggingFace integration but the project ref is MLflow.
        _create_integration(
            tmp_path,
            integration_id="wrong-type",
            integration_type="huggingface",
            current_yaml="{}",
        )
        ref = MLflowTrackingRef(integration="wrong-type", experiment_name="x")
        cfg = _mk_pipeline_config(mlflow_ref=ref)
        registry = IntegrationRegistry(root=tmp_path)

        with pytest.raises(IntegrationUnresolvedError) as exc_info:
            resolve_pipeline_config(cfg, registry=registry)

        assert "huggingface" in str(exc_info.value)
        assert "mlflow" in str(exc_info.value)

    def test_missing_huggingface_integration_raises_not_found(self, tmp_path: Path) -> None:
        from src.config.integrations.huggingface import HuggingFaceHubConfig

        hf_ref = HuggingFaceHubConfig(integration="missing-hf", repo_id="user/repo")
        cfg = _mk_pipeline_config(hf_ref=hf_ref)
        registry = IntegrationRegistry(root=tmp_path)

        with pytest.raises(IntegrationNotFoundError) as exc_info:
            resolve_pipeline_config(cfg, registry=registry)

        assert exc_info.value.integration_type == "huggingface"


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_no_experiment_tracking_block_passthrough(self, tmp_path: Path) -> None:
        # cfg has no experiment_tracking → resolver is a no-op.
        class _Cfg:
            experiment_tracking = None

        cfg = _Cfg()
        registry = IntegrationRegistry(root=tmp_path)

        resolve_pipeline_config(cfg, registry=registry)
        assert cfg.experiment_tracking is None

    def test_mlflow_ref_with_no_integration_passthrough(self, tmp_path: Path) -> None:
        # integration=None → MLflow tracking disabled, ref stays as ref.
        ref = MLflowTrackingRef(integration=None, experiment_name=None)
        cfg = _mk_pipeline_config(mlflow_ref=ref)
        registry = IntegrationRegistry(root=tmp_path)

        resolve_pipeline_config(cfg, registry=registry)
        assert cfg.experiment_tracking.mlflow is ref  # untouched

    def test_optional_fields_null_in_integration_yaml(self, tmp_path: Path) -> None:
        # tracking_uri only, everything else null.
        minimal_yaml = 'tracking_uri: "https://example.com"\n'
        _create_integration(
            tmp_path,
            integration_id="minimal",
            integration_type="mlflow",
            current_yaml=minimal_yaml,
        )
        ref = MLflowTrackingRef(integration="minimal", experiment_name="x")
        cfg = _mk_pipeline_config(mlflow_ref=ref)
        registry = IntegrationRegistry(root=tmp_path)

        resolve_pipeline_config(cfg, registry=registry)
        resolved = cfg.experiment_tracking.mlflow
        assert isinstance(resolved, MLflowConfig)
        assert resolved.tracking_uri == "https://example.com"
        assert resolved.local_tracking_uri is None
        assert resolved.ca_bundle_path is None

    def test_yaml_top_level_not_a_mapping(self, tmp_path: Path) -> None:
        # current.yaml is a list — invalid for MLflowIntegrationConfig.
        _create_integration(
            tmp_path,
            integration_id="weird",
            integration_type="mlflow",
            current_yaml="- a\n- b\n",
        )
        ref = MLflowTrackingRef(integration="weird", experiment_name="x")
        cfg = _mk_pipeline_config(mlflow_ref=ref)
        registry = IntegrationRegistry(root=tmp_path)

        with pytest.raises(IntegrationUnresolvedError) as exc_info:
            resolve_pipeline_config(cfg, registry=registry)
        assert "must be a mapping" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_resolver_idempotent(self, tmp_path: Path) -> None:
        # Resolving an already-resolved config should be a no-op.
        _create_integration(
            tmp_path,
            integration_id="my-mlflow",
            integration_type="mlflow",
            current_yaml=_MLFLOW_INTEGRATION_YAML,
        )
        ref = MLflowTrackingRef(integration="my-mlflow", experiment_name="x")
        cfg = _mk_pipeline_config(mlflow_ref=ref)
        registry = IntegrationRegistry(root=tmp_path)

        resolve_pipeline_config(cfg, registry=registry)
        first = cfg.experiment_tracking.mlflow
        assert isinstance(first, MLflowConfig)

        # Second pass — already resolved, should not change.
        resolve_pipeline_config(cfg, registry=registry)
        second = cfg.experiment_tracking.mlflow
        assert second is first

    def test_resolver_returns_same_instance(self, tmp_path: Path) -> None:
        cfg = _mk_pipeline_config()
        registry = IntegrationRegistry(root=tmp_path)
        result = resolve_pipeline_config(cfg, registry=registry)
        assert result is cfg

    def test_resolved_structurally_matches_manual_merge(self, tmp_path: Path) -> None:
        _create_integration(
            tmp_path,
            integration_id="my-mlflow",
            integration_type="mlflow",
            current_yaml=_MLFLOW_INTEGRATION_YAML,
        )
        ref = MLflowTrackingRef(integration="my-mlflow", experiment_name="my-exp")
        cfg = _mk_pipeline_config(mlflow_ref=ref)
        registry = IntegrationRegistry(root=tmp_path)
        resolve_pipeline_config(cfg, registry=registry)

        # Manual reference: what the merged config SHOULD look like.
        expected = MLflowConfig(
            tracking_uri="https://mlflow.example.com",
            local_tracking_uri="http://localhost:5002",
            ca_bundle_path=None,
            experiment_name="my-exp",
            run_description_file=None,
            system_metrics=SystemMetricsConfig(callback_enabled=True),
        )
        actual = cfg.experiment_tracking.mlflow
        assert isinstance(actual, MLflowConfig)
        assert actual.model_dump() == expected.model_dump()


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_unreadable_current_yaml_surfaces(self, tmp_path: Path, monkeypatch) -> None:
        _create_integration(
            tmp_path,
            integration_id="my-mlflow",
            integration_type="mlflow",
            current_yaml=_MLFLOW_INTEGRATION_YAML,
        )
        # Replace current.yaml read with permission denied
        ref = MLflowTrackingRef(integration="my-mlflow", experiment_name="x")
        cfg = _mk_pipeline_config(mlflow_ref=ref)
        registry = IntegrationRegistry(root=tmp_path)

        # Make the file unreadable by patching IntegrationStore.current_yaml_text
        from src.config.integrations import resolver as resolver_mod

        original = resolver_mod.IntegrationStore.current_yaml_text

        def boom(self) -> str:
            raise OSError("permission denied")

        monkeypatch.setattr(resolver_mod.IntegrationStore, "current_yaml_text", boom)

        # IOError is a hard dependency failure — surfaces directly (not wrapped),
        # so caller can distinguish "infrastructure problem" from "config problem".
        with pytest.raises(OSError, match="permission denied"):
            resolve_pipeline_config(cfg, registry=registry)


# ---------------------------------------------------------------------------
# 6. Regression
# ---------------------------------------------------------------------------


class TestRegression:
    def test_legacy_inline_tracking_uri_still_rejected(self) -> None:
        # The migration hint in ExperimentTrackingConfig._reject_legacy_keys
        # must keep rejecting old-format YAMLs even after the resolver lands.
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ExperimentTrackingConfig.model_validate(
                {
                    "mlflow": {
                        "tracking_uri": "https://example.com",
                        "experiment_name": "x",
                    }
                }
            )
        assert "tracking_uri" in str(exc_info.value)
        assert "Settings → Integrations" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_project_fields_win_over_integration_fields(self, tmp_path: Path) -> None:
        # Pin: ``experiment_name`` and ``run_description_file`` come from the
        # project ref, not the integration. (Integration doesn't have them
        # by design — project fields are project-local.)
        _create_integration(
            tmp_path,
            integration_id="shared",
            integration_type="mlflow",
            current_yaml=_MLFLOW_INTEGRATION_YAML,
        )
        ref_a = MLflowTrackingRef(integration="shared", experiment_name="project-A")
        ref_b = MLflowTrackingRef(integration="shared", experiment_name="project-B")
        registry = IntegrationRegistry(root=tmp_path)

        cfg_a = _mk_pipeline_config(mlflow_ref=ref_a)
        cfg_b = _mk_pipeline_config(mlflow_ref=ref_b)
        resolve_pipeline_config(cfg_a, registry=registry)
        resolve_pipeline_config(cfg_b, registry=registry)

        # Same tracking_uri, but different experiment_name.
        assert cfg_a.experiment_tracking.mlflow.tracking_uri == "https://mlflow.example.com"
        assert cfg_b.experiment_tracking.mlflow.tracking_uri == "https://mlflow.example.com"
        assert cfg_a.experiment_tracking.mlflow.experiment_name == "project-A"
        assert cfg_b.experiment_tracking.mlflow.experiment_name == "project-B"

    def test_resolver_does_not_consult_unrelated_blocks(self, tmp_path: Path) -> None:
        # Pin: resolver only touches ``experiment_tracking``. If user had
        # something custom on the cfg object outside that, it stays put.
        _create_integration(
            tmp_path,
            integration_id="my-mlflow",
            integration_type="mlflow",
            current_yaml=_MLFLOW_INTEGRATION_YAML,
        )
        ref = MLflowTrackingRef(integration="my-mlflow", experiment_name="x")
        cfg = _mk_pipeline_config(mlflow_ref=ref)
        cfg.unrelated_attribute = "preserved"  # type: ignore[attr-defined]
        registry = IntegrationRegistry(root=tmp_path)

        resolve_pipeline_config(cfg, registry=registry)

        assert cfg.unrelated_attribute == "preserved"  # type: ignore[attr-defined]
