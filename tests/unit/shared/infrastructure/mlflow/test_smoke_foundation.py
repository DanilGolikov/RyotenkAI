"""Smoke tests for Phase M1 foundation modules.

Covers ``protocols``, ``run_handle``, ``config``, ``auth``, ``uri``,
``taxonomy``. Each test class focuses on one production module so the
``test_every_module_has_tests`` sentinel finds at least one reference.

These are smoke tests, not full 7-class production suites — the
foundation modules are pure value-objects / validators / enums whose
behaviour is mostly type-driven. Full coverage lands when M2 wires
them through control-plane lifecycle.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from ryotenkai_shared.config.integrations.mlflow_project import MLflowProjectConfig
from ryotenkai_shared.infrastructure.mlflow.auth import (
    MlflowAuthAdapter,
    _AuthBasic,
    _AuthBearer,
    _AuthNone,
)
from ryotenkai_shared.infrastructure.mlflow.config import MLflowConnectionConfig
from ryotenkai_shared.infrastructure.mlflow.protocols import (
    IArtifactSink,
    IJournalUploader,
    IMetricSink,
    IModelRegistry,
    IPromptRegistry,
    IRunQuery,
    ITrackingClient,
    RunStatus,
)
from ryotenkai_shared.infrastructure.mlflow.run_handle import RunHandle
from ryotenkai_shared.infrastructure.mlflow.taxonomy import (
    MetricKey,
    ParamKey,
    ReservedPrefixGuard,
    TagKey,
)
from ryotenkai_shared.infrastructure.mlflow.uri import (
    RuntimeUri,
    RuntimeUriResolver,
)


class TestRunHandle:
    def test_constructs(self) -> None:
        h = RunHandle(
            run_id="r1",
            experiment_id="e1",
            parent_run_id=None,
            tracking_uri="http://x",
            status=RunStatus.RUNNING,
        )
        assert h.run_id == "r1"
        assert h.status == RunStatus.RUNNING

    def test_is_hashable_by_run_id(self) -> None:
        a = RunHandle("r1", "e1", None, "u", RunStatus.RUNNING)
        b = RunHandle("r1", "e2", "p", "v", RunStatus.FAILED)
        assert hash(a) == hash(b)


class TestRunStatus:
    def test_terminal_values_pinned(self) -> None:
        assert RunStatus.RUNNING.value == "RUNNING"
        assert RunStatus.FINISHED.value == "FINISHED"
        assert RunStatus.FAILED.value == "FAILED"
        assert RunStatus.KILLED.value == "KILLED"


class TestProtocolsAreRuntimeCheckable:
    @pytest.mark.parametrize(
        "proto",
        [
            ITrackingClient,
            IMetricSink,
            IArtifactSink,
            IRunQuery,
            IModelRegistry,
            IJournalUploader,
            IPromptRegistry,
        ],
    )
    def test_runtime_checkable(self, proto: type) -> None:
        assert hasattr(proto, "__protocol_attrs__") or hasattr(
            proto,
            "_is_runtime_protocol",
        )


class TestMLflowConnectionConfig:
    def test_requires_at_least_one_uri(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            MLflowConnectionConfig()  # type: ignore[call-arg]

    def test_rejects_userinfo_in_uri(self) -> None:
        with pytest.raises(ValueError, match="userinfo"):
            MLflowConnectionConfig(tracking_uri="http://user:pass@host:5000")

    def test_accepts_local_only(self) -> None:
        cfg = MLflowConnectionConfig(local_tracking_uri="http://localhost:5000")
        assert cfg.local_tracking_uri == "http://localhost:5000"
        assert cfg.tracking_uri is None

    def test_normalizes_blank(self) -> None:
        cfg = MLflowConnectionConfig(
            local_tracking_uri="  http://x:5000  ",
            tracking_uri="",
        )
        assert cfg.local_tracking_uri == "http://x:5000"
        assert cfg.tracking_uri is None


class TestMLflowProjectConfig:
    def test_requires_experiment_pattern(self) -> None:
        with pytest.raises(ValueError, match="env__team__purpose"):
            MLflowProjectConfig(
                local_tracking_uri="http://x",
                experiment_name="single_segment",
            )

    def test_rejects_uppercase(self) -> None:
        with pytest.raises(ValueError):
            MLflowProjectConfig(
                local_tracking_uri="http://x",
                experiment_name="DEV__alignment__smoke",
            )

    def test_accepts_three_segments(self) -> None:
        cfg = MLflowProjectConfig(
            local_tracking_uri="http://x",
            experiment_name="dev__alignment__sft_smoke",
        )
        assert cfg.experiment_name == "dev__alignment__sft_smoke"
        assert cfg.alias_on_success == "challenger"

    def test_template_requires_placeholders(self) -> None:
        with pytest.raises(ValueError, match="placeholders"):
            MLflowProjectConfig(
                local_tracking_uri="http://x",
                experiment_name="a__b__c",
                model_registry_name_template="no-template",
            )


class TestRuntimeUriResolver:
    def test_control_plane_prefers_local(self) -> None:
        cfg = MLflowConnectionConfig(
            tracking_uri="https://public.example.com",
            local_tracking_uri="http://localhost:5000",
        )
        uri = RuntimeUriResolver.for_control_plane(cfg)
        assert uri.uri == "http://localhost:5000"
        assert uri.role == "control_plane"

    def test_control_plane_falls_back_to_tracking_uri(self) -> None:
        cfg = MLflowConnectionConfig(tracking_uri="https://public.example.com")
        uri = RuntimeUriResolver.for_control_plane(cfg)
        assert uri.uri == "https://public.example.com"

    def test_training_env_override_wins(self) -> None:
        cfg = MLflowConnectionConfig(tracking_uri="https://public.example.com")
        uri = RuntimeUriResolver.for_training(
            cfg,
            env_override="https://override.example.com",
        )
        assert uri.uri == "https://override.example.com"
        assert uri.role == "training"

    def test_training_reads_env_var_when_no_override(self) -> None:
        cfg = MLflowConnectionConfig(tracking_uri="https://public.example.com")
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "https://env.example.com"}):
            uri = RuntimeUriResolver.for_training(cfg, env_override=None)
        assert uri.uri == "https://env.example.com"

    def test_training_falls_back_to_tracking_uri(self) -> None:
        cfg = MLflowConnectionConfig(tracking_uri="https://public.example.com")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MLFLOW_TRACKING_URI", None)
            uri = RuntimeUriResolver.for_training(cfg, env_override=None)
        assert uri.uri == "https://public.example.com"

    def test_runtime_uri_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            RuntimeUri(uri="", role="control_plane")


class TestReservedPrefixGuard:
    def test_accepts_ryotenkai_ns(self) -> None:
        ReservedPrefixGuard.assert_safe("ryotenkai.lineage.run_id")

    @pytest.mark.parametrize(
        "key",
        [
            "mlflow.parentRunId",
            "mlflow.runName",
            "mlflow.note.content",
            "mlflow.user",
            "mlflow.source.git.commit",
            "mlflow.source.name",
        ],
    )
    def test_accepts_mlflow_whitelist(self, key: str) -> None:
        ReservedPrefixGuard.assert_safe(key)

    def test_rejects_unknown_mlflow(self) -> None:
        with pytest.raises(ValueError, match="reserved 'mlflow.'"):
            ReservedPrefixGuard.assert_safe("mlflow.custom.field")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ReservedPrefixGuard.assert_safe("")

    def test_accepts_hf_ns(self) -> None:
        ReservedPrefixGuard.assert_safe("hf.some.metric")


class TestTaxonomyEnums:
    def test_tag_keys_all_under_ryotenkai_ns(self) -> None:
        for member in TagKey:
            assert member.value.startswith("ryotenkai."), member.value

    def test_param_keys_all_under_ryotenkai_ns(self) -> None:
        for member in ParamKey:
            assert member.value.startswith("ryotenkai."), member.value

    def test_metric_keys_all_under_ryotenkai_ns(self) -> None:
        for member in MetricKey:
            assert member.value.startswith("ryotenkai."), member.value


class TestAuthDiscriminator:
    def test_none_default(self) -> None:
        adapter = MlflowAuthAdapter(_AuthNone())
        assert adapter.authorization_header() is None

    def test_basic_reads_env(self) -> None:
        cfg = _AuthBasic(kind="basic", username="u", password_env_var="MLFLOW_PWD_TEST")
        with patch.dict(os.environ, {"MLFLOW_PWD_TEST": "secret"}):
            header = MlflowAuthAdapter(cfg).authorization_header()
        assert header is not None
        assert header.startswith("Basic ")

    def test_bearer_reads_env(self) -> None:
        cfg = _AuthBearer(kind="bearer", token_env_var="MLFLOW_BEARER_TEST")
        with patch.dict(os.environ, {"MLFLOW_BEARER_TEST": "tok-123"}):
            header = MlflowAuthAdapter(cfg).authorization_header()
        assert header == "Bearer tok-123"

    def test_basic_missing_env_raises(self) -> None:
        cfg = _AuthBasic(kind="basic", username="u", password_env_var="MISSING_VAR_XYZ")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MISSING_VAR_XYZ", None)
            with pytest.raises(KeyError, match="MISSING_VAR_XYZ"):
                MlflowAuthAdapter(cfg).authorization_header()

    def test_bearer_empty_env_raises(self) -> None:
        cfg = _AuthBearer(kind="bearer", token_env_var="EMPTY_TOK_TEST")
        with patch.dict(os.environ, {"EMPTY_TOK_TEST": ""}):
            with pytest.raises(KeyError, match="empty"):
                MlflowAuthAdapter(cfg).authorization_header()
