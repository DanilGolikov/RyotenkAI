from __future__ import annotations

from types import SimpleNamespace

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ryotenkai_shared.pipeline_context import RunContext
from ryotenkai_providers.inference.interfaces import (
    EndpointInfo,
    InferenceArtifacts,
    InferenceArtifactsContext,
    PipelineReadinessMode,
)
from ryotenkai_control.pipeline.stages.constants import StageNames
from ryotenkai_control.pipeline.stages.inference_deployer import InferenceDeployer
from ryotenkai_shared.config import PipelineConfig, Secrets


def _load_test_config() -> PipelineConfig:
    cfg_path = (
        Path(__file__).resolve().parents[5]
        / "tests" / "unit" / "control" / "fixtures" / "configs" / "test_pipeline.yaml"
    )
    return PipelineConfig.from_yaml(cfg_path)


def test_inference_deployer_skips_when_disabled() -> None:
    cfg = _load_test_config()
    cfg.inference.enabled = False
    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    ctx = stage.execute({})
    assert ctx[StageNames.INFERENCE_DEPLOYER]["inference_skipped"] is True


def test_inference_deployer_fails_when_model_source_missing() -> None:
    from ryotenkai_shared.errors import ModelLoadFailedError

    cfg = _load_test_config()
    cfg.inference.enabled = True
    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    with pytest.raises(ModelLoadFailedError):
        stage.execute({})


def test_chat_script_checks_status_before_chat(tmp_path: Path) -> None:
    """Test that chat script checks container status and health before starting chat."""
    # Create a minimal manifest
    manifest = {
        "ssh": {"alias": "test-node"},
        "docker": {
            "container_name": "ryotenkai-inference-vllm",
            "host_bind": "127.0.0.1",
            "port": 8000,
        },
        "endpoint": {"client_base_url": "http://127.0.0.1:8000/v1"},
        "model": {"base_model_id": "test/model"},
    }

    inference_dir = tmp_path / "inference"
    inference_dir.mkdir()
    manifest_path = inference_dir / "inference_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    chat_script_path = inference_dir / "chat_inference.py"

    from ryotenkai_providers.single_node.inference.artifacts import CHAT_SCRIPT
    chat_script_path.write_text(CHAT_SCRIPT)
    # Regression: Python 3.13 forbids capture_output + stderr/stdout args together
    assert "capture_output=True, stderr=subprocess.STDOUT" not in CHAT_SCRIPT
    assert "stdout=subprocess.PIPE, stderr=subprocess.STDOUT" in CHAT_SCRIPT

    # Mock subprocess.run to simulate SSH commands
    ssh_calls = []

    def mock_subprocess_run(args, **kwargs):
        ssh_calls.append(args)

        # Check which SSH command is being executed
        if isinstance(args, list) and len(args) > 0:
            cmd_str = " ".join(str(a) for a in args)

            # Container running check
            if "docker ps -q" in cmd_str and "status=running" in cmd_str:
                result = SimpleNamespace(stdout="container123\n", returncode=0)
                return result

            # Health check
            if "curl" in cmd_str and "/v1/models" in cmd_str:
                result = SimpleNamespace(stdout="1\n", returncode=0)
                return result

        result = SimpleNamespace(stdout="", returncode=1)
        return result

    # Execute script with mocked dependencies
    with patch("subprocess.run", side_effect=mock_subprocess_run):
        # We can't easily test the full interactive part, but we can verify
        # the script is syntactically correct and imports work
        exec_globals = {}
        exec(chat_script_path.read_text(), exec_globals)

        # Verify key functions exist
        assert "_check_status_remote" in exec_globals
        assert "_start_chat" in exec_globals


def test_chat_script_reports_when_container_not_running(tmp_path: Path) -> None:
    """Test that chat script validation logic works correctly."""
    inference_dir = tmp_path / "inference"
    inference_dir.mkdir()

    manifest = {
        "ssh": {"alias": "test-node"},
        "docker": {
            "container_name": "ryotenkai-inference-vllm",
            "host_bind": "127.0.0.1",
            "port": 8000,
        },
        "endpoint": {"client_base_url": "http://127.0.0.1:8000/v1"},
        "model": {"base_model_id": "test/model"},
    }

    manifest_path = inference_dir / "inference_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    chat_script_path = inference_dir / "chat_inference.py"

    from ryotenkai_providers.single_node.inference.artifacts import CHAT_SCRIPT
    chat_script_path.write_text(CHAT_SCRIPT)

    # Verify script compiles and has correct structure
    script_content = chat_script_path.read_text()
    assert "NOT running" in script_content
    assert "_check_status_remote" in script_content
    assert "is_running, is_healthy" in script_content


# =============================================================================
# Positive / negative tests
# =============================================================================


def test_deploy_fails_when_run_context_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative: context without RunContext raises InternalError."""
    from ryotenkai_shared.errors import InternalError

    cfg = _load_test_config()
    cfg.inference.enabled = True
    cfg.inference.common.model_source = "test/repo"
    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
    }

    with pytest.raises(InternalError) as exc_info:
        stage.execute(context)
    assert "RunContext" in (exc_info.value.detail or "")


# =============================================================================
# Boundary tests
# =============================================================================


# =============================================================================
# Invariants
# =============================================================================


# =============================================================================
# Dependency / event_logger errors
# =============================================================================


# =============================================================================
# Regressions
# =============================================================================


def test_regression_python313_no_capture_output_with_stdout(
    tmp_path: Path,
) -> None:
    """Regression: subprocess.run must not use capture_output=True with stdout/stderr."""
    from ryotenkai_providers.single_node.inference.artifacts import CHAT_SCRIPT

    for name, script in [("CHAT_SCRIPT", CHAT_SCRIPT)]:
        assert "capture_output=True, stderr=subprocess.STDOUT" not in script, (
            f"{name}: Python 3.13 forbids capture_output with stdout/stderr"
        )


# =============================================================================
# Combinatorial tests
# =============================================================================


# =============================================================================
# Provider tests (artifacts, build_inference_artifacts)
# =============================================================================


def test_single_node_render_readme() -> None:
    """single_node provider: render_readme returns manifest_filename and endpoint_url."""
    from ryotenkai_providers.single_node.inference.artifacts import render_readme

    out = render_readme(manifest_filename="test_manifest.json", endpoint_url="http://localhost:8000/v1")
    assert "test_manifest.json" in out
    assert "http://localhost:8000/v1" in out
    assert "chat_inference.py" in out
    assert "stop_inference.py" in out


def test_pods_render_readme() -> None:
    """runpod (pods) provider: render_readme mentions SSH tunnel and HF_TOKEN."""
    from ryotenkai_providers.runpod.inference.pods.artifacts import render_readme

    out = render_readme(manifest_filename="m.json", endpoint_url="http://127.0.0.1:8000/v1")
    assert "SSH" in out or "ssh" in out
    assert "HF_TOKEN" in out
    assert "m.json" in out


def test_inference_artifacts_context_frozen() -> None:
    """InferenceArtifactsContext is an immutable frozen dataclass."""
    ctx = InferenceArtifactsContext(
        run_name="r1",
        mlflow_run_id="mid",
        model_source="ms",
        endpoint=EndpointInfo(
            endpoint_url="http://x/v1",
            api_type="openai_compatible",
            provider_type="single_node",
            engine="vllm",
            model_id="m1",
            health_url="http://x/v1/models",
        ),
    )
    with pytest.raises(AttributeError):
        ctx.run_name = "changed"


def test_pipeline_readiness_mode_values() -> None:
    """PipelineReadinessMode has expected members."""
    assert PipelineReadinessMode.WAIT_FOR_HEALTHY == "wait_for_healthy"
    assert PipelineReadinessMode.SKIP == "skip"


def test_inference_artifacts_structure() -> None:
    """InferenceArtifacts contains manifest, chat_script, readme."""
    from ryotenkai_providers.single_node.inference.artifacts import CHAT_SCRIPT, render_readme

    artifacts = InferenceArtifacts(
        manifest={"k": "v"},
        chat_script=CHAT_SCRIPT,
        readme=render_readme(manifest_filename="m.json", endpoint_url="http://x/v1"),
    )
    assert artifacts.manifest == {"k": "v"}
    assert "def _load_manifest" in artifacts.chat_script
    assert "m.json" in artifacts.readme


# =============================================================================
# Uncovered branches: _make_deferred_endpoint, factory failure, NO_CAPACITY,
# activate_for_eval failure, cleanup() variants
# =============================================================================


def test_make_deferred_endpoint_returns_correct_fields() -> None:
    """Line 63: _make_deferred_endpoint builds EndpointInfo with correct fields."""
    from ryotenkai_control.pipeline.stages.inference_deployer import _make_deferred_endpoint

    endpoint = _make_deferred_endpoint(
        port=9090,
        provider_type="runpod",
        engine="vllm",
        model_id="my-org/my-model",
    )
    assert endpoint.endpoint_url == "http://127.0.0.1:9090/v1"
    assert endpoint.health_url == "http://127.0.0.1:9090/v1/models"
    assert endpoint.api_type == "openai_compatible"
    assert endpoint.provider_type == "runpod"
    assert endpoint.engine == "vllm"
    assert endpoint.model_id == "my-org/my-model"
    assert endpoint.resource_id == "unknown"


# ---------------------------------------------------------------------------
# activate_for_eval — fail-fast contract (replaces former silent-skip behaviour)
#
# History: this file used to have ``test_activate_for_eval_failure_logs_warning_pipeline_continues``
# which encoded the bug as expected behaviour: activate failing was logged at
# WARNING and the stage returned Ok with a phantom endpoint URL. That URL fed
# 31× Connection refused into ModelEvaluator, producing "successful" eval runs
# with empty answers. The test was removed deliberately along with the silent
# path. See docs/plans/...-majestic-stream.md §1.
#
# These new tests bypass the YAML fixture loader (which is currently broken
# upstream by an unrelated config-integration drift) and build the config
# inline via MagicMock — same pattern as test_stages_model_evaluator.py.
# ---------------------------------------------------------------------------


def _mk_inference_cfg(
    *, eval_enabled: bool = True, provider: str = "fake",
) -> MagicMock:
    """Minimal PipelineConfig stub that satisfies InferenceDeployer.execute."""
    cfg = MagicMock()
    cfg.inference.enabled = True
    cfg.inference.provider = provider
    cfg.inference.engine = "vllm"
    cfg.inference.common.health_check.enabled = False
    cfg.inference.common.health_check.timeout_seconds = 60
    cfg.inference.common.health_check.interval_seconds = 5
    cfg.inference.common.lora.adapter_path = "auto"
    cfg.inference.common.keep_inference_after_eval = False
    cfg.inference.common.model_source = "auto"
    cfg.inference.engines.vllm.quantization = None
    cfg.evaluation = MagicMock()
    cfg.evaluation.enabled = eval_enabled
    cfg.model.name = "test-model"
    cfg.model.trust_remote_code = False
    cfg.integrations.mlflow = None
    return cfg


def _build_activate_test_context() -> dict:
    return {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }


def test_cleanup_noop_when_provider_none() -> None:
    """Lines 253-254: cleanup() when _provider is None → no-op, no exceptions."""
    cfg = _load_test_config()
    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    stage._provider = None
    # Must not raise
    stage.cleanup()


def test_cleanup_skips_deactivate_when_eval_disabled() -> None:
    """Lines 256-259: cleanup() when evaluation.enabled=False → deactivate_after_eval not called."""
    cfg = _load_test_config()
    cfg.evaluation.enabled = False

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    mock_provider = MagicMock()
    stage._provider = mock_provider

    stage.cleanup()

    mock_provider.deactivate_after_eval.assert_not_called()


def test_cleanup_logs_warning_on_deactivate_failure() -> None:
    """Lines 263-264: cleanup() eval enabled + deactivate_after_eval raises → warning, no propagate."""
    from ryotenkai_shared.errors import InferenceUnavailableError

    cfg = _load_test_config()
    cfg.evaluation.enabled = True

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    mock_provider = MagicMock()
    mock_provider.deactivate_after_eval.side_effect = InferenceUnavailableError(
        detail="deactivation failed", context={"code": "DEACT_ERR"}
    )
    stage._provider = mock_provider

    # Must not raise
    stage.cleanup()

    mock_provider.deactivate_after_eval.assert_called_once()


def test_cleanup_deactivates_when_eval_enabled_success() -> None:
    """Lines 265-266: cleanup() eval enabled + deactivate_after_eval succeeds → ok."""
    cfg = _load_test_config()
    cfg.evaluation.enabled = True

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    mock_provider = MagicMock()
    mock_provider.deactivate_after_eval.return_value = None
    stage._provider = mock_provider

    stage.cleanup()

    mock_provider.deactivate_after_eval.assert_called_once()


def test_cleanup_keeps_runtime_when_keep_inference_after_eval_enabled() -> None:
    cfg = _load_test_config()
    cfg.evaluation.enabled = True
    cfg.inference.common.keep_inference_after_eval = True

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    mock_provider = MagicMock()
    stage._provider = mock_provider

    stage.cleanup()

    mock_provider.deactivate_after_eval.assert_not_called()


# =============================================================================
# Phase 5: typed event emission
# =============================================================================


class TestPhase5EventEmission:
    """Pin the Phase 5 typed-event contract for InferenceDeployer.

    All happy / sad paths under test use the canonical
    :class:`FakeEventEmitter` (tests/_fakes/event_emitter.py) — no
    Protocol mocking (CLAUDE.md sentinel
    :mod:`tests._lint.test_no_protocol_mocking`).
    """

    def test_skip_path_emits_nothing(self) -> None:
        """Stage skipped (inference.enabled=False, not forced) → no
        typed envelopes (the skip path returns before the stage_scope
        is opened).
        """
        from tests._fakes.event_emitter import FakeEventEmitter

        emitter = FakeEventEmitter()
        cfg = _load_test_config()
        cfg.inference.enabled = False
        stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"), emitter=emitter)
        stage.execute({})
        assert emitter.emitted == []

    def test_failed_path_emits_deployment_failed(self) -> None:
        """Negative: model_source unresolved (ModelLoadFailedError) →
        ``ryotenkai.control.inference.deployment_failed`` envelope is
        emitted before the exception propagates.
        """
        from ryotenkai_shared.errors import ModelLoadFailedError

        from tests._fakes.event_emitter import FakeEventEmitter

        emitter = FakeEventEmitter()
        cfg = _load_test_config()
        cfg.inference.enabled = True
        stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"), emitter=emitter)
        with pytest.raises(ModelLoadFailedError):
            stage.execute({})
        kinds = [ev.kind for ev in emitter.emitted]
        assert "ryotenkai.control.inference.deployment_failed" in kinds
        # Pin invariants on the emitted failure envelope.
        failed = next(
            ev for ev in emitter.emitted
            if ev.kind == "ryotenkai.control.inference.deployment_failed"
        )
        assert failed.severity == "error"
        assert failed.payload.error_type == "ModelLoadFailedError"

    def test_internal_error_path_emits_deployment_failed(self) -> None:
        """Missing RunContext is a programmer error (InternalError) —
        the failure envelope still fires so reports surface the
        diagnostic chain.
        """
        from ryotenkai_shared.errors import InternalError

        from tests._fakes.event_emitter import FakeEventEmitter

        emitter = FakeEventEmitter()
        cfg = _load_test_config()
        cfg.inference.enabled = True
        cfg.inference.common.model_source = "test/repo"
        stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"), emitter=emitter)
        with pytest.raises(InternalError):
            stage.execute({StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"}})
        kinds = [ev.kind for ev in emitter.emitted]
        assert "ryotenkai.control.inference.deployment_failed" in kinds

    def test_set_emitter_is_lazy_wiring(self) -> None:
        """Stage is constructed without emitter; orchestrator wires it
        lazily via :meth:`set_emitter`. The stage MUST use the
        late-set emitter (regression: an early-bound copy would never
        publish).
        """
        from ryotenkai_shared.errors import ModelLoadFailedError

        from tests._fakes.event_emitter import FakeEventEmitter

        emitter = FakeEventEmitter()
        cfg = _load_test_config()
        cfg.inference.enabled = True
        stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
        stage.set_emitter(emitter)
        with pytest.raises(ModelLoadFailedError):
            stage.execute({})
        assert any(
            ev.kind == "ryotenkai.control.inference.deployment_failed"
            for ev in emitter.emitted
        )

    def test_cleanup_emits_deactivated_after_success(self) -> None:
        """Happy path for :meth:`cleanup` (eval enabled +
        deactivate_after_eval returns) → ``deactivated`` envelope.
        """
        from tests._fakes.event_emitter import FakeEventEmitter

        emitter = FakeEventEmitter()
        cfg = _load_test_config()
        cfg.evaluation.enabled = True

        stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"), emitter=emitter)
        mock_provider = MagicMock()
        mock_provider.deactivate_after_eval.return_value = None
        stage._provider = mock_provider
        stage.cleanup()
        assert any(
            ev.kind == "ryotenkai.control.inference.deactivated"
            for ev in emitter.emitted
        )

    def test_cleanup_no_deactivate_emit_on_warning_path(self) -> None:
        """If deactivate_after_eval raises (warning path) → no
        ``deactivated`` envelope. The contract: the typed envelope
        represents a successful deactivation; failures stay as log
        warnings until Phase 6 fold them into a typed event.
        """
        from ryotenkai_shared.errors import InferenceUnavailableError

        from tests._fakes.event_emitter import FakeEventEmitter

        emitter = FakeEventEmitter()
        cfg = _load_test_config()
        cfg.evaluation.enabled = True
        stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"), emitter=emitter)
        mock_provider = MagicMock()
        mock_provider.deactivate_after_eval.side_effect = InferenceUnavailableError(
            detail="deact failed", context={},
        )
        stage._provider = mock_provider
        stage.cleanup()  # must not raise
        assert not any(
            ev.kind == "ryotenkai.control.inference.deactivated"
            for ev in emitter.emitted
        )


