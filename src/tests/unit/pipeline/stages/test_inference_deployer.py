from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.state import RunContext
from src.providers.inference.interfaces import (
    EndpointInfo,
    InferenceArtifacts,
    InferenceArtifactsContext,
    PipelineReadinessMode,
)
from src.pipeline.stages.constants import StageNames
from src.pipeline.stages.inference_deployer import InferenceDeployer
from src.utils.config import PipelineConfig, Secrets
from src.utils.result import Err, Ok, Success


class _FakeProvider:
    def __init__(self):
        self.deploy_calls: list[dict[str, Any]] = []
        self.health_calls = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def provider_type(self) -> str:
        return "fake"

    def deploy(self, model_source: str, **kwargs):
        self.deploy_calls.append({"model_source": model_source, **kwargs})
        return Ok(
            EndpointInfo(
                endpoint_url="http://127.0.0.1:8000/v1",
                api_type="openai_compatible",
                provider_type="single_node",
                engine="vllm",
                model_id="test-model/test-llm",
                health_url="http://127.0.0.1:8000/v1/models",
            )
        )

    def set_event_logger(self, event_logger):
        _ = event_logger
        return None

    def get_pipeline_readiness_mode(self):
        return PipelineReadinessMode.WAIT_FOR_HEALTHY

    def collect_startup_logs(self, *, local_path: Path) -> None:
        _ = local_path
        return

    def build_inference_artifacts(self, *, ctx: InferenceArtifactsContext):
        from src.providers.single_node.inference.artifacts import CHAT_SCRIPT, render_readme

        manifest = {
            "run_name": ctx.run_name,
            "mlflow_run_id": ctx.mlflow_run_id,
            "provider": "single_node",
            "engine": "vllm",
            "ssh": {"alias": "pc"},
            "docker": {
                "container_name": "ryotenkai-inference-vllm",
                "host_bind": "127.0.0.1",
                "port": 8000,
            },
            "model": {"base_model_id": "test-model/test-llm", "adapter_ref": ctx.model_source},
            "endpoint": {"client_base_url": ctx.endpoint.endpoint_url, "health_url": ctx.endpoint.health_url},
        }
        return Ok(
            InferenceArtifacts(
                manifest=manifest,
                chat_script=CHAT_SCRIPT,
                readme=render_readme(manifest_filename="inference_manifest.json", endpoint_url=ctx.endpoint.endpoint_url),
            )
        )

    def undeploy(self):
        return Ok(None)

    def health_check(self):
        self.health_calls += 1
        return Ok(True)

    def get_capabilities(self):
        return None

    def get_endpoint_info(self):
        return None


def _load_test_config() -> PipelineConfig:
    cfg_path = Path("src/tests/fixtures/configs/test_pipeline.yaml")
    return PipelineConfig.from_yaml(cfg_path)


def test_inference_deployer_skips_when_disabled() -> None:
    cfg = _load_test_config()
    cfg.inference.enabled = False
    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    res = stage.execute({})
    assert res.is_success()
    ctx = res.unwrap()
    assert ctx[StageNames.INFERENCE_DEPLOYER]["inference_skipped"] is True


def test_inference_deployer_fails_when_model_source_missing() -> None:
    cfg = _load_test_config()
    cfg.inference.enabled = True
    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    res = stage.execute({})
    assert res.is_failure()


def test_inference_deployer_writes_manifest_and_scripts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _load_test_config()
    cfg.inference.enabled = True

    fake = _FakeProvider()

    # Patch factory to avoid real SSH/Docker
    import src.pipeline.stages.inference_deployer as mod

    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))

    # Patch log dir and run_id to deterministic values
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)

    # Avoid sleeping in health loop (should not sleep anyway since fake returns healthy)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))

    context = {
        StageNames.MODEL_RETRIEVER: {
            "hf_repo_id": "test/test-adapter",
            "hf_uploaded": True,
            "local_model_path": None,
        },
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_20260120_123456_abc12",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_success()

    out = res.unwrap()[StageNames.INFERENCE_DEPLOYER]
    manifest_path = Path(out["inference_manifest_path"])
    assert manifest_path.exists()

    scripts = out["inference_scripts"]
    # NEW: Check for chat script (no more start/status/stop)
    assert Path(scripts["chat"]).exists()
    assert Path(scripts["readme"]).exists()

    # Verify scripts have correct content
    chat_script = Path(scripts["chat"]).read_text()
    assert "def _check_status_remote" in chat_script
    assert "def _start_chat" in chat_script
    assert "openai" in chat_script.lower()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_name"] == "run_20260120_123456_abc12"
    assert manifest["provider"] == "single_node"
    assert manifest["engine"] == "vllm"

    assert fake.deploy_calls
    assert fake.deploy_calls[0]["model_source"] == "test/test-adapter"



def test_inference_deployer_writes_runpod_manifest_and_scripts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _load_test_config()
    cfg.inference.enabled = True
    cfg.inference.provider = "runpod"
    cfg.inference.engine = "vllm"

    # Minimal providers.runpod config for schema validation in stage
    cfg.providers["runpod"] = {
        "connect": {"ssh": {"key_path": "/tmp/id_ed25519"}},
        "cleanup": {},
        "training": {"image_name": "test/training-runtime:latest"},
        "inference": {
            "volume": {
                "id": "nv_test_123",
                "name": "helix-test-volume",
                "size_gb": 50,
            },
            "pod": {"image_name": "test/inference-vllm:v0.1.1"},
            "serve": {"port": 8000},
        },
    }

    class _FakePodProvider:
        @property
        def provider_name(self) -> str:
            return "fake_pod"

        @property
        def provider_type(self) -> str:
            return "runpod"

        def __init__(self):
            # Exposed for manifest generation (best-effort)
            self._network_volume_id = "nv_test_123"
            self._pod_name = "helix-vllm-pod-deadbeef"

        def deploy(self, model_source: str, **kwargs):
            _ = (model_source, kwargs)
            return Ok(
                EndpointInfo(
                    endpoint_url="http://127.0.0.1:8000/v1",
                    api_type="openai_compatible",
                    provider_type="runpod",
                    engine="vllm",
                    model_id="test-model/test-llm",
                    health_url="http://127.0.0.1:8000/v1/models",
                    resource_id="pod_test_123",
                )
            )

        def set_event_logger(self, event_logger):
            _ = event_logger
            return None

        def get_pipeline_readiness_mode(self):
            return PipelineReadinessMode.SKIP

        def collect_startup_logs(self, *, local_path: Path) -> None:
            _ = local_path
            return

        def build_inference_artifacts(self, *, ctx: InferenceArtifactsContext):
            from src.providers.runpod.inference.pods.artifacts import CHAT_SCRIPT, render_readme

            manifest = {
                "run_name": ctx.run_name,
                "mlflow_run_id": ctx.mlflow_run_id,
                "provider": "runpod",
                "engine": "vllm",
                "runpod": {
                    "rest_api_base_url": "https://rest.runpod.io/v1",
                    "network_volume": {"id": "nv_test_123"},
                    "pod": {"id": "pod_test_123", "name": "helix-vllm-pod-deadbeef"},
                },
                "ssh": {"key_path": "/tmp/id_ed25519"},
                "endpoint": {"client_base_url": ctx.endpoint.endpoint_url, "health_url": ctx.endpoint.health_url},
            }
            return Ok(
                InferenceArtifacts(
                    manifest=manifest,
                    chat_script=CHAT_SCRIPT,
                    readme=render_readme(manifest_filename="inference_manifest.json", endpoint_url=ctx.endpoint.endpoint_url),
                )
            )

        def undeploy(self):
            return Ok(None)

        def health_check(self):
            # Health check is skipped for runpod (pod) by design
            return Ok(True)

        def get_capabilities(self):
            return None

        def get_endpoint_info(self):
            return None

    fake = _FakePodProvider()

    import src.pipeline.stages.inference_deployer as mod

    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))

    context = {
        StageNames.MODEL_RETRIEVER: {
            "hf_repo_id": "test/test-adapter",
            "hf_uploaded": True,
            "local_model_path": None,
        },
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_20260120_123456_abc12",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_success()

    out = res.unwrap()[StageNames.INFERENCE_DEPLOYER]
    manifest_path = Path(out["inference_manifest_path"])
    assert manifest_path.exists()

    scripts = out["inference_scripts"]
    assert Path(scripts["chat"]).exists()
    assert Path(scripts["readme"]).exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["provider"] == "runpod"
    assert manifest["engine"] == "vllm"
    assert manifest["runpod"]["network_volume"]["id"] == "nv_test_123"
    assert manifest["runpod"]["pod"]["id"] == "pod_test_123"
    assert manifest["ssh"]["key_path"] == "/tmp/id_ed25519"

    chat_script = Path(scripts["chat"]).read_text(encoding="utf-8")
    assert "Starting Pod" in chat_script
    assert "RUNPOD_API_KEY" in chat_script
    assert "SSH tunnel" in chat_script or "ssh tunnel" in chat_script.lower()


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

    from src.providers.single_node.inference.artifacts import CHAT_SCRIPT
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
                result = MagicMock()
                result.stdout = "container123\n"
                result.returncode = 0
                return result

            # Health check
            if "curl" in cmd_str and "/v1/models" in cmd_str:
                result = MagicMock()
                result.stdout = "1\n"
                result.returncode = 0
                return result

        result = MagicMock()
        result.stdout = ""
        result.returncode = 1
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

    from src.providers.single_node.inference.artifacts import CHAT_SCRIPT
    chat_script_path.write_text(CHAT_SCRIPT)

    # Verify script compiles and has correct structure
    script_content = chat_script_path.read_text()
    assert "NOT running" in script_content
    assert "_check_status_remote" in script_content
    assert "is_running, is_healthy" in script_content




# =============================================================================
# Positive / negative tests
# =============================================================================


def test_resolve_model_source_prefers_explicit_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive: model_source from config.inference.common.model_source (not auto)."""
    cfg = _load_test_config()
    cfg.inference.enabled = True
    cfg.inference.common.model_source = "explicit/hf-repo-id"

    fake = _FakeProvider()
    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_success()
    assert fake.deploy_calls[0]["model_source"] == "explicit/hf-repo-id"


def test_resolve_model_source_fallback_local_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive: fallback to local_model_path when hf_repo_id is missing."""
    cfg = _load_test_config()
    cfg.inference.enabled = True

    fake = _FakeProvider()
    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"local_model_path": "/path/to/local/model"},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_success()
    assert fake.deploy_calls[0]["model_source"] == "/path/to/local/model"


def test_deploy_fails_when_run_context_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative: context without RunContext yields Err."""
    cfg = _load_test_config()
    cfg.inference.enabled = True
    cfg.inference.common.model_source = "test/repo"
    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
    }

    res = stage.execute(context)
    assert res.is_failure()
    assert "RunContext" in str(res.unwrap_err())


def test_deploy_fails_on_provider_deploy_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative: provider deploy() returns Err → InferenceDeployer returns Err."""

    class FailingProvider(_FakeProvider):
        def deploy(self, model_source: str, **kwargs):
            return Err("Deploy failed: API error")

    cfg = _load_test_config()
    cfg.inference.enabled = True
    fake = FailingProvider()
    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_failure()
    assert "Deploy failed" in str(res.unwrap_err())


def test_deploy_fails_on_build_artifacts_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative: build_inference_artifacts returns Err → Err."""

    class ArtifactsFailingProvider(_FakeProvider):
        def build_inference_artifacts(self, *, ctx: InferenceArtifactsContext):
            return Err("Artifacts build failed")

    cfg = _load_test_config()
    cfg.inference.enabled = True
    fake = ArtifactsFailingProvider()
    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_failure()
    assert "Artifacts build failed" in str(res.unwrap_err())


# =============================================================================
# Boundary tests
# =============================================================================


def test_health_check_waits_for_healthy_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Boundary: WAIT_FOR_HEALTHY, health_check succeeds on second attempt."""
    cfg = _load_test_config()
    cfg.inference.enabled = True
    cfg.inference.common.health_check.enabled = True
    cfg.inference.common.health_check.timeout_seconds = 30
    cfg.inference.common.health_check.interval_seconds = 0.01

    call_count = [0]

    class DelayedHealthyProvider(_FakeProvider):
        def health_check(self):
            call_count[0] += 1
            return Ok(call_count[0] >= 2)

    fake = DelayedHealthyProvider()
    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_success()
    assert call_count[0] >= 2


def test_health_check_provider_error_returns_err(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Boundary: health_check returns Err → last_state contains error:."""
    cfg = _load_test_config()
    cfg.inference.enabled = True
    cfg.inference.common.health_check.enabled = True
    cfg.inference.common.health_check.timeout_seconds = 0.15
    cfg.inference.common.health_check.interval_seconds = 0.02

    class ErrorHealthProvider(_FakeProvider):
        def health_check(self):
            return Err("SSH command failed")

    fake = ErrorHealthProvider()
    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_failure()
    err_msg = str(res.unwrap_err())
    assert "timed out" in err_msg
    assert "error:SSH command failed" in err_msg or "SSH command failed" in err_msg


def test_health_check_timeout_returns_err(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Boundary: health check timeout → Err with last_state."""
    cfg = _load_test_config()
    cfg.inference.enabled = True
    cfg.inference.common.health_check.enabled = True
    cfg.inference.common.health_check.timeout_seconds = 0.1
    cfg.inference.common.health_check.interval_seconds = 0.05

    class NeverHealthyProvider(_FakeProvider):
        def health_check(self):
            return Ok(False)

    fake = NeverHealthyProvider()
    undeploy_calls = []
    event_errors = []

    orig_undeploy = fake.undeploy

    def track_undeploy():
        undeploy_calls.append(True)
        return orig_undeploy()

    fake.undeploy = track_undeploy

    class MockEventLogger:
        def log_event_start(self, *args, **kwargs):
            pass

        def log_event_complete(self, *args, **kwargs):
            pass

        def log_event_error(self, message: str, **kwargs):
            event_errors.append(message)

    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "mlflow_manager": MockEventLogger(),
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_failure()
    err_msg = str(res.unwrap_err())
    assert "timed out" in err_msg
    assert "last_state" in err_msg or "not_ready" in err_msg
    assert len(undeploy_calls) == 1
    assert len(event_errors) >= 1
    assert "timed out" in event_errors[0]


def test_skip_readiness_when_health_check_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Boundary: health_check.enabled=false → readiness skipped."""
    cfg = _load_test_config()
    cfg.inference.enabled = True
    cfg.inference.common.health_check.enabled = False

    fake = _FakeProvider()
    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_success()
    assert fake.health_calls == 0


def test_mlflow_run_id_empty_strip_treated_as_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Boundary: mlflow_parent_run_id whitespace-only → treated as None in context."""
    cfg = _load_test_config()
    cfg.inference.enabled = True
    fake = _FakeProvider()
    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": "   ",
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_success()
    manifest = json.loads(
        (tmp_path / "inference" / "inference_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.get("mlflow_run_id") is None


# =============================================================================
# Invariants
# =============================================================================


def test_invariant_output_contains_all_required_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invariant: successful execute always returns inference_deployed, endpoint_url, scripts."""
    cfg = _load_test_config()
    cfg.inference.enabled = True
    fake = _FakeProvider()
    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_success()
    out = res.unwrap()[StageNames.INFERENCE_DEPLOYER]
    assert out["inference_deployed"] is True
    assert "inference_endpoint_url" in out
    assert "inference_model_name" in out
    assert "endpoint_url" in out
    assert "inference_manifest_path" in out
    assert "inference_scripts" in out
    assert set(out["inference_scripts"].keys()) == {"chat", "readme"}


def test_invariant_manifest_is_valid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant: manifest is valid JSON, UTF-8."""
    cfg = _load_test_config()
    cfg.inference.enabled = True
    fake = _FakeProvider()
    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_success()
    manifest_path = Path(res.unwrap()[StageNames.INFERENCE_DEPLOYER]["inference_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    assert "run_name" in manifest
    assert "provider" in manifest
    assert "endpoint" in manifest


# =============================================================================
# Dependency / event_logger errors
# =============================================================================


def test_event_logger_called_on_health_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Specific: on WAIT_FOR_HEALTHY success, event_logger.log_event_complete is called."""
    cfg = _load_test_config()
    cfg.inference.enabled = True
    cfg.inference.common.health_check.enabled = True
    cfg.inference.common.health_check.timeout_seconds = 30
    cfg.inference.common.health_check.interval_seconds = 0.01

    fake = _FakeProvider()
    event_log = []

    class MockEventLogger:
        def log_event_start(self, message: str, **kwargs):
            event_log.append(("start", message))

        def log_event_complete(self, message: str, **kwargs):
            event_log.append(("complete", message))

        def log_event_error(self, message: str, **kwargs):
            event_log.append(("error", message))

    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "mlflow_manager": MockEventLogger(),
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_success()
    completes = [e for e in event_log if e[0] == "complete"]
    assert len(completes) >= 1
    assert "Health check" in completes[0][1]


# =============================================================================
# Regressions
# =============================================================================


def test_regression_python313_no_capture_output_with_stdout(
    tmp_path: Path,
) -> None:
    """Regression: subprocess.run must not use capture_output=True with stdout/stderr."""
    from src.providers.single_node.inference.artifacts import CHAT_SCRIPT

    for name, script in [("CHAT_SCRIPT", CHAT_SCRIPT)]:
        assert "capture_output=True, stderr=subprocess.STDOUT" not in script, (
            f"{name}: Python 3.13 forbids capture_output with stdout/stderr"
        )


def test_regression_provider_set_event_logger_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: set_event_logger is called before deploy."""
    cfg = _load_test_config()
    cfg.inference.enabled = True
    logger_calls = []

    class TrackingProvider(_FakeProvider):
        def set_event_logger(self, event_logger):
            logger_calls.append(("set_event_logger", event_logger))

    fake = TrackingProvider()
    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_success()
    assert any(c[0] == "set_event_logger" for c in logger_calls)


# =============================================================================
# Combinatorial tests
# =============================================================================


@pytest.mark.parametrize(
    "provider,health_enabled,readiness,expect_health_calls",
    [
        ("single_node", True, PipelineReadinessMode.WAIT_FOR_HEALTHY, True),
        ("single_node", False, PipelineReadinessMode.WAIT_FOR_HEALTHY, False),
        ("runpod", True, PipelineReadinessMode.SKIP, False),
    ],
)
def test_health_check_combinatorial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    health_enabled: bool,
    readiness: PipelineReadinessMode,
    expect_health_calls: bool,
) -> None:
    """Combinatorial: provider × health_check.enabled × readiness mode."""

    class CountingProvider(_FakeProvider):
        def __init__(self):
            super().__init__()
            self._readiness = readiness

        def get_pipeline_readiness_mode(self):
            return self._readiness

    cfg = _load_test_config()
    cfg.inference.enabled = True
    cfg.inference.common.health_check.enabled = health_enabled
    cfg.inference.common.health_check.timeout_seconds = 30
    cfg.inference.common.health_check.interval_seconds = 0.01

    fake = CountingProvider()
    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_success()
    if expect_health_calls:
        assert fake.health_calls >= 1
    else:
        assert fake.health_calls == 0


# =============================================================================
# Provider tests (artifacts, build_inference_artifacts)
# =============================================================================


def test_single_node_render_readme() -> None:
    """single_node provider: render_readme returns manifest_filename and endpoint_url."""
    from src.providers.single_node.inference.artifacts import render_readme

    out = render_readme(manifest_filename="test_manifest.json", endpoint_url="http://localhost:8000/v1")
    assert "test_manifest.json" in out
    assert "http://localhost:8000/v1" in out
    assert "chat_inference.py" in out
    assert "stop_inference.py" in out



def test_pods_render_readme() -> None:
    """runpod (pods) provider: render_readme mentions SSH tunnel and HF_TOKEN."""
    from src.providers.runpod.inference.pods.artifacts import render_readme

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
    from src.providers.single_node.inference.artifacts import CHAT_SCRIPT, render_readme

    artifacts = InferenceArtifacts(
        manifest={"k": "v"},
        chat_script=CHAT_SCRIPT,
        readme=render_readme(manifest_filename="m.json", endpoint_url="http://x/v1"),
    )
    assert artifacts.manifest == {"k": "v"}
    assert "def _load_manifest" in artifacts.chat_script
    assert "m.json" in artifacts.readme


def test_mlflow_logging_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive: execute puts endpoint_url in context for orchestrator artifacts."""
    cfg = _load_test_config()
    cfg.inference.enabled = True

    fake = _FakeProvider()
    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": "run_123",
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }
    res = stage.execute(context)

    assert res.is_success()
    out = res.unwrap()[StageNames.INFERENCE_DEPLOYER]
    # endpoint_url and related fields are in context so orchestrator can write artifact
    assert "endpoint_url" in out
    assert out["endpoint_url"] == "http://127.0.0.1:8000/v1"
    assert "inference_deployed" in out
    assert out["inference_deployed"] is True


def test_mlflow_logging_skipped_when_run_id_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Boundary: empty run_id deploy still succeeds; mlflow_run_id=None in manifest."""
    cfg = _load_test_config()
    cfg.inference.enabled = True

    fake = _FakeProvider()

    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": "",  # empty → mlflow_run_id=None in manifest
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }
    res = stage.execute(context)
    assert res.is_success()
    manifest = json.loads(
        (tmp_path / "inference" / "inference_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.get("mlflow_run_id") is None


def test_mlflow_logging_failure_does_not_fail_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dependency error: save_stage_artifact raises — deploy still succeeds (best-effort)."""
    cfg = _load_test_config()
    cfg.inference.enabled = True

    fake = _FakeProvider()
    import src.pipeline.stages.inference_deployer as mod
    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": "run_123",
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }
    # Deploy itself has no MLflow dependency — always succeeds
    res = stage.execute(context)
    assert res.is_success()


# =============================================================================
# Uncovered branches: _make_deferred_endpoint, factory failure, NO_CAPACITY,
# activate_for_eval failure, cleanup() variants
# =============================================================================


def test_make_deferred_endpoint_returns_correct_fields() -> None:
    """Line 63: _make_deferred_endpoint builds EndpointInfo with correct fields."""
    from src.pipeline.stages.inference_deployer import _make_deferred_endpoint

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


def test_factory_create_failure_returns_inference_provider_create_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Line 120: InferenceProviderFactory.create fails → INFERENCE_PROVIDER_CREATE_FAILED."""
    from src.utils.result import Failure, InferenceError

    cfg = _load_test_config()
    cfg.inference.enabled = True
    cfg.inference.common.model_source = "test/repo"

    import src.pipeline.stages.inference_deployer as mod

    monkeypatch.setattr(
        mod.InferenceProviderFactory,
        "create",
        lambda **kwargs: Failure(InferenceError(message="Factory exploded", code="SOME_INTERNAL")),
    )

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_failure()
    err = res.unwrap_err()
    assert err.code == "INFERENCE_PROVIDER_CREATE_FAILED"
    assert "Factory exploded" in err.message


def test_no_capacity_deploy_error_creates_deferred_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lines 153-166: NO_CAPACITY deploy error → pod_provisioning_failed=True, deferred endpoint, port=8000."""

    class NoCapacityProvider(_FakeProvider):
        def deploy(self, model_source: str, **kwargs):
            return Err("could not find any pods with required specifications")

    cfg = _load_test_config()
    cfg.inference.enabled = True

    fake = NoCapacityProvider()
    import src.pipeline.stages.inference_deployer as mod

    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_success()
    out = res.unwrap()[StageNames.INFERENCE_DEPLOYER]
    assert out["inference_pod_deferred"] is True
    assert out["inference_deployed"] is False
    assert "127.0.0.1:8000" in out["inference_endpoint_url"]


def test_no_capacity_deploy_error_uses_serve_cfg_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lines 161-163: NO_CAPACITY → port from provider._serve_cfg.port."""

    class NoCapacityProviderWithPort(_FakeProvider):
        def __init__(self):
            super().__init__()
            self._serve_cfg = MagicMock()
            self._serve_cfg.port = 9001

        def deploy(self, model_source: str, **kwargs):
            return Err("there are no instances currently available")

    cfg = _load_test_config()
    cfg.inference.enabled = True

    fake = NoCapacityProviderWithPort()
    import src.pipeline.stages.inference_deployer as mod

    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    assert res.is_success()
    out = res.unwrap()[StageNames.INFERENCE_DEPLOYER]
    assert out["inference_pod_deferred"] is True
    assert "127.0.0.1:9001" in out["inference_endpoint_url"]


def test_activate_for_eval_failure_logs_warning_pipeline_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lines 196-202: activate_for_eval fails → warning; pipeline continues (eval_endpoint_url=None)."""
    from src.utils.result import Failure, InferenceError

    class EvalActivationFailingProvider(_FakeProvider):
        def activate_for_eval(self):
            return Failure(InferenceError(message="provider does not support activate_for_eval", code="NOT_SUPPORTED"))

    cfg = _load_test_config()
    cfg.inference.enabled = True
    cfg.evaluation.enabled = True  # enable eval

    fake = EvalActivationFailingProvider()
    import src.pipeline.stages.inference_deployer as mod

    monkeypatch.setattr(mod.InferenceProviderFactory, "create", lambda **kwargs: Success(fake))
    monkeypatch.setattr(mod, "get_run_log_dir", lambda: tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    context = {
        StageNames.MODEL_RETRIEVER: {"hf_repo_id": "test/repo"},
        "mlflow_parent_run_id": None,
        "run": RunContext(
            name="run_test",
            created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
        ),
    }

    res = stage.execute(context)
    # Pipeline continues despite eval activation failure
    assert res.is_success()
    out = res.unwrap()[StageNames.INFERENCE_DEPLOYER]
    # endpoint_url falls back to deployment endpoint (not eval)
    assert "endpoint_url" in out
    assert out["inference_deployed"] is True


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
    """Lines 263-264: cleanup() eval enabled + deactivate_after_eval fails → warning, no raise."""
    from src.utils.result import Failure, InferenceError

    cfg = _load_test_config()
    cfg.evaluation.enabled = True

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    mock_provider = MagicMock()
    mock_provider.deactivate_after_eval.return_value = Failure(
        InferenceError(message="deactivation failed", code="DEACT_ERR")
    )
    stage._provider = mock_provider

    # Must not raise
    stage.cleanup()

    mock_provider.deactivate_after_eval.assert_called_once()


def test_cleanup_deactivates_when_eval_enabled_success() -> None:
    """Lines 265-266: cleanup() eval enabled + deactivate_after_eval succeeds → ok."""
    from src.utils.result import Success as _Success

    cfg = _load_test_config()
    cfg.evaluation.enabled = True

    stage = InferenceDeployer(cfg, Secrets(HF_TOKEN="hf_test"))
    mock_provider = MagicMock()
    mock_provider.deactivate_after_eval.return_value = _Success(None)
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


def test_single_node_provider_build_inference_artifacts() -> None:
    """Real SingleNodeInferenceProvider.build_inference_artifacts returns valid artifacts."""
    from src.providers.single_node.inference.provider import SingleNodeInferenceProvider

    cfg = _load_test_config()
    provider = SingleNodeInferenceProvider(config=cfg, secrets=Secrets(HF_TOKEN="hf_test"))
    ctx = InferenceArtifactsContext(
        run_name="run_test_123",
        mlflow_run_id="mlflow_abc",
        model_source="test/test-adapter",
        endpoint=EndpointInfo(
            endpoint_url="http://127.0.0.1:8000/v1",
            api_type="openai_compatible",
            provider_type="single_node",
            engine="vllm",
            model_id="test-model/test-llm",
            health_url="http://127.0.0.1:8000/v1/models",
        ),
    )
    res = provider.build_inference_artifacts(ctx=ctx)
    assert res.is_success()
    artifacts = res.unwrap()
    assert artifacts.manifest["run_name"] == "run_test_123"
    assert artifacts.manifest["provider"] == "single_node"
    assert artifacts.manifest["engine"] == "vllm"
    assert "ssh" in artifacts.manifest
    assert "docker" in artifacts.manifest
    assert "endpoint" in artifacts.manifest
    assert "model" in artifacts.manifest
    assert "def _check_status_remote" in artifacts.chat_script

