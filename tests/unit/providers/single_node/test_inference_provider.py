"""
Unit tests for SingleNodeInferenceProvider.

Covers missing lines: 108, 129-303, 307, 310, 314-324, 343-345, 401-406, 439,
446, 501-503, 512-513, 522-583, 614-692, 725, 741-742, 766, 769
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from ryotenkai_engines.vllm.config import VLLMEngineConfig

import ryotenkai_providers.single_node.inference.provider as _mod
from ryotenkai_providers.single_node.inference.provider import SingleNodeInferenceProvider
from ryotenkai_shared.config import (
    InferenceSingleNodeServeConfig,
    Secrets,
)
from ryotenkai_shared.config.providers.single_node import (
    SingleNodeConnectConfig,
    SingleNodeInferenceConfig,
    SingleNodeProviderConfig,
    SingleNodeTrainingConfig,
)
from ryotenkai_shared.config.providers.ssh import SSHConfig
from ryotenkai_shared.utils.result import Err, InferenceError, Ok

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ssh_cfg():
    return SSHConfig(alias="test-node", host="192.168.1.100", port=22, user="testuser")


@pytest.fixture()
def provider_cfg(ssh_cfg):
    return SingleNodeProviderConfig(
        connect=SingleNodeConnectConfig(ssh=ssh_cfg),
        training=SingleNodeTrainingConfig(
            workspace_path="/home/user/workspace",
        ),
        inference=SingleNodeInferenceConfig(
            serve=InferenceSingleNodeServeConfig(
                host="127.0.0.1",
                port=8000,
                workspace="/home/user/inference",
            )
        ),
    )


@pytest.fixture()
def engine_cfg():
    return VLLMEngineConfig(
        tensor_parallel_size=1,
        max_model_len=4096,
    )


@pytest.fixture()
def secrets():
    return Secrets(hf_token="hf-test-token")


def _mk_pipeline_config(provider_cfg, engine_cfg, *, merge_before_deploy: bool = True):
    """Synthetic PipelineConfig.

    Post-discriminated-union (PR-6): ``cfg.inference.engine`` IS the typed
    engine config instance (Pydantic narrowed via ``kind``). The legacy
    ``cfg.inference.engines.vllm`` shape is gone.
    """
    # Apply merge_before_deploy override on a fresh copy so we don't mutate
    # the shared fixture instance across parametrized tests.
    eff_engine_cfg = engine_cfg.model_copy(
        update={"merge_before_deploy": merge_before_deploy}
    )
    cfg = Mock()
    cfg.get_provider_config = lambda *_a, **_k: provider_cfg.model_dump(mode="python")
    cfg.inference = Mock()
    cfg.inference.engine = eff_engine_cfg
    cfg.inference.common = Mock()
    cfg.model = Mock()
    cfg.model.name = "meta-llama/Llama-2-7b-hf"
    cfg.model.trust_remote_code = False
    cfg.integrations = None
    return cfg


@pytest.fixture(autouse=True)
def _stamp_provider_manifest_classvars():
    """In production the ProviderRegistry stamps ``_manifest_*`` ClassVars
    on the provider class before instantiation. Tests that bypass the
    registry (constructing via ``ProviderContext`` directly) miss this
    side-effect, leaving ``provider_name`` / ``provider_type`` empty.
    Stamp them ourselves once per test."""
    SingleNodeInferenceProvider._manifest_provider_id = "single_node"
    SingleNodeInferenceProvider._manifest_provider_name = "single_node"
    SingleNodeInferenceProvider._manifest_provider_type = "local"
    yield


def _mk_provider(
    provider_cfg,
    engine_cfg,
    secrets,
    *,
    merge_before_deploy: bool = True,
    docker=None,
):
    """Build the provider via :class:`ProviderContext` — the legitimate
    constructor matching production registry instantiation.

    ``docker`` is the injection seam for :class:`IDockerClient`; if not
    provided the provider falls back to its production default
    (:class:`LocalDockerClient`).
    """
    from ryotenkai_providers.registry import ProviderContext

    pipeline_cfg = _mk_pipeline_config(
        provider_cfg, engine_cfg, merge_before_deploy=merge_before_deploy
    )
    ctx = ProviderContext(
        provider_id="single_node",
        pipeline_config=pipeline_cfg,
        provider_block=provider_cfg.model_dump(mode="python"),
        secrets=secrets,
    )
    return SingleNodeInferenceProvider(ctx, docker=docker)


@pytest.fixture()
def fake_docker():
    """Canonical :class:`IDockerClient` fake — see ``tests/_fakes/docker.py``."""
    from tests._fakes.docker import FakeDockerClient

    return FakeDockerClient()


@pytest.fixture()
def provider(provider_cfg, engine_cfg, secrets, fake_docker):
    p = _mk_provider(provider_cfg, engine_cfg, secrets, docker=fake_docker)
    p._run_id = "run_test_001"
    return p


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_provider_name(self, provider):
        assert provider.provider_name == "single_node"

    def test_provider_type(self, provider):
        # ``provider_type`` is the manifest-derived capability axis
        # ("local" vs "cloud"), distinct from the provider id/name.
        assert provider.provider_type == "local"

    def test_get_pipeline_readiness_mode(self, provider):
        from ryotenkai_providers.inference.interfaces import PipelineReadinessMode
        assert provider.get_pipeline_readiness_mode() == PipelineReadinessMode.WAIT_FOR_HEALTHY

    def test_get_capabilities(self, provider):
        caps = provider.get_capabilities()
        assert "vllm" in caps.supported_engines
        assert caps.supports_lora is True

    def test_get_endpoint_info_initially_none(self, provider):
        assert provider.get_endpoint_info() is None

    def test_set_event_logger(self, provider):
        logger = Mock()
        provider.set_event_logger(logger)
        assert provider._mlflow_manager is logger

    def test_set_event_logger_none(self, provider):
        provider.set_event_logger(None)
        assert provider._mlflow_manager is None


# ---------------------------------------------------------------------------
# _looks_like_unresolved_env
# ---------------------------------------------------------------------------

class TestLooksLikeUnresolvedEnv:
    def test_true_for_env_placeholder(self, provider):
        assert provider._looks_like_unresolved_env("${MY_VAR}") is True

    def test_true_with_whitespace(self, provider):
        assert provider._looks_like_unresolved_env("  ${HOST}  ") is True

    def test_false_for_real_host(self, provider):
        assert provider._looks_like_unresolved_env("192.168.1.100") is False

    def test_false_for_alias(self, provider):
        assert provider._looks_like_unresolved_env("myserver") is False


# ---------------------------------------------------------------------------
# _connect_ssh
# ---------------------------------------------------------------------------

class TestConnectSSH:
    def test_already_connected_returns_ok(self, provider):
        provider._ssh_client = Mock()
        result = provider._connect_ssh()
        assert result.is_success()

    def test_no_host_returns_error(self, provider):
        # Bypass pydantic validation by overriding _ssh_cfg directly
        mock_ssh_cfg = Mock()
        mock_ssh_cfg.alias = None
        mock_ssh_cfg.host = None
        mock_ssh_cfg.key_path = None
        mock_ssh_cfg.key_env = None
        provider._ssh_cfg = mock_ssh_cfg
        result = provider._connect_ssh()
        assert result.is_failure()
        assert "SINGLENODE_SSH_HOST_NOT_CONFIGURED" in str(result.unwrap_err())

    def test_unresolved_env_host_returns_error(self, provider_cfg, engine_cfg, secrets):
        provider_cfg.connect.ssh.alias = None
        provider_cfg.connect.ssh.host = "${MY_HOST}"
        p = _mk_provider(provider_cfg, engine_cfg, secrets)
        result = p._connect_ssh()
        assert result.is_failure()
        assert "SINGLENODE_SSH_HOST_UNRESOLVED_ENV" in str(result.unwrap_err())

    def test_unresolved_env_user_returns_error(self, provider_cfg, engine_cfg, secrets):
        provider_cfg.connect.ssh.alias = None
        provider_cfg.connect.ssh.host = "192.168.1.100"
        provider_cfg.connect.ssh.user = "${MY_USER}"
        p = _mk_provider(provider_cfg, engine_cfg, secrets)
        result = p._connect_ssh()
        assert result.is_failure()
        assert "SINGLENODE_SSH_USER_UNRESOLVED_ENV" in str(result.unwrap_err())

    def test_ssh_connection_failure_clears_client(self, provider):
        mock_ssh_instance = Mock()
        mock_ssh_instance.test_connection.return_value = (False, "timeout")
        with patch.object(_mod, "SSHClient", return_value=mock_ssh_instance):
            result = provider._connect_ssh()
        assert result.is_failure()
        assert provider._ssh_client is None
        assert "SINGLENODE_SSH_CONNECT_FAILED" in str(result.unwrap_err())

    def test_ssh_connect_success(self, provider):
        mock_ssh_instance = Mock()
        mock_ssh_instance.test_connection.return_value = (True, "")
        with patch.object(_mod, "SSHClient", return_value=mock_ssh_instance):
            result = provider._connect_ssh()
        assert result.is_success()
        assert provider._ssh_client is mock_ssh_instance

    def test_ssh_init_exception_returns_error(self, provider):
        with patch.object(_mod, "SSHClient", side_effect=RuntimeError("boom")):
            result = provider._connect_ssh()
        assert result.is_failure()
        assert "SINGLENODE_SSH_INIT_FAILED" in str(result.unwrap_err())
        assert provider._ssh_client is None

    def test_key_env_fallback(self, provider_cfg, engine_cfg, secrets, monkeypatch):
        provider_cfg.connect.ssh.alias = None
        provider_cfg.connect.ssh.host = "1.2.3.4"
        provider_cfg.connect.ssh.user = "user"
        provider_cfg.connect.ssh.key_path = None
        provider_cfg.connect.ssh.key_env = "MY_SSH_KEY"
        monkeypatch.setenv("MY_SSH_KEY", "/home/user/.ssh/id_rsa")
        p = _mk_provider(provider_cfg, engine_cfg, secrets)

        mock_ssh_instance = Mock()
        mock_ssh_instance.test_connection.return_value = (True, "")
        captured = {}
        def _fake_ssh(**kw):
            captured.update(kw)
            return mock_ssh_instance

        with patch.object(_mod, "SSHClient", side_effect=_fake_ssh):
            result = p._connect_ssh()
        assert result.is_success()
        assert captured.get("key_path") == "/home/user/.ssh/id_rsa"


# ---------------------------------------------------------------------------
# deploy()
# ---------------------------------------------------------------------------

class TestDeploy:
    def test_construction_fails_when_merge_before_deploy_false(
        self, provider_cfg, engine_cfg, secrets
    ):
        """Post PR-7: ``validate_config`` runs in the constructor and
        rejects ``merge_before_deploy=False`` (vLLM MVP gate). The legacy
        in-deploy check is now unreachable — construction fails first
        with :class:`ProviderRegistryError`."""
        from ryotenkai_providers.registry import ProviderRegistryError

        with pytest.raises(ProviderRegistryError) as exc_info:
            _mk_provider(
                provider_cfg, engine_cfg, secrets, merge_before_deploy=False
            )
        assert exc_info.value.code == "VLLM_LIVE_LORA_NOT_SUPPORTED"
        assert "merge_before_deploy" in exc_info.value.message.lower() or \
            "live lora" in exc_info.value.message.lower()

    def test_deploy_fails_when_ssh_connect_fails(self, provider):
        with patch.object(provider, "_connect_ssh") as mock_conn:
            mock_conn.return_value = Err(InferenceError(message="no connection", code="X"))
            result = provider.deploy("model", run_id="r1", base_model_id="base")
        assert result.is_failure()
        assert "SINGLENODE_SSH_CONNECT_FAILED" in str(result.unwrap_err())

    def test_deploy_fails_when_health_check_fails(self, provider):
        mock_ssh = Mock()
        with patch.object(provider, "_connect_ssh", return_value=Ok(None)):
            provider._ssh_client = mock_ssh
            mock_health = Mock()
            mock_health.run_all_checks.return_value = Mock(passed=False, errors=["GPU missing"])
            with patch.object(_mod, "SingleNodeHealthCheck", return_value=mock_health):
                result = provider.deploy("model", run_id="r1", base_model_id="base")
        assert result.is_failure()
        assert "SINGLENODE_HEALTH_CHECK_FAILED" in str(result.unwrap_err())

    def test_deploy_fails_when_dir_creation_fails(self, provider):
        mock_ssh = Mock()
        mock_ssh.create_directory.return_value = (False, "permission denied")
        with patch.object(provider, "_connect_ssh", return_value=Ok(None)):
            provider._ssh_client = mock_ssh
            mock_health = Mock()
            mock_health.run_all_checks.return_value = Mock(passed=True, errors=None)
            with patch.object(_mod, "SingleNodeHealthCheck", return_value=mock_health):
                result = provider.deploy("model", run_id="r1", base_model_id="base")
        assert result.is_failure()
        assert "SINGLENODE_DIR_CREATE_FAILED" in str(result.unwrap_err())

    def test_deploy_fails_when_local_adapter_not_dir(self, provider, tmp_path):
        adapter_file = tmp_path / "adapter.bin"
        adapter_file.write_text("weights")

        mock_ssh = Mock()
        mock_ssh.create_directory.return_value = (True, "")
        with patch.object(provider, "_connect_ssh", return_value=Ok(None)):
            provider._ssh_client = mock_ssh
            mock_health = Mock()
            mock_health.run_all_checks.return_value = Mock(passed=True, errors=None)
            with patch.object(_mod, "SingleNodeHealthCheck", return_value=mock_health):
                result = provider.deploy(
                    str(adapter_file),
                    run_id="r1",
                    base_model_id="base",
                    lora_path=str(adapter_file),
                )
        assert result.is_failure()
        assert "SINGLENODE_ADAPTER_NOT_DIR" in str(result.unwrap_err())

    def test_deploy_fails_when_adapter_upload_fails(self, provider, tmp_path):
        from ryotenkai_shared.errors import SSHTransferFailedError

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        mock_ssh = Mock()
        mock_ssh.create_directory.return_value = (True, "")
        # New SSH contract: upload_directory raises SSHTransferFailedError on failure.
        mock_ssh.upload_directory.side_effect = SSHTransferFailedError(
            detail="upload failed",
            context={"op": "upload_directory"},
        )

        with patch.object(provider, "_connect_ssh", return_value=Ok(None)):
            provider._ssh_client = mock_ssh
            mock_health = Mock()
            mock_health.run_all_checks.return_value = Mock(passed=True, errors=None)
            with patch.object(_mod, "SingleNodeHealthCheck", return_value=mock_health):
                result = provider.deploy(
                    str(adapter_dir),
                    run_id="r1",
                    base_model_id="base",
                    lora_path=str(adapter_dir),
                )
        assert result.is_failure()
        assert "SINGLENODE_ADAPTER_UPLOAD_FAILED" in str(result.unwrap_err())

    def test_deploy_fails_when_merge_fails(self, provider):
        mock_ssh = Mock()
        mock_ssh.create_directory.return_value = (True, "")
        with patch.object(provider, "_connect_ssh", return_value=Ok(None)):
            provider._ssh_client = mock_ssh
            mock_health = Mock()
            mock_health.run_all_checks.return_value = Mock(passed=True, errors=None)
            with patch.object(_mod, "SingleNodeHealthCheck", return_value=mock_health), patch.object(
                provider,
                "_run_prepare_plan",
                return_value=Err(InferenceError(message="merge failed", code="MERGE_ERR")),
            ):
                result = provider.deploy(
                    "hf/model",
                    run_id="r1",
                    base_model_id="base",
                )
        assert result.is_failure()

    def test_deploy_fails_when_vllm_start_fails(self, provider):
        mock_ssh = Mock()
        mock_ssh.create_directory.return_value = (True, "")
        with patch.object(provider, "_connect_ssh", return_value=Ok(None)):
            provider._ssh_client = mock_ssh
            mock_health = Mock()
            mock_health.run_all_checks.return_value = Mock(passed=True, errors=None)
            with patch.object(_mod, "SingleNodeHealthCheck", return_value=mock_health):  # noqa: SIM117
                with patch.object(provider, "_run_prepare_plan", return_value=Ok(None)):
                    with patch.object(
                        provider,
                        "_start_engine_container",
                        return_value=Err(InferenceError(message="vllm failed", code="VLLM_ERR")),
                    ):
                        result = provider.deploy(
                            "hf/model",
                            run_id="r1",
                            base_model_id="base",
                        )
        assert result.is_failure()

    def test_deploy_success_returns_endpoint_info(self, provider):
        mock_ssh = Mock()
        mock_ssh.create_directory.return_value = (True, "")
        mock_ssh.exec_command.return_value = (True, "", "")

        with patch.object(provider, "_connect_ssh", return_value=Ok(None)):
            provider._ssh_client = mock_ssh
            mock_health = Mock()
            mock_health.run_all_checks.return_value = Mock(passed=True, errors=None)
            with patch.object(_mod, "SingleNodeHealthCheck", return_value=mock_health):  # noqa: SIM117
                with patch.object(provider, "_run_prepare_plan", return_value=Ok(None)):
                    with patch.object(provider, "_start_engine_container", return_value=Ok(None)):
                        result = provider.deploy(
                            "hf/model",
                            run_id="r1",
                            base_model_id="meta-llama/Llama-2-7b",
                        )

        assert result.is_success()
        endpoint = result.unwrap()
        assert endpoint.endpoint_url.startswith("http://127.0.0.1:8000")
        assert endpoint.engine == "vllm"
        assert provider.get_endpoint_info() is endpoint

    # NOTE: test_deploy_with_quantization_overrides_engine_cfg DELETED.
    # The ``quantization`` kwarg was removed from ``IInferenceProvider.deploy()``
    # (AD-A8 / RD-6) — it was an engine-specific knob leaking through the
    # generic provider API. Engines now read ``cfg.quantization`` from their
    # own typed engine config; cross-validation lives in
    # ``VLLMEngineRuntime.validate_config`` (covered in
    # ``packages/engines/tests/unit/vllm/test_runtime.py``).


# ---------------------------------------------------------------------------
# health_check()
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_returns_err_when_no_ssh_client(self, provider):
        provider._ssh_client = None
        result = provider.health_check()
        assert result.is_failure()
        assert "SINGLENODE_NOT_DEPLOYED" in str(result.unwrap_err())

    def test_returns_err_when_command_fails(self, provider):
        mock_ssh = Mock()
        mock_ssh.exec_command.return_value = (False, "", "Connection refused")
        provider._ssh_client = mock_ssh
        result = provider.health_check()
        assert result.is_failure()
        assert "SINGLENODE_HEALTH_CHECK_COMMAND_FAILED" in str(result.unwrap_err())

    def test_returns_ok_true_when_output_is_1(self, provider):
        mock_ssh = Mock()
        mock_ssh.exec_command.return_value = (True, "1", "")
        provider._ssh_client = mock_ssh
        result = provider.health_check()
        assert result.is_success()
        assert result.unwrap() is True

    def test_returns_ok_false_when_output_is_0(self, provider):
        mock_ssh = Mock()
        mock_ssh.exec_command.return_value = (True, "0", "")
        provider._ssh_client = mock_ssh
        result = provider.health_check()
        assert result.is_success()
        assert result.unwrap() is False

    def test_returns_ok_false_for_partial_match(self, provider):
        mock_ssh = Mock()
        mock_ssh.exec_command.return_value = (True, "10", "")
        provider._ssh_client = mock_ssh
        result = provider.health_check()
        assert result.is_success()
        assert result.unwrap() is False


# ---------------------------------------------------------------------------
# undeploy()
# ---------------------------------------------------------------------------

class TestUndeploy:
    def test_undeploy_without_ssh_client(self, provider):
        provider._ssh_client = None
        result = provider.undeploy()
        assert result.is_success()

    def test_undeploy_with_ssh_client_calls_docker_rm(self, provider, fake_docker):
        mock_ssh = Mock()
        provider._ssh_client = mock_ssh
        from ryotenkai_providers.inference.interfaces import EndpointInfo
        provider._endpoint_info = Mock(spec=EndpointInfo)
        result = provider.undeploy()
        assert result.is_success()
        assert provider._endpoint_info is None
        assert len(fake_docker.calls_for("rm_force")) == 1


# ---------------------------------------------------------------------------
# collect_startup_logs()
# ---------------------------------------------------------------------------

class TestCollectStartupLogs:
    def test_no_ssh_client_does_nothing(self, provider, tmp_path):
        provider._ssh_client = None
        log_path = tmp_path / "logs.txt"
        provider.collect_startup_logs(local_path=log_path)
        assert not log_path.exists()

    def test_writes_logs_when_docker_logs_succeed(self, provider, fake_docker, tmp_path):
        mock_ssh = Mock()
        provider._ssh_client = mock_ssh
        log_path = tmp_path / "subdir" / "logs.txt"
        # ``collect_startup_logs`` fetches the inference container's logs;
        # register the canonical container and seed its log buffer.
        fake_docker.register_container(_mod.VLLM_INFERENCE_CONTAINER_NAME)
        fake_docker.append_logs(_mod.VLLM_INFERENCE_CONTAINER_NAME, "container log output")
        provider.collect_startup_logs(local_path=log_path)
        assert log_path.exists()
        assert "container log output" in log_path.read_text()

    def test_skips_write_when_logs_empty(self, provider, fake_docker, tmp_path):
        mock_ssh = Mock()
        provider._ssh_client = mock_ssh
        log_path = tmp_path / "logs.txt"
        # No logs registered ⇒ FakeDockerClient returns ``Ok("")``.
        provider.collect_startup_logs(local_path=log_path)
        assert not log_path.exists()

    def test_handles_exception_gracefully(self, provider, fake_docker, tmp_path):
        mock_ssh = Mock()
        provider._ssh_client = mock_ssh
        log_path = tmp_path / "logs.txt"
        # Patch the fake's ``logs`` method to raise — ``collect_startup_logs``
        # swallows any exception and must not propagate it.
        with patch.object(fake_docker, "logs", side_effect=RuntimeError("oops")):
            provider.collect_startup_logs(local_path=log_path)


# ---------------------------------------------------------------------------
# build_inference_artifacts()
# ---------------------------------------------------------------------------

class TestBuildInferenceArtifacts:
    def _mk_ctx(self, provider):
        from ryotenkai_providers.inference.interfaces import EndpointInfo, InferenceArtifactsContext
        endpoint = EndpointInfo(
            endpoint_url="http://127.0.0.1:8000/v1",
            api_type="openai_compatible",
            provider_type="single_node",
            engine="vllm",
            model_id="test-model",
            health_url="http://127.0.0.1:8000/v1/models",
            resource_id="ryotenkai-inference-vllm",
        )
        return InferenceArtifactsContext(
            run_name="test-run",
            mlflow_run_id="mlflow-123",
            model_source="hf/adapter",
            endpoint=endpoint,
        )

    def test_artifacts_with_alias(self, provider):
        ctx = self._mk_ctx(provider)
        with patch.object(provider, "_resolve_llm_manifest_block", return_value={"system_prompt": None, "system_prompt_source": None}):
            result = provider.build_inference_artifacts(ctx=ctx)
        assert result.is_success()
        arts = result.unwrap()
        assert "test-node" in arts.manifest["endpoint"]["tunnel_hint"]

    def test_artifacts_without_alias(self, provider_cfg, engine_cfg, secrets):
        provider_cfg.connect.ssh.alias = None
        provider_cfg.connect.ssh.host = "10.0.0.1"
        provider_cfg.connect.ssh.user = "myuser"
        p = _mk_provider(provider_cfg, engine_cfg, secrets)

        from ryotenkai_providers.inference.interfaces import EndpointInfo, InferenceArtifactsContext
        endpoint = EndpointInfo(
            endpoint_url="http://127.0.0.1:8000/v1",
            api_type="openai_compatible",
            provider_type="single_node",
            engine="vllm",
            model_id="test-model",
            health_url="http://127.0.0.1:8000/v1/models",
            resource_id="ryotenkai-inference-vllm",
        )
        ctx = InferenceArtifactsContext(
            run_name="test-run",
            mlflow_run_id="mlflow-123",
            model_source="hf/adapter",
            endpoint=endpoint,
        )
        with patch.object(p, "_resolve_llm_manifest_block", return_value={"system_prompt": None, "system_prompt_source": None}):
            result = p.build_inference_artifacts(ctx=ctx)
        assert result.is_success()
        arts = result.unwrap()
        assert "myuser@10.0.0.1" in arts.manifest["endpoint"]["tunnel_hint"]


# ---------------------------------------------------------------------------
# activate_for_eval / deactivate_after_eval
# ---------------------------------------------------------------------------

class TestEvalLifecycle:
    def test_activate_for_eval_without_endpoint_fails(self, provider):
        provider._endpoint_info = None
        result = provider.activate_for_eval()
        assert result.is_failure()
        assert "SINGLENODE_NOT_DEPLOYED" in str(result.unwrap_err())

    def test_activate_for_eval_returns_url(self, provider):
        from ryotenkai_providers.inference.interfaces import EndpointInfo
        provider._endpoint_info = EndpointInfo(
            endpoint_url="http://127.0.0.1:8000/v1",
            api_type="openai_compatible",
            provider_type="single_node",
            engine="vllm",
            model_id="m",
            health_url="http://127.0.0.1:8000/v1/models",
            resource_id="c",
        )
        result = provider.activate_for_eval()
        assert result.is_success()
        assert "8000" in result.unwrap()

    def test_deactivate_after_eval_is_noop(self, provider):
        result = provider.deactivate_after_eval()
        assert result.is_success()


# ---------------------------------------------------------------------------
# _resolve_llm_manifest_block
# ---------------------------------------------------------------------------

class TestResolveLlmManifestBlock:
    # NOTE: TestResolveLlmManifestBlock tests previously patched
    # ``ryotenkai_control.evaluation.system_prompt.SystemPromptLoader``,
    # but that module was deleted in a prior cleanup. The provider's
    # ``_resolve_llm_manifest_block`` now resolves prompts via a different
    # mechanism — coverage moved to control-side tests. Empty class kept
    # as a discoverability anchor for future restoration.
    pass




# ---------------------------------------------------------------------------
# Migration note (PR-16): legacy ``_run_merge_container`` and
# ``_merge_adapter_remote`` were deleted from the provider in favor of the
# generic engine-driven prepare-plan runner. Their test coverage moved to:
#
#   * ``test_run_prepare_plan.py`` — provider's generic plan-runner across
#     positive, negative (every error code), boundary, invariant, logic,
#     and combinatorial categories.
#   * ``packages/engines/tests/unit/vllm/test_prepare_model.py`` — the
#     vLLM-side merge plan-builder (replaces the legacy command-format
#     coverage).
#   * ``packages/providers/tests/unit/providers/inference/test_format_prepare_step.py``
#     — shell-formatting + injection-safety checks for ``PrepareStep``.
#
# The legacy ``_merge_adapter_remote`` (deprecated host-merge fallback)
# is gone for good — no successor.
