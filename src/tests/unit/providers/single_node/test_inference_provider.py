"""
Unit tests for SingleNodeInferenceProvider.

Covers missing lines: 108, 129-303, 307, 310, 314-324, 343-345, 401-406, 439,
446, 501-503, 512-513, 522-583, 614-692, 725, 741-742, 766, 769
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

import src.providers.single_node.inference.provider as _mod
from src.config.providers.single_node import (
    SingleNodeConfig,
    SingleNodeConnectConfig,
    SingleNodeInferenceConfig,
    SingleNodeTrainingConfig,
)
from src.config.providers.ssh import SSHConfig
from src.providers.single_node.inference.provider import SingleNodeInferenceProvider
from src.providers.training.interfaces import GPUInfo
from src.utils.config import (
    InferenceSingleNodeServeConfig,
    InferenceVLLMEngineConfig,
    Secrets,
)
from src.utils.result import Err, InferenceError, Ok


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ssh_cfg():
    return SSHConfig(alias="test-node", host="192.168.1.100", port=22, user="testuser")


@pytest.fixture()
def provider_cfg(ssh_cfg):
    return SingleNodeConfig(
        connect=SingleNodeConnectConfig(ssh=ssh_cfg),
        training=SingleNodeTrainingConfig(
            workspace_path="/home/user/workspace",
            docker_image="test/runtime:latest",
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
    return InferenceVLLMEngineConfig(
        merge_image="test-merge:v1.0",
        serve_image="test-vllm:v0.6.3",
        tensor_parallel_size=1,
        max_model_len=4096,
    )


@pytest.fixture()
def secrets():
    return Secrets(hf_token="hf-test-token")


def _mk_pipeline_config(provider_cfg, engine_cfg, *, merge_before_deploy: bool = True):
    cfg = Mock()
    cfg.get_provider_config = lambda *a, **k: provider_cfg.model_dump(mode="python")
    cfg.inference = Mock()
    cfg.inference.engines = Mock()
    cfg.inference.engines.vllm = engine_cfg
    cfg.inference.common = Mock()
    cfg.inference.common.lora = Mock()
    cfg.inference.common.lora.merge_before_deploy = merge_before_deploy
    cfg.inference.engine = "vllm"
    cfg.model = Mock()
    cfg.model.name = "meta-llama/Llama-2-7b-hf"
    cfg.model.trust_remote_code = False
    cfg.experiment_tracking = None
    return cfg


@pytest.fixture()
def provider(provider_cfg, engine_cfg, secrets):
    pipeline_cfg = _mk_pipeline_config(provider_cfg, engine_cfg)
    p = SingleNodeInferenceProvider(config=pipeline_cfg, secrets=secrets)
    p._run_id = "run_test_001"
    return p


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_provider_name(self, provider):
        assert provider.provider_name == "single_node"

    def test_provider_type(self, provider):
        assert provider.provider_type == "single_node"

    def test_get_pipeline_readiness_mode(self, provider):
        from src.providers.inference.interfaces import PipelineReadinessMode
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
        pipeline_cfg = _mk_pipeline_config(provider_cfg, engine_cfg)
        p = SingleNodeInferenceProvider(config=pipeline_cfg, secrets=secrets)
        result = p._connect_ssh()
        assert result.is_failure()
        assert "SINGLENODE_SSH_HOST_UNRESOLVED_ENV" in str(result.unwrap_err())

    def test_unresolved_env_user_returns_error(self, provider_cfg, engine_cfg, secrets):
        provider_cfg.connect.ssh.alias = None
        provider_cfg.connect.ssh.host = "192.168.1.100"
        provider_cfg.connect.ssh.user = "${MY_USER}"
        pipeline_cfg = _mk_pipeline_config(provider_cfg, engine_cfg)
        p = SingleNodeInferenceProvider(config=pipeline_cfg, secrets=secrets)
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
        pipeline_cfg = _mk_pipeline_config(provider_cfg, engine_cfg)
        p = SingleNodeInferenceProvider(config=pipeline_cfg, secrets=secrets)

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
    def test_deploy_fails_when_merge_before_deploy_false(self, provider_cfg, engine_cfg, secrets):
        pipeline_cfg = _mk_pipeline_config(provider_cfg, engine_cfg, merge_before_deploy=False)
        p = SingleNodeInferenceProvider(config=pipeline_cfg, secrets=secrets)
        result = p.deploy(
            "hf-repo/model",
            run_id="run1",
            base_model_id="meta-llama/Llama-2-7b",
        )
        assert result.is_failure()
        assert "SINGLENODE_LORA_MERGE_REQUIRED" in str(result.unwrap_err())

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
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        mock_ssh = Mock()
        mock_ssh.create_directory.return_value = (True, "")
        upload_err = Mock()
        upload_err.is_failure.return_value = True
        upload_err.unwrap_err.return_value = "upload failed"
        mock_ssh.upload_directory.return_value = upload_err

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
            with patch.object(_mod, "SingleNodeHealthCheck", return_value=mock_health):
                with patch.object(
                    provider,
                    "_run_merge_container",
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
            with patch.object(_mod, "SingleNodeHealthCheck", return_value=mock_health):
                with patch.object(provider, "_run_merge_container", return_value=Ok(None)):
                    with patch.object(
                        provider,
                        "_start_vllm_container",
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
            with patch.object(_mod, "SingleNodeHealthCheck", return_value=mock_health):
                with patch.object(provider, "_run_merge_container", return_value=Ok(None)):
                    with patch.object(provider, "_start_vllm_container", return_value=Ok(None)):
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

    def test_deploy_with_quantization_overrides_engine_cfg(self, provider):
        mock_ssh = Mock()
        mock_ssh.create_directory.return_value = (True, "")
        mock_ssh.exec_command.return_value = (True, "", "")

        captured_engine_cfg = {}
        def _fake_start(*, ssh, engine_cfg, workspace_host_path, model_path_in_container):
            captured_engine_cfg["quantization"] = engine_cfg.quantization
            return Ok(None)

        with patch.object(provider, "_connect_ssh", return_value=Ok(None)):
            provider._ssh_client = mock_ssh
            mock_health = Mock()
            mock_health.run_all_checks.return_value = Mock(passed=True, errors=None)
            with patch.object(_mod, "SingleNodeHealthCheck", return_value=mock_health):
                with patch.object(provider, "_run_merge_container", return_value=Ok(None)):
                    with patch.object(provider, "_start_vllm_container", side_effect=_fake_start):
                        provider.deploy(
                            "hf/model",
                            run_id="r1",
                            base_model_id="base",
                            quantization="awq",
                        )

        assert captured_engine_cfg.get("quantization") == "awq"


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

    def test_undeploy_with_ssh_client_calls_docker_rm(self, provider):
        mock_ssh = Mock()
        provider._ssh_client = mock_ssh
        from src.providers.inference.interfaces import EndpointInfo
        provider._endpoint_info = Mock(spec=EndpointInfo)
        with patch.object(_mod, "docker_rm_force") as mock_rm:
            mock_rm.return_value = Ok(None)
            result = provider.undeploy()
        assert result.is_success()
        assert provider._endpoint_info is None
        mock_rm.assert_called_once()


# ---------------------------------------------------------------------------
# collect_startup_logs()
# ---------------------------------------------------------------------------

class TestCollectStartupLogs:
    def test_no_ssh_client_does_nothing(self, provider, tmp_path):
        provider._ssh_client = None
        log_path = tmp_path / "logs.txt"
        provider.collect_startup_logs(local_path=log_path)
        assert not log_path.exists()

    def test_writes_logs_when_docker_logs_succeed(self, provider, tmp_path):
        mock_ssh = Mock()
        provider._ssh_client = mock_ssh
        log_path = tmp_path / "subdir" / "logs.txt"
        with patch.object(_mod, "docker_logs", return_value=Ok("container log output")):
            provider.collect_startup_logs(local_path=log_path)
        assert log_path.exists()
        assert "container log output" in log_path.read_text()

    def test_skips_write_when_logs_empty(self, provider, tmp_path):
        mock_ssh = Mock()
        provider._ssh_client = mock_ssh
        log_path = tmp_path / "logs.txt"
        with patch.object(_mod, "docker_logs", return_value=Ok("")):
            provider.collect_startup_logs(local_path=log_path)
        assert not log_path.exists()

    def test_handles_exception_gracefully(self, provider, tmp_path):
        mock_ssh = Mock()
        provider._ssh_client = mock_ssh
        log_path = tmp_path / "logs.txt"
        with patch.object(_mod, "docker_logs", side_effect=RuntimeError("oops")):
            provider.collect_startup_logs(local_path=log_path)


# ---------------------------------------------------------------------------
# build_inference_artifacts()
# ---------------------------------------------------------------------------

class TestBuildInferenceArtifacts:
    def _mk_ctx(self, provider):
        from src.providers.inference.interfaces import EndpointInfo, InferenceArtifactsContext
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
        pipeline_cfg = _mk_pipeline_config(provider_cfg, engine_cfg)
        p = SingleNodeInferenceProvider(config=pipeline_cfg, secrets=secrets)

        from src.providers.inference.interfaces import EndpointInfo, InferenceArtifactsContext
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
        from src.providers.inference.interfaces import EndpointInfo
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
    def test_returns_none_on_value_error(self, provider):
        with patch("src.evaluation.system_prompt.SystemPromptLoader.load", side_effect=ValueError("bad config")):
            block = provider._resolve_llm_manifest_block()
        assert block["system_prompt"] is None
        assert block["system_prompt_source"] is None

    def test_returns_prompt_when_loader_succeeds(self, provider):
        fake_result = Mock()
        fake_result.text = "You are a helpful assistant."
        fake_result.source = "inline"
        with patch("src.evaluation.system_prompt.SystemPromptLoader.load", return_value=fake_result):
            block = provider._resolve_llm_manifest_block()
        assert block["system_prompt"] == "You are a helpful assistant."
        assert block["system_prompt_source"] == "inline"


# ---------------------------------------------------------------------------
# _run_merge_container — error paths
# ---------------------------------------------------------------------------

class TestRunMergeContainerErrors:
    def test_fails_when_merge_image_not_configured(self, provider, engine_cfg, provider_cfg, secrets):
        engine_cfg_no_img = InferenceVLLMEngineConfig(
            merge_image="",
            serve_image="test-vllm:v0.6.3",
        )
        pipeline_cfg = _mk_pipeline_config(provider_cfg, engine_cfg_no_img)
        p = SingleNodeInferenceProvider(config=pipeline_cfg, secrets=secrets)
        mock_ssh = Mock()
        result = p._run_merge_container(
            ssh=mock_ssh,
            base_model="base",
            adapter_path="hf/adapter",
            output_path="/home/user/inference/runs/test/model",
            cache_dir="/home/user/inference/hf_cache",
        )
        assert result.is_failure()
        assert "SINGLENODE_MERGE_IMAGE_NOT_CONFIGURED" in str(result.unwrap_err())

    def test_fails_when_image_pull_fails(self, provider):
        workspace = provider._provider_cfg.inference.serve.workspace.rstrip("/")
        mock_ssh = Mock()
        with patch.object(
            provider,
            "_ensure_docker_image",
            return_value=Err(InferenceError(message="pull failed", code="IMG_FAIL")),
        ):
            result = provider._run_merge_container(
                ssh=mock_ssh,
                base_model="base",
                adapter_path="hf/adapter",
                output_path=f"{workspace}/runs/test/model",
                cache_dir=f"{workspace}/hf_cache",
            )
        assert result.is_failure()
        assert "SINGLENODE_MERGE_IMAGE_PULL_FAILED" in str(result.unwrap_err())

    def test_fails_when_output_path_outside_workspace(self, provider):
        mock_ssh = Mock()
        with patch.object(provider, "_ensure_docker_image", return_value=Ok(None)):
            result = provider._run_merge_container(
                ssh=mock_ssh,
                base_model="base",
                adapter_path="hf/adapter",
                output_path="/some/other/path/model",
                cache_dir="/home/user/inference/hf_cache",
            )
        assert result.is_failure()
        assert "INFERENCE_MERGE_INVALID_PATH" in str(result.unwrap_err())

    def test_fails_when_cache_dir_outside_workspace(self, provider):
        workspace = provider._provider_cfg.inference.serve.workspace.rstrip("/")
        mock_ssh = Mock()
        mock_ssh.exec_command.return_value = (True, "", "")
        with patch.object(provider, "_ensure_docker_image", return_value=Ok(None)):
            result = provider._run_merge_container(
                ssh=mock_ssh,
                base_model="base",
                adapter_path="hf/adapter",
                output_path=f"{workspace}/runs/test/model",
                cache_dir="/some/other/cache",
            )
        assert result.is_failure()
        assert "INFERENCE_MERGE_INVALID_PATH" in str(result.unwrap_err())

    def test_fails_when_adapter_absolute_path_outside_workspace(self, provider):
        workspace = provider._provider_cfg.inference.serve.workspace.rstrip("/")
        mock_ssh = Mock()
        mock_ssh.exec_command.return_value = (True, "", "")
        with patch.object(provider, "_ensure_docker_image", return_value=Ok(None)):
            result = provider._run_merge_container(
                ssh=mock_ssh,
                base_model="base",
                adapter_path="/some/other/adapter",
                output_path=f"{workspace}/runs/test/model",
                cache_dir=f"{workspace}/hf_cache",
            )
        assert result.is_failure()
        assert "INFERENCE_MERGE_INVALID_PATH" in str(result.unwrap_err())

    def test_fails_when_container_start_fails(self, provider):
        workspace = provider._provider_cfg.inference.serve.workspace.rstrip("/")
        mock_ssh = Mock()
        mock_ssh.exec_command.side_effect = [
            (True, "", ""),   # rm -rf
            (True, "", ""),   # mkdir
            (False, "", "docker error"),  # docker run --detach
        ]
        mock_ssh.upload_file.return_value = (True, "")

        with patch.object(provider, "_ensure_docker_image", return_value=Ok(None)):
            with patch("pathlib.Path.exists", return_value=True):
                result = provider._run_merge_container(
                    ssh=mock_ssh,
                    base_model="base",
                    adapter_path="hf/adapter",
                    output_path=f"{workspace}/runs/test/model",
                    cache_dir=f"{workspace}/hf_cache",
                )
        assert result.is_failure()
        assert "SINGLENODE_MERGE_CONTAINER_START_FAILED" in str(result.unwrap_err())

    def test_fails_when_exit_code_nonzero(self, provider):
        workspace = provider._provider_cfg.inference.serve.workspace.rstrip("/")
        mock_ssh = MagicMock()
        mock_ssh.upload_file.return_value = (True, "")
        mock_ssh.exec_command.side_effect = [
            (True, "", ""),   # rm -rf
            (True, "", ""),   # mkdir
            (True, "container123", ""),  # docker run --detach
        ]

        with patch.object(provider, "_ensure_docker_image", return_value=Ok(None)):
            with patch("pathlib.Path.exists", return_value=True):
                with patch.object(_mod, "docker_is_container_running", return_value=False):
                    with patch.object(_mod, "docker_logs", return_value=Ok("no success")):
                        with patch.object(
                            _mod, "docker_container_exit_code", return_value=Ok(1)
                        ):
                            with patch.object(_mod, "docker_rm_force", return_value=Ok(None)):
                                result = provider._run_merge_container(
                                    ssh=mock_ssh,
                                    base_model="base",
                                    adapter_path="hf/adapter",
                                    output_path=f"{workspace}/runs/test/model",
                                    cache_dir=f"{workspace}/hf_cache",
                                )
        assert result.is_failure()
        assert "SINGLENODE_MERGE_CONTAINER_FAILED" in str(result.unwrap_err())

    def test_fails_when_no_success_marker_in_logs(self, provider):
        workspace = provider._provider_cfg.inference.serve.workspace.rstrip("/")
        mock_ssh = MagicMock()
        mock_ssh.upload_file.return_value = (True, "")
        mock_ssh.exec_command.side_effect = [
            (True, "", ""),
            (True, "", ""),
            (True, "container123", ""),
        ]

        with patch.object(provider, "_ensure_docker_image", return_value=Ok(None)):
            with patch("pathlib.Path.exists", return_value=True):
                with patch.object(_mod, "docker_is_container_running", return_value=False):
                    with patch.object(_mod, "docker_logs", return_value=Ok("completed without marker")):
                        with patch.object(
                            _mod, "docker_container_exit_code", return_value=Ok(0)
                        ):
                            with patch.object(_mod, "docker_rm_force", return_value=Ok(None)):
                                result = provider._run_merge_container(
                                    ssh=mock_ssh,
                                    base_model="base",
                                    adapter_path="hf/adapter",
                                    output_path=f"{workspace}/runs/test/model",
                                    cache_dir=f"{workspace}/hf_cache",
                                )
        assert result.is_failure()
        assert "SINGLENODE_MERGE_NO_SUCCESS_MARKER" in str(result.unwrap_err())

    def test_fails_when_artifacts_not_found_after_success(self, provider):
        workspace = provider._provider_cfg.inference.serve.workspace.rstrip("/")
        mock_ssh = MagicMock()
        mock_ssh.upload_file.return_value = (True, "")

        call_count = [0]
        def _exec_side_effect(cmd, **kw):
            call_count[0] += 1
            if "rm -rf" in cmd or "mkdir" in cmd:
                return (True, "", "")
            if "docker run" in cmd:
                return (True, "container123", "")
            if "test -f" in cmd:
                return (True, "MISSING", "")
            if "ls -lah" in cmd:
                return (True, "", "")
            return (True, "", "")

        mock_ssh.exec_command.side_effect = _exec_side_effect

        with patch.object(provider, "_ensure_docker_image", return_value=Ok(None)):
            with patch("pathlib.Path.exists", return_value=True):
                with patch.object(_mod, "docker_is_container_running", return_value=False):
                    with patch.object(_mod, "docker_logs", return_value=Ok("MERGE_SUCCESS")):
                        with patch.object(
                            _mod, "docker_container_exit_code", return_value=Ok(0)
                        ):
                            with patch.object(_mod, "docker_rm_force", return_value=Ok(None)):
                                result = provider._run_merge_container(
                                    ssh=mock_ssh,
                                    base_model="base",
                                    adapter_path="hf/adapter",
                                    output_path=f"{workspace}/runs/test/model",
                                    cache_dir=f"{workspace}/hf_cache",
                                )
        assert result.is_failure()
        assert "SINGLENODE_MERGE_ARTIFACTS_NOT_FOUND" in str(result.unwrap_err())


# ---------------------------------------------------------------------------
# _merge_adapter_remote (deprecated host-based merge)
# ---------------------------------------------------------------------------

class TestMergeAdapterRemote:
    def test_fails_when_python_deps_missing(self, provider):
        mock_ssh = Mock()
        mock_ssh.exec_command.return_value = (False, "", "No module named transformers")
        result = provider._merge_adapter_remote(
            ssh=mock_ssh,
            base_model_id="base",
            adapter_ref="hf/adapter",
            merged_dir="/output",
            hf_cache_dir="/cache",
            trust_remote_code=False,
        )
        assert result.is_failure()
        assert "SINGLENODE_MERGE_DEPS_MISSING" in str(result.unwrap_err())

    def test_fails_when_merge_script_no_ok_marker(self, provider):
        mock_ssh = Mock()
        mock_ssh.exec_command.side_effect = [
            (True, "OK", ""),   # deps check
            (True, "", ""),     # rm -rf
            (True, "", ""),     # mkdir cache
            (True, "no marker", ""),  # merge python script
        ]
        result = provider._merge_adapter_remote(
            ssh=mock_ssh,
            base_model_id="base",
            adapter_ref="hf/adapter",
            merged_dir="/output",
            hf_cache_dir="/cache",
            trust_remote_code=False,
        )
        assert result.is_failure()
        assert "SINGLENODE_MERGE_FAILED" in str(result.unwrap_err())

    def test_succeeds_when_merge_ok_in_stdout(self, provider):
        mock_ssh = Mock()
        mock_ssh.exec_command.side_effect = [
            (True, "OK", ""),       # deps check
            (True, "", ""),         # rm -rf
            (True, "", ""),         # mkdir cache
            (True, "MERGE_OK\n", ""),  # merge script
        ]
        result = provider._merge_adapter_remote(
            ssh=mock_ssh,
            base_model_id="base",
            adapter_ref="hf/adapter",
            merged_dir="/output",
            hf_cache_dir="/cache",
            trust_remote_code=False,
        )
        assert result.is_success()
