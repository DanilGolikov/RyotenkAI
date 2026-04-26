"""
Single-node inference provider (SSH + Docker(vLLM)).

MVP assumptions:
- inference node is reachable via SSH (alias or explicit)
- Docker is installed and has NVIDIA runtime configured
- vLLM is run in a Docker container
- LoRA adapters are merged into base model before serving (merge_before_deploy=true)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.constants import INFERENCE_MANIFEST_FILENAME, PROVIDER_SINGLE_NODE, VLLM_INFERENCE_CONTAINER_NAME
from src.pipeline.inference.vllm import VLLMEngine
from src.providers.constants import CATEGORY_INFERENCE as _KEY_INFERENCE
from src.providers.constants import ENCODING_UTF8 as _ENCODING_UTF8
from src.providers.constants import SHA12_LEN
from src.providers.inference.interfaces import (
    EndpointInfo,
    IInferenceProvider,
    InferenceArtifacts,
    InferenceArtifactsContext,
    InferenceCapabilities,
    InferenceEventLogger,
    PipelineReadinessMode,
)
from src.providers.single_node.training.health_check import SingleNodeHealthCheck
from src.utils.constants import LOG_OUTPUT_LONG_CHARS, LOG_OUTPUT_SHORT_CHARS
from src.utils.docker import (
    docker_container_exit_code,
    docker_is_container_running,
    docker_logs,
    docker_rm_force,
    ensure_docker_image,
)
from src.utils.logger import get_run_log_dir, logger
from src.utils.result import Err, InferenceError, Ok, Result
from src.utils.ssh_client import SSHClient

from .artifacts import CHAT_SCRIPT as _CHAT_SCRIPT
from .artifacts import render_readme as _render_readme

if TYPE_CHECKING:
    from src.utils.config import InferenceVLLMEngineConfig, PipelineConfig, Secrets

PULL_TIMEOUT = 1200
MERGE_TIMEOUT = 3600
SOURCE_SINGLE_NODE_INFERENCE = "SingleNodeInferenceProvider"

# JSON dict keys used in >3 places (WPS226)
_KEY_ENGINE = "engine"
_KEY_BASE_MODEL_ID = "base_model_id"
_KEY_ADAPTER_REF = "adapter_ref"

# SSH command timeouts (seconds)
_SETUP_CMD_TIMEOUT_S = 120  # mkdir / rm -rf skeleton commands
_QUICK_CMD_TIMEOUT_S = 30  # fast one-liners (container start probe, cleanup)

# ls output truncation in diagnostics
_DIR_LISTING_MAX_CHARS = 800


class SingleNodeInferenceProvider(IInferenceProvider):
    """Deploy vLLM inference endpoint on a single SSH-accessible node."""

    _CONTAINER_NAME = VLLM_INFERENCE_CONTAINER_NAME

    def __init__(self, *, config: PipelineConfig, secrets: Secrets):
        self._cfg = config
        self._secrets = secrets

        self._inf_cfg = config.inference

        # NEW (v3): Get providers.single_node config
        provider_cfg_raw = self._cfg.get_provider_config(PROVIDER_SINGLE_NODE)
        from src.config.providers.single_node import SingleNodeConfig

        self._provider_cfg = SingleNodeConfig(**provider_cfg_raw)

        # Access inference-specific settings via .inference
        self._serve_cfg = self._provider_cfg.inference.serve

        # SSH config from connect.ssh (single source of truth)
        self._ssh_cfg = self._provider_cfg.connect.ssh

        self._engine_cfg = self._inf_cfg.engines.vllm
        self._engine = VLLMEngine()

        self._ssh_client: SSHClient | None = None
        self._endpoint_info: EndpointInfo | None = None
        self._run_id: str | None = None
        self._mlflow_manager: Any | None = None  # Optional MLflow integration

    # ---------------------------------------------------------------------
    # Interface properties
    # ---------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return PROVIDER_SINGLE_NODE

    @property
    def provider_type(self) -> str:
        return PROVIDER_SINGLE_NODE

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def deploy(
        self,
        model_source: str,
        *,
        run_id: str,
        base_model_id: str,
        trust_remote_code: bool = False,
        lora_path: str | None = None,
        quantization: str | None = None,
        keep_running: bool = False,  # noqa: ARG002
    ) -> Result[EndpointInfo, InferenceError]:
        self._run_id = run_id

        # MVP safety: we do not support native LoRA in vLLM yet.
        if not self._inf_cfg.common.lora.merge_before_deploy:
            return Err(
                InferenceError(
                    message=(
                        "inference.common.lora.merge_before_deploy=false is not supported in MVP. "
                        "Set it to true (merge) to serve the trained adapter reliably."
                    ),
                    code="SINGLENODE_LORA_MERGE_REQUIRED",
                )
            )

        ssh_res = self._connect_ssh()
        if ssh_res.is_failure():
            err = ssh_res.unwrap_err()
            return Err(InferenceError(message=str(err), code="SINGLENODE_SSH_CONNECT_FAILED"))

        assert self._ssh_client is not None
        ssh = self._ssh_client

        # Pre-flight checks: docker + nvidia runtime + disk + GPU visibility
        health = SingleNodeHealthCheck(ssh).run_all_checks(
            workspace_path=self._serve_cfg.workspace,
        )
        if not health.passed:
            errs = "; ".join(health.errors or ["Unknown error"])
            return Err(
                InferenceError(
                    message=f"Inference node health checks failed: {errs}",
                    code="SINGLENODE_HEALTH_CHECK_FAILED",
                )
            )

        workspace = self._serve_cfg.workspace.rstrip("/")
        hf_cache = f"{workspace}/hf_cache"

        # Canonical run directory name (single source of truth): run_YYYYMMDD_HHMMSS_id5
        run_dir_name = run_id
        run_dir = f"{workspace}/runs/{run_dir_name}"
        adapter_dir = f"{run_dir}/adapter"
        merged_dir = f"{run_dir}/model"

        # Prepare workspace dirs
        for d in (workspace, hf_cache, f"{workspace}/runs", run_dir):
            ok, dir_err = ssh.create_directory(d)
            if not ok:
                return Err(
                    InferenceError(
                        message=f"Failed to create remote directory '{d}': {dir_err}",
                        code="SINGLENODE_DIR_CREATE_FAILED",
                    )
                )

        # Resolve adapter reference
        adapter_ref = lora_path or model_source
        adapter_ref_for_merge = adapter_ref

        # If adapter is local on the orchestrator machine, upload to remote run dir.
        local_adapter = Path(adapter_ref).expanduser()
        if local_adapter.exists():
            if not local_adapter.is_dir():
                return Err(
                    InferenceError(
                        message=f"Adapter local path must be a directory, got: {local_adapter}",
                        code="SINGLENODE_ADAPTER_NOT_DIR",
                    )
                )

            ok, dir_err = ssh.create_directory(adapter_dir)
            if not ok:
                return Err(
                    InferenceError(
                        message=f"Failed to create remote adapter dir '{adapter_dir}': {dir_err}",
                        code="SINGLENODE_DIR_CREATE_FAILED",
                    )
                )

            up = ssh.upload_directory(local_path=local_adapter, remote_path=adapter_dir)
            if up.is_failure():
                up_err = up.unwrap_err()
                return Err(
                    InferenceError(
                        message=f"Failed to upload adapter to inference node: {up_err}",
                        code="SINGLENODE_ADAPTER_UPLOAD_FAILED",
                    )
                )
            adapter_ref_for_merge = adapter_dir

        # Merge adapter into base model on inference node (two-container strategy).
        # Uses ephemeral merge container with transformers+peft.
        merge_res = self._run_merge_container(
            ssh=ssh,
            base_model=base_model_id,
            adapter_path=adapter_ref_for_merge,
            output_path=merged_dir,
            cache_dir=hf_cache,
            trust_remote_code=trust_remote_code,
        )
        if merge_res.is_failure():
            merge_err = merge_res.unwrap_err()
            return Err(
                merge_err
                if isinstance(merge_err, InferenceError)
                else InferenceError(message=str(merge_err), code="SINGLENODE_MERGE_FAILED")
            )

        # Engine config for this deployment (avoid mutating global PipelineConfig)
        engine_cfg = self._engine_cfg.model_copy()
        if quantization is not None:
            engine_cfg.quantization = quantization

        # Start vLLM container (idempotent: replace container)
        container_model_path = f"/workspace/runs/{run_dir_name}/model"

        start_res = self._start_vllm_container(
            ssh=ssh,
            engine_cfg=engine_cfg,
            workspace_host_path=workspace,
            model_path_in_container=container_model_path,
        )
        if start_res.is_failure():
            start_err = start_res.unwrap_err()
            return Err(
                start_err
                if isinstance(start_err, InferenceError)
                else InferenceError(message=str(start_err), code="SINGLENODE_VLLM_START_FAILED")
            )

        # Endpoint is bound on node to serve.host:serve.port, but MVP expects local access via SSH tunnel.
        port = self._serve_cfg.port
        endpoint_url = f"http://127.0.0.1:{port}/v1"
        health_url = f"http://127.0.0.1:{port}/v1/models"

        self._endpoint_info = EndpointInfo(
            endpoint_url=endpoint_url,
            api_type="openai_compatible",
            provider_type=self.provider_type,
            engine="vllm",
            model_id=base_model_id,
            health_url=health_url,
            resource_id=self._CONTAINER_NAME,
        )

        # Save remote runtime metadata (best-effort, no secrets)
        _ = self._write_runtime_json_best_effort(
            ssh=ssh,
            run_dir=run_dir,
            runtime={
                "run_id": run_id,
                "provider": self.provider_type,
                _KEY_ENGINE: "vllm",
                "container_name": self._CONTAINER_NAME,
                "serve_image": self._engine_cfg.serve_image,
                "merge_image": self._engine_cfg.merge_image,
                "host_bind": self._serve_cfg.host,
                "port": port,
                "workspace": workspace,
                "merged_model_dir": merged_dir,
                "config_hash": self._sha12(
                    json.dumps(
                        {
                            _KEY_BASE_MODEL_ID: base_model_id,
                            _KEY_ADAPTER_REF: adapter_ref,
                            _KEY_ENGINE: self._engine_cfg.model_dump(mode="python", exclude_none=True),
                        },
                        sort_keys=True,
                    )
                ),
            },
        )

        logger.info(f"✅ Inference deployed (single_node/vLLM): {endpoint_url} (use SSH tunnel)")
        return Ok(self._endpoint_info)

    def set_event_logger(self, event_logger: InferenceEventLogger | None) -> None:
        # Backward-compatible internal name (used across provider internals)
        self._mlflow_manager = event_logger  # type: ignore[assignment]

    def get_pipeline_readiness_mode(self) -> PipelineReadinessMode:
        return PipelineReadinessMode.WAIT_FOR_HEALTHY

    def collect_startup_logs(self, *, local_path: Path) -> None:
        """Collect logs from inference container (best-effort)."""
        try:
            if not self._ssh_client:
                return
            logs_res = docker_logs(self._ssh_client, container_name=self._CONTAINER_NAME, timeout_seconds=10)
            if logs_res.is_ok():
                stdout = logs_res.unwrap()
                if stdout:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_text(stdout, encoding=_ENCODING_UTF8)
        except Exception as e:
            logger.debug(f"Failed to collect inference logs: {e}")

    def build_inference_artifacts(
        self, *, ctx: InferenceArtifactsContext
    ) -> Result[InferenceArtifacts, InferenceError]:
        ssh_cfg = self._ssh_cfg
        ssh_block = {
            "alias": ssh_cfg.alias,
            "host": ssh_cfg.host,
            "port": ssh_cfg.port,
            "user": ssh_cfg.user,
            "key_path": ssh_cfg.key_path,
            "key_env": ssh_cfg.key_env,
        }

        alias = ssh_block.get("alias")
        if isinstance(alias, str) and alias:
            target = alias
        else:
            host = ssh_block.get("host") or "<SSH_HOST>"
            user = ssh_block.get("user") or "<SSH_USER>"
            target = f"{user}@{host}"
        tunnel_hint = f"ssh -L {int(self._serve_cfg.port)}:127.0.0.1:{int(self._serve_cfg.port)} {target}"

        workspace = self._serve_cfg.workspace.rstrip("/")
        port = int(self._serve_cfg.port)

        manifest: dict[str, Any] = {
            "run_name": ctx.run_name,
            "mlflow_run_id": ctx.mlflow_run_id,
            "provider": self.provider_type,
            _KEY_ENGINE: self._inf_cfg.engine,
            "ssh": ssh_block,
            "docker": {
                "merge_image": self._engine_cfg.merge_image,
                "serve_image": self._engine_cfg.serve_image,
                "container_name": VLLM_INFERENCE_CONTAINER_NAME,
                "host_bind": self._serve_cfg.host,
                "port": port,
                "workspace": workspace,
                "hf_cache_path": f"{workspace}/hf_cache",
            },
            "model": {
                _KEY_BASE_MODEL_ID: self._cfg.model.name,
                _KEY_ADAPTER_REF: ctx.model_source,
                "merged_model_path": f"{workspace}/runs/{ctx.run_name}/model",
            },
            "llm": self._resolve_llm_manifest_block(),
            "config_hash": self._sha12(
                json.dumps(
                    {
                        _KEY_BASE_MODEL_ID: self._cfg.model.name,
                        _KEY_ADAPTER_REF: ctx.model_source,
                        _KEY_ENGINE: self._engine_cfg.model_dump(mode="python", exclude_none=True),
                        "serve": self._serve_cfg.model_dump(mode="python", exclude_none=True),
                    },
                    sort_keys=True,
                )
            ),
            "endpoint": {
                "client_base_url": ctx.endpoint.endpoint_url,
                "health_url": ctx.endpoint.health_url,
                "tunnel_hint": tunnel_hint,
            },
        }

        return Ok(
            InferenceArtifacts(
                manifest=manifest,
                chat_script=_CHAT_SCRIPT,
                readme=_render_readme(
                    manifest_filename=INFERENCE_MANIFEST_FILENAME, endpoint_url=ctx.endpoint.endpoint_url
                ),
            )
        )

    def undeploy(self) -> Result[None, InferenceError]:
        if not self._ssh_client:
            return Ok(None)

        _ = docker_rm_force(self._ssh_client, container_name=self._CONTAINER_NAME, timeout_seconds=60)
        self._endpoint_info = None
        return Ok(None)

    def health_check(self) -> Result[bool, InferenceError]:
        if not self._ssh_client:
            return Err(
                InferenceError(
                    message="SSH client not initialized (deploy was not called)",
                    code="SINGLENODE_NOT_DEPLOYED",
                )
            )

        # IMPORTANT: Check on REMOTE host, not localhost
        # Container binds to 0.0.0.0 inside, but we access via host's 127.0.0.1
        host = self._serve_cfg.host  # Should be 127.0.0.1 on remote
        port = self._serve_cfg.port
        cmd = self._engine.build_healthcheck_command(host=host, port=port)
        ok, stdout, stderr = self._ssh_client.exec_command(cmd, timeout=10, silent=True)

        if not ok:
            return Err(
                InferenceError(
                    message=f"Health check command failed: {stderr[:LOG_OUTPUT_SHORT_CHARS]}",
                    code="SINGLENODE_HEALTH_CHECK_COMMAND_FAILED",
                )
            )

        # Check output: "1" = ready, "0" = not ready
        if stdout.strip() == "1":
            return Ok(True)

        return Ok(False)  # Service not ready yet

    def get_capabilities(self) -> InferenceCapabilities:
        return InferenceCapabilities(
            provider_type=self.provider_type,
            supported_engines=["vllm"],
            supports_lora=True,  # via merge_before_deploy
        )

    def get_endpoint_info(self) -> EndpointInfo | None:
        return self._endpoint_info

    def activate_for_eval(self) -> Result[str, InferenceError]:
        """
        Single-node: endpoint is already live after deploy() — return existing URL.

        No additional startup needed; SSH tunnel and vLLM container were started
        during deploy(). Evaluation can immediately use the endpoint.
        """
        endpoint = self._endpoint_info
        if endpoint is None:
            return Err(
                InferenceError(
                    message="single_node: activate_for_eval called but endpoint is not deployed. Call deploy() first.",
                    code="SINGLENODE_NOT_DEPLOYED",
                )
            )
        logger.info(f"[EVAL] single_node endpoint ready for evaluation: {endpoint.endpoint_url}")
        return Ok(endpoint.endpoint_url)

    def deactivate_after_eval(self) -> Result[None, InferenceError]:
        """
        Single-node: no-op — endpoint lifecycle is managed by the user (via generated scripts).

        We do NOT stop the endpoint here because:
        - The user may want to interact with it after the pipeline.
        - The generated stop-script handles teardown on user request.
        """
        logger.info("[EVAL] single_node: deactivate_after_eval is a no-op (endpoint managed externally)")
        return Ok(None)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    @staticmethod
    def _sha12(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:SHA12_LEN]

    def _resolve_llm_manifest_block(self) -> dict[str, Any]:
        """
        Resolve the system prompt once and return the llm manifest block.

        The resolved prompt text (not path) is stored in the manifest so that
        the chat script on the remote machine does not need MLflow access.
        system_prompt_source carries audit metadata (origin type + name/path + version).
        """
        from src.evaluation.system_prompt import SystemPromptLoader

        llm_cfg = self._provider_cfg.inference.llm
        mlflow_cfg = getattr(getattr(self._cfg, "experiment_tracking", None), "mlflow", None)
        gateway = getattr(self._mlflow_manager, "_gateway", None) if self._mlflow_manager is not None else None

        try:
            result = SystemPromptLoader.load(llm_cfg, mlflow_cfg=mlflow_cfg, gateway=gateway)
        except ValueError as exc:
            logger.error(f"[INFERENCE] System prompt configuration error: {exc}")
            result = None

        return {
            "system_prompt": result.text if result else None,
            "system_prompt_source": result.source if result else None,
        }

    @staticmethod
    def _looks_like_unresolved_env(value: str) -> bool:
        s = value.strip()
        return s.startswith("${") and s.endswith("}")

    def _connect_ssh(self) -> Result[None, InferenceError]:
        """
        Connect to inference node via SSH.

        NEW (v3): SSH config comes from providers.single_node.connect.ssh
        No fallback logic needed - single source of truth.
        """
        if self._ssh_client is not None:
            return Ok(None)

        try:
            ssh_cfg = self._ssh_cfg

            # Resolve key path via key_env if present
            key_path = ssh_cfg.key_path
            if not key_path and ssh_cfg.key_env:
                key_path = os.environ.get(ssh_cfg.key_env)

            host = ssh_cfg.alias or ssh_cfg.host
            if not host:
                return Err(
                    InferenceError(
                        message="SSH host is not configured in providers.single_node.connect.ssh",
                        code="SINGLENODE_SSH_HOST_NOT_CONFIGURED",
                    )
                )

            if self._looks_like_unresolved_env(host):
                return Err(
                    InferenceError(
                        message=f"SSH host looks like unresolved env placeholder: {host}",
                        code="SINGLENODE_SSH_HOST_UNRESOLVED_ENV",
                    )
                )

            username = None if ssh_cfg.alias else ssh_cfg.user
            if username and self._looks_like_unresolved_env(username):
                return Err(
                    InferenceError(
                        message=f"SSH user looks like unresolved env placeholder: {username}",
                        code="SINGLENODE_SSH_USER_UNRESOLVED_ENV",
                    )
                )

            self._ssh_client = SSHClient(
                host=host,
                port=ssh_cfg.port,
                username=username,
                key_path=key_path if key_path else None,
                connect_timeout=int(ssh_cfg.connect_settings.timeout_seconds),
            )

            ok, err = self._ssh_client.test_connection(
                max_retries=ssh_cfg.connect_settings.max_retries,
                retry_delay=ssh_cfg.connect_settings.retry_delay_seconds,
            )
            if not ok:
                self._ssh_client = None
                return Err(
                    InferenceError(
                        message=f"SSH connection failed: {err}",
                        code="SINGLENODE_SSH_CONNECT_FAILED",
                    )
                )

            return Ok(None)
        except Exception as e:
            self._ssh_client = None
            return Err(InferenceError(message=f"SSH initialization failed: {e!s}", code="SINGLENODE_SSH_INIT_FAILED"))

    @staticmethod
    def _ensure_docker_image(
        *,
        ssh: SSHClient,
        image: str,
    ) -> Result[None, InferenceError]:
        res = ensure_docker_image(ssh=ssh, image=image, pull_timeout_seconds=PULL_TIMEOUT)
        if res.is_failure():
            err = res.unwrap_err()
            return Err(InferenceError(message=str(err), code="SINGLENODE_DOCKER_IMAGE_FAILED"))
        return Ok(None)

    def _merge_adapter_remote(
        self,
        *,
        ssh: SSHClient,
        base_model_id: str,
        adapter_ref: str,
        merged_dir: str,
        hf_cache_dir: str,
        trust_remote_code: bool,
    ) -> Result[None, InferenceError]:
        """
        DEPRECATED: Host-based merge (requires transformers+peft on inference node).

        This method is kept for reference but should NOT be used in production.
        Use _run_merge_container() instead for two-container strategy.
        """
        # Ensure dependencies exist (fail-fast with actionable hint)
        deps_cmd = "python3 -c \"import transformers, peft; print('OK')\""
        ok, _stdout, stderr = ssh.exec_command(deps_cmd, timeout=60, silent=True)
        if not ok:
            return Err(
                InferenceError(
                    message=(
                        "LoRA merge requires python deps on inference node: 'transformers' and 'peft'. "
                        f"Install them or switch to a custom inference image. stderr={stderr[:LOG_OUTPUT_SHORT_CHARS]}"
                    ),
                    code="SINGLENODE_MERGE_DEPS_MISSING",
                )
            )

        # Clean merged_dir
        ssh.exec_command(f"rm -rf {merged_dir} && mkdir -p {merged_dir}", timeout=_SETUP_CMD_TIMEOUT_S, silent=True)
        ssh.exec_command(f"mkdir -p {hf_cache_dir}", timeout=_QUICK_CMD_TIMEOUT_S, silent=True)

        payload = {
            _KEY_BASE_MODEL_ID: base_model_id,
            _KEY_ADAPTER_REF: adapter_ref,
            "merged_dir": merged_dir,
            "hf_cache_dir": hf_cache_dir,
            "trust_remote_code": trust_remote_code,
        }
        payload_b64 = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")

        # NOTE: avoid leaking secrets into generated scripts/manifest; here we do pass HF_TOKEN to remote env.
        # SSHClient masks HF_TOKEN in logs.
        hf_token = str(self._secrets.hf_token)
        cmd = f"""
HF_TOKEN="{hf_token}" python3 - <<'PY'
import base64
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

cfg = json.loads(base64.b64decode("{payload_b64}").decode("utf-8"))
base_model_id = cfg["base_model_id"]
adapter_ref = cfg["adapter_ref"]
merged_dir = cfg["merged_dir"]
cache_dir = cfg["hf_cache_dir"]
trust_remote_code = bool(cfg.get("trust_remote_code", False))

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(merged_dir, exist_ok=True)

# Tokenizer from base model (adapter may not include it)
tok = AutoTokenizer.from_pretrained(
    base_model_id,
    trust_remote_code=trust_remote_code,
    cache_dir=cache_dir,
)

base = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=trust_remote_code,
    cache_dir=cache_dir,
    low_cpu_mem_usage=True,
    device_map="auto",
)

model = PeftModel.from_pretrained(base, adapter_ref)
merged = model.merge_and_unload()
merged.save_pretrained(merged_dir, safe_serialization=True)
tok.save_pretrained(merged_dir)
print("MERGE_OK")
PY
""".strip()

        ok, stdout, stderr = ssh.exec_command(cmd, timeout=MERGE_TIMEOUT, silent=False)
        if not ok or "MERGE_OK" not in stdout:
            return Err(
                InferenceError(
                    message=f"LoRA merge failed: {stderr[:LOG_OUTPUT_LONG_CHARS] if stderr else stdout[:LOG_OUTPUT_LONG_CHARS]}",
                    code="SINGLENODE_MERGE_FAILED",
                )
            )
        return Ok(None)

    def _run_merge_container(
        self,
        *,
        ssh: SSHClient,
        base_model: str,
        adapter_path: str,
        output_path: str,
        cache_dir: str,
        trust_remote_code: bool = False,
    ) -> Result[None, InferenceError]:
        """
        Run ephemeral merge job in Docker container (two-container strategy).

        Args:
            ssh: SSH client for remote execution
            base_model: Base model ID (HuggingFace repo) or local path
            adapter_path: Either a HF repo ID (e.g. "org/adapter") OR a remote *host* path
                inside the inference workspace (e.g. "/home/user/inference/runs/X/adapter").
                If host path is provided, it will be mapped into the container via /workspace mount.
            output_path: Remote *host* output directory for merged model (must be inside workspace_host_path).
                Will be mapped into the container via /workspace mount.
            cache_dir: Remote *host* HuggingFace cache directory (must be inside workspace_host_path).
                Will be mapped into the container via /workspace mount.
            trust_remote_code: Trust remote code when loading model

        Returns:
            Ok(None) if merge succeeded
            Err(str) if merge failed
        """
        merge_image = (self._engine_cfg.merge_image or "").strip()
        if not merge_image:
            return Err(
                InferenceError(
                    message=(
                        "inference.provider='single_node' requires inference.engines.vllm.merge_image "
                        "(two-container strategy: merge container)"
                    ),
                    code="SINGLENODE_MERGE_IMAGE_NOT_CONFIGURED",
                )
            )

        # 1. Ensure merge image is available
        ensure_res = self._ensure_docker_image(
            ssh=ssh,
            image=merge_image,
        )
        if ensure_res.is_failure():
            ensure_err = ensure_res.unwrap_err()
            return Err(
                InferenceError(
                    message=f"Merge image not available: {ensure_err}",
                    code="SINGLENODE_MERGE_IMAGE_PULL_FAILED",
                )
            )

        # 2. Clean output directory
        ssh.exec_command(f"rm -rf {output_path} && mkdir -p {output_path}", timeout=_SETUP_CMD_TIMEOUT_S, silent=True)
        ssh.exec_command(f"mkdir -p {cache_dir}", timeout=_QUICK_CMD_TIMEOUT_S, silent=True)

        # 3. Build Docker command for merge job
        workspace_host_path = self._provider_cfg.inference.serve.workspace.rstrip("/")
        hf_token = str(self._secrets.hf_token)

        workspace_container_path = "/workspace"

        def _to_container_path(host_or_container_path: str) -> str:
            """
            Map a remote host path inside workspace_host_path to a container path inside /workspace.
            If the value already looks like a container path (/workspace/...), returns it as-is.
            """
            p = host_or_container_path.rstrip("/")
            if p == workspace_container_path or p.startswith(workspace_container_path + "/"):
                return p
            if p == workspace_host_path or p.startswith(workspace_host_path + "/"):
                return workspace_container_path + p[len(workspace_host_path) :]
            return host_or_container_path

        # Validate that host paths are actually inside the mounted workspace
        if not (output_path == workspace_host_path or output_path.startswith(workspace_host_path + "/")):
            return Err(
                InferenceError(
                    message="Merge output_path must be inside inference workspace. "
                    f"output_path={output_path!r}, workspace={workspace_host_path!r}",
                    code="INFERENCE_MERGE_INVALID_PATH",
                )
            )
        if not (cache_dir == workspace_host_path or cache_dir.startswith(workspace_host_path + "/")):
            return Err(
                InferenceError(
                    message="Merge cache_dir must be inside inference workspace. "
                    f"cache_dir={cache_dir!r}, workspace={workspace_host_path!r}",
                    code="INFERENCE_MERGE_INVALID_PATH",
                )
            )

        # Map host paths to in-container paths (critical: otherwise artifacts are written to ephemeral FS)
        output_path_in_container = _to_container_path(output_path)
        cache_dir_in_container = _to_container_path(cache_dir)

        # Adapter: if it's a host path, map it; otherwise treat as HF repo ID.
        adapter_arg = adapter_path
        if adapter_path.startswith("/"):
            if adapter_path.startswith(workspace_container_path + "/"):
                adapter_arg = adapter_path
            elif adapter_path.startswith(workspace_host_path + "/") or adapter_path == workspace_host_path:
                adapter_arg = _to_container_path(adapter_path)
            else:
                return Err(
                    InferenceError(
                        message="Merge adapter_path (when absolute path) must be inside inference workspace. "
                        f"adapter_path={adapter_path!r}, workspace={workspace_host_path!r}",
                        code="INFERENCE_MERGE_INVALID_PATH",
                    )
                )

        # Build merge command (conditional trust_flag)
        trust_arg = "--trust-remote-code" if trust_remote_code else ""
        container_name = f"helix-merge-{self._run_id}"

        # ------------------------------------------------------------------
        # IMPORTANT: keep HF token name unified (HF_TOKEN only).
        #
        # Instead of relying on library-specific env var discovery in the merge image,
        # we upload our merge script and run it explicitly inside the container.
        # The script reads HF_TOKEN and passes it as `token=...` (with legacy fallback).
        # ------------------------------------------------------------------
        # NOTE: Do NOT rely on a fixed `.parents[N]` depth here.
        # This provider module can be moved during refactors (e.g. pipeline/ → providers/).
        # We locate the merge script by walking up until we find the expected repo layout.
        local_merge_script: Path | None = None
        here = Path(__file__).resolve()
        for p in (here, *here.parents):
            candidate = p / "docker" / "inference" / "scripts" / "merge_lora.py"
            if candidate.exists():
                local_merge_script = candidate
                break

        if local_merge_script is None:
            # Best-effort fallback path for error message.
            local_merge_script = here.parent / "docker" / "inference" / "scripts" / "merge_lora.py"
        remote_merge_script_host = f"{Path(output_path).parent}/merge_lora.py"
        merge_script_in_container = _to_container_path(remote_merge_script_host)

        if local_merge_script.exists():
            try:
                up_res = ssh.upload_file(str(local_merge_script), remote_merge_script_host, verify=False)
                # Real SSHClient returns (bool, str). In tests, this may be a MagicMock.
                ok_up = True
                err_up = ""
                if isinstance(up_res, tuple) and len(up_res) == 2:
                    ok_up, err_up = bool(up_res[0]), str(up_res[1])
                if not ok_up:
                    return Err(
                        InferenceError(
                            message=f"Failed to upload merge script to inference node: {err_up}",
                            code="SINGLENODE_MERGE_SCRIPT_UPLOAD_FAILED",
                        )
                    )
            except Exception as e:
                return Err(
                    InferenceError(
                        message=f"Failed to upload merge script to inference node: {e!s}",
                        code="SINGLENODE_MERGE_SCRIPT_UPLOAD_FAILED",
                    )
                )
        else:
            return Err(
                InferenceError(
                    message=f"Merge script not found locally: {local_merge_script}",
                    code="SINGLENODE_MERGE_SCRIPT_NOT_FOUND",
                )
            )

        merge_cmd = (
            f"docker run --detach "  # Changed from --rm to --detach (no auto-remove for log collection)
            f"--name {container_name} "
            f"--gpus all "
            f"-v {workspace_host_path}:/workspace "
            f'-e HF_TOKEN="{hf_token}" '
            f"-e HF_HOME={cache_dir_in_container} "
            f"-e HUGGINGFACE_HUB_CACHE={cache_dir_in_container} "
            f"-e TRANSFORMERS_CACHE={cache_dir_in_container} "
            f"--entrypoint python3 "
            f"{merge_image} "
            f"{merge_script_in_container} "
            f"--base-model {base_model} "
            f"--adapter {adapter_arg} "
            f"--output {output_path_in_container} "
            f"--cache-dir {cache_dir_in_container} "
            f"{trust_arg}"
        ).strip()

        logger.info(f"Starting merge job: {base_model} + {adapter_path} → {output_path}")
        logger.info(f"Merge container: {merge_image}")

        # Log merge start event
        merge_start_time = time.time()
        if self._mlflow_manager:
            self._mlflow_manager.log_event_start(
                f"Merge started: {base_model} + adapter",
                category=_KEY_INFERENCE,
                source=SOURCE_SINGLE_NODE_INFERENCE,
                base_model=base_model,
            )

        # 4. Start merge container in background
        ok, _container_id, stderr = ssh.exec_command(merge_cmd, timeout=_QUICK_CMD_TIMEOUT_S, silent=False)
        if not ok:
            logger.error(f"Failed to start merge container: {stderr[:LOG_OUTPUT_LONG_CHARS]}")
            if self._mlflow_manager:
                self._mlflow_manager.log_event_error(
                    f"Merge failed to start: {stderr[:LOG_OUTPUT_SHORT_CHARS]}",
                    category=_KEY_INFERENCE,
                    source=SOURCE_SINGLE_NODE_INFERENCE,
                )
            return Err(
                InferenceError(
                    message=f"Merge container start failed: {stderr[:LOG_OUTPUT_LONG_CHARS]}",
                    code="SINGLENODE_MERGE_CONTAINER_START_FAILED",
                )
            )

        # 5. Poll logs while container is running
        poll_interval = 2  # Merge log collection interval (seconds)
        try:
            log_dir = get_run_log_dir()
        except RuntimeError:
            # Unit tests may run without PipelineOrchestrator init_run_logging().
            log_dir = Path("/tmp/helix_logs")
        merge_log_path = log_dir / "inference" / "merge.log"
        merge_log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"📝 Collecting merge logs → {merge_log_path}")

        merge_success = False
        exit_code = None

        while True:
            # Check if container is still running
            is_running = docker_is_container_running(ssh, name_filter=container_name, timeout_seconds=5)

            # Collect current logs
            logs_res = docker_logs(ssh, container_name=container_name, timeout_seconds=10)
            if logs_res.is_ok():
                logs_content = logs_res.unwrap()
                if logs_content:
                    # Best-effort: local FS issues must not break merge job.
                    try:
                        merge_log_path.parent.mkdir(parents=True, exist_ok=True)
                        merge_log_path.write_text(logs_content, encoding=_ENCODING_UTF8)
                    except OSError as e:
                        logger.warning(f"Failed to write merge logs to {merge_log_path}: {e}")
                    # Check for success marker in logs
                    if "MERGE_SUCCESS" in logs_content:
                        merge_success = True

            # If container stopped, get exit code and break
            if not is_running:
                exit_res = docker_container_exit_code(ssh, container_name=container_name, timeout_seconds=5)
                if exit_res.is_ok():
                    exit_code = exit_res.unwrap()
                break

            # Wait before next poll
            time.sleep(poll_interval)

        # 6. Cleanup container
        _ = docker_rm_force(ssh, container_name=container_name, timeout_seconds=_QUICK_CMD_TIMEOUT_S)

        # 7. Validate result
        logger.info(f"📥 Merge logs saved: {merge_log_path}")

        merge_duration = time.time() - merge_start_time

        if exit_code != 0:
            logger.error(f"Merge job failed with exit code {exit_code}")
            if self._mlflow_manager:
                self._mlflow_manager.log_event_error(
                    f"Merge failed with exit code {exit_code}",
                    category=_KEY_INFERENCE,
                    source=SOURCE_SINGLE_NODE_INFERENCE,
                    duration_seconds=merge_duration,
                    exit_code=exit_code,
                )
            return Err(
                InferenceError(
                    message=f"Merge container failed with exit code {exit_code}. Check {merge_log_path}",
                    code="SINGLENODE_MERGE_CONTAINER_FAILED",
                )
            )

        if not merge_success:
            logger.error("Merge job completed but no success marker found")
            if self._mlflow_manager:
                self._mlflow_manager.log_event_error(
                    "Merge completed without success marker",
                    category=_KEY_INFERENCE,
                    source=SOURCE_SINGLE_NODE_INFERENCE,
                    duration_seconds=merge_duration,
                )
            return Err(
                InferenceError(
                    message=f"Merge job did not complete successfully (missing MERGE_SUCCESS marker). Check {merge_log_path}",
                    code="SINGLENODE_MERGE_NO_SUCCESS_MARKER",
                )
            )

        # 8. Verify output artifacts exist on HOST (guardrail against wrong path mapping)
        verify_cmd = f"test -f {output_path}/config.json && echo OK || echo MISSING"
        _ok_v, v_stdout, _v_stderr = ssh.exec_command(verify_cmd, timeout=10, silent=True)
        if "OK" not in v_stdout:
            _ok_ls, ls_stdout, _ls_stderr = ssh.exec_command(f"ls -lah {output_path} || true", timeout=10, silent=True)
            return Err(
                InferenceError(
                    message=(
                        "Merge container reported success but merged model artifacts were not found on host. "
                        f"Expected config.json at: {output_path}/config.json. "
                        f"Directory listing:\n{ls_stdout[:_DIR_LISTING_MAX_CHARS]}"
                    ),
                    code="SINGLENODE_MERGE_ARTIFACTS_NOT_FOUND",
                )
            )

        logger.info(f"✅ LoRA merge completed successfully: {output_path}")

        # Log merge completion event
        if self._mlflow_manager:
            self._mlflow_manager.log_event_complete(
                f"Merge completed successfully ({merge_duration:.1f}s)",
                category=_KEY_INFERENCE,
                source=SOURCE_SINGLE_NODE_INFERENCE,
                duration_seconds=merge_duration,
                base_model=base_model,
            )

        return Ok(None)

    def _start_vllm_container(
        self,
        *,
        ssh: SSHClient,
        engine_cfg: InferenceVLLMEngineConfig,
        workspace_host_path: str,
        model_path_in_container: str,
    ) -> Result[None, InferenceError]:
        host_bind = self._provider_cfg.inference.serve.host
        port = self._provider_cfg.inference.serve.port

        serve_image = (engine_cfg.serve_image or "").strip()
        if not serve_image:
            return Err(
                InferenceError(
                    message=(
                        "inference.provider='single_node' requires inference.engines.vllm.serve_image "
                        "(two-container strategy: serve container)"
                    ),
                    code="SINGLENODE_SERVE_IMAGE_NOT_CONFIGURED",
                )
            )

        # Ensure serve image is available
        ensure_res = self._ensure_docker_image(
            ssh=ssh,
            image=serve_image,
        )
        if ensure_res.is_failure():
            ensure_err = ensure_res.unwrap_err()
            return Err(
                InferenceError(
                    message=f"Serve image not available: {ensure_err}",
                    code="SINGLENODE_SERVE_IMAGE_PULL_FAILED",
                )
            )

        run_cmd = self._engine.build_docker_run_command(
            cfg=engine_cfg,
            image=serve_image,
            container_name=self._CONTAINER_NAME,
            host_bind=host_bind,
            port=port,
            workspace_host_path=workspace_host_path,
            model_path_in_container=model_path_in_container,
        )

        # Stop old container (if exists) and start new one
        _ = docker_rm_force(ssh, container_name=self._CONTAINER_NAME, timeout_seconds=60)

        ok, _stdout, stderr = ssh.exec_command(run_cmd, timeout=PULL_TIMEOUT, silent=False)
        if not ok:
            _ = docker_logs(
                ssh,
                container_name=self._CONTAINER_NAME,
                tail=LOG_OUTPUT_SHORT_CHARS,
                timeout_seconds=60,
            )
            return Err(
                InferenceError(
                    message=f"Failed to start vLLM container: {stderr[:LOG_OUTPUT_LONG_CHARS]}",
                    code="SINGLENODE_VLLM_START_FAILED",
                )
            )
        return Ok(None)

    @staticmethod
    def _write_runtime_json_best_effort(
        *,
        ssh: SSHClient,
        run_dir: str,
        runtime: dict[str, Any],
    ) -> bool:
        try:
            content = json.dumps(runtime, ensure_ascii=False, indent=2)
            b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
            cmd = f"echo '{b64}' | base64 -d > {run_dir}/runtime.json"
            ok, _stdout, _stderr = ssh.exec_command(cmd, timeout=_QUICK_CMD_TIMEOUT_S, silent=True)
            return ok
        except Exception:
            return False
