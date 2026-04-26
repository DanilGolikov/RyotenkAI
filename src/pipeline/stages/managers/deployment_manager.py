"""
Training Deployment Manager - Manages training deployment on RunPod.

Handles file uploads, dependency installation, and training execution.
Extracted from RunPodDeployer as part of SOLID refactoring (Phase 4/5).
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from src.infrastructure.mlflow.uri_resolver import resolve_mlflow_uris
from src.pipeline.stages.constants import PipelineContextKeys
from src.pipeline.stages.managers.deployment.code_syncer import CodeSyncer
from src.pipeline.stages.managers.deployment.dependency_installer import DependencyInstaller
from src.pipeline.stages.managers.deployment.file_uploader import FileUploader
from src.pipeline.stages.managers.deployment.provider_config import (
    get_active_provider_name,
    get_cloud_training_cfg,
    get_single_node_training_cfg,
    is_single_node_provider,
)
from src.pipeline.stages.managers.deployment_constants import (
    DEPLOYMENT_CONFIG_PATH,
    DEPLOYMENT_CONTAINER_NAME_MAX_LEN,
    DEPLOYMENT_DOCKER_VALUE,
    DEPLOYMENT_ERROR_TRUNCATE,
    DEPLOYMENT_LAUNCH_TIMEOUT,
    DEPLOYMENT_LOG_TRUNCATE,
    DEPLOYMENT_MARKER_EXISTS,
    DEPLOYMENT_MODE_KEY,
    DEPLOYMENT_SCRIPT_CHMOD_TIMEOUT,
    DEPLOYMENT_TRAINING_START_TIMEOUT,
    DEPLOYMENT_VERIFY_TIMEOUT,
)
from src.providers.training.interfaces import TrainingScriptHooks
from src.utils.docker import docker_is_container_running
from src.utils.logger import logger
from src.utils.result import AppError, Err, Ok, ProviderError, Result

if TYPE_CHECKING:
    from src.providers.training.interfaces import IGPUProvider
    from src.utils.config import PipelineConfig, Secrets
    from src.utils.ssh_client import SSHClient


TRAINING_START_PROBE_MIN_TIMEOUT_SECONDS = 20


class TrainingDeploymentManager:
    """
    Manages training deployment on GPU servers.

    Responsibilities:
    - Upload datasets and training scripts
    - Verify dependencies (prebuilt Docker images)
    - Start training process in background
    - File operations via SSH

    Environment:
    - All providers use Docker images with pre-installed packages
    - RunPod: Pod created with image
    - single_node: Docker container started via SSH

    Does NOT handle:
    - Pod creation (RunPodAPIClient)
    - Pod lifecycle management (PodLifecycleManager)
    - SSH connection creation (stays in GPUDeployer)
    """

    # Default workspace for cloud providers (RunPod)
    DEFAULT_WORKSPACE = "/workspace"

    # UV installation script
    UV_INSTALL_SCRIPT = "curl -LsSf https://astral.sh/uv/install.sh | sh"

    def __init__(self, config: PipelineConfig, secrets: Secrets):
        """
        Initialize training deployment manager.

        Args:
            config: Pipeline configuration
            secrets: Secrets (for HF_TOKEN, etc.)
        """
        self.config = config
        self.secrets = secrets
        self._workspace = self.DEFAULT_WORKSPACE
        self._code_syncer = CodeSyncer(config=config, secrets=secrets)
        self._file_uploader = FileUploader(config=config, secrets=secrets, code_syncer=self._code_syncer)
        self._deps_installer = DependencyInstaller(config=config, secrets=secrets)
        self._code_syncer.set_workspace(self._workspace)
        self._file_uploader.set_workspace(self._workspace)
        self._deps_installer.set_workspace(self._workspace)
        logger.debug("🚀 TrainingDeploymentManager initialized")

    @property
    def workspace(self) -> str:
        """Remote workspace path where code/configs are deployed."""
        return self._workspace

    def set_workspace(
        self,
        workspace_path: str,
    ) -> None:
        """
        Set workspace path for deployment.

        Args:
            workspace_path: Run directory (e.g., /workspace or /home/user/run_xxx/)
        """
        self._workspace = workspace_path
        self._code_syncer.set_workspace(workspace_path)
        self._file_uploader.set_workspace(workspace_path)
        self._deps_installer.set_workspace(workspace_path)
        logger.debug(f"[DEPLOY] Workspace: {self._workspace}")

    # =========================================================================
    # SOURCE CODE SYNC — delegates to CodeSyncer
    # =========================================================================

    def _sync_source_code(self, ssh_client: SSHClient) -> Result[None, AppError]:
        """Proxy to :meth:`CodeSyncer.sync` — kept until callers migrate."""
        return self._code_syncer.sync(ssh_client)

    # =========================================================================
    # FILES DEPLOY — delegates to FileUploader
    # =========================================================================

    def deploy_files(self, ssh_client: SSHClient, context: dict[str, Any]) -> Result[None, AppError]:
        """Proxy to :meth:`FileUploader.deploy_files`."""
        return self._file_uploader.deploy_files(ssh_client, context)

    # =========================================================================
    # PROVIDER HELPERS — proxies onto deployment.provider_config functions
    # =========================================================================

    def _get_active_provider_name(self) -> str:
        return get_active_provider_name(self.config)

    def _is_single_node_provider(self) -> bool:
        return is_single_node_provider(self.config)

    def _get_single_node_training_cfg(self) -> dict[str, Any]:
        return get_single_node_training_cfg(self.config)

    def _get_cloud_training_cfg(self) -> dict[str, Any]:
        return get_cloud_training_cfg(self.config)

    # =========================================================================
    # DEPENDENCY INSTALLATION — delegates to DependencyInstaller
    # =========================================================================

    def install_dependencies(self, ssh_client: SSHClient) -> Result[None, AppError]:
        """Proxy to :meth:`DependencyInstaller.install`."""
        return self._deps_installer.install(ssh_client)

    @staticmethod
    def _verify_prebuilt_dependencies(ssh_client: SSHClient) -> Result[None, ProviderError]:
        """Proxy to :meth:`DependencyInstaller.verify_prebuilt_dependencies`."""
        return DependencyInstaller.verify_prebuilt_dependencies(ssh_client)

    def _verify_single_node_docker_runtime(self, ssh_client: SSHClient) -> Result[None, AppError]:
        """Proxy to :meth:`DependencyInstaller._verify_single_node_docker_runtime`."""
        return self._deps_installer._verify_single_node_docker_runtime(ssh_client)

    def _ensure_docker_image_present(
        self, ssh_client: SSHClient, *, image: str
    ) -> Result[None, ProviderError]:
        """Proxy to :meth:`DependencyInstaller._ensure_docker_image_present` — used by TrainingLauncher tests."""
        return self._deps_installer._ensure_docker_image_present(ssh_client, image=image)

    # =========================================================================
    # TRAINING EXECUTION
    # =========================================================================

    def _create_env_file(
        self,
        ssh_client: SSHClient,
        context: dict[str, Any] | None = None,
        extra_env_vars: dict[str, str] | None = None,
    ) -> Result[str, ProviderError]:
        """
        Create .env file on remote server with training environment variables.

        ``extra_env_vars`` carries provider-specific variables contributed via
        ``IGPUProvider.prepare_training_script_hooks`` (e.g., RunPod auto-stop
        credentials). They are merged LAST and override any built-in keys.
        """
        # Environment variables
        # Experiment tracking goes to MLflow (report_to=["mlflow"] in TrainingArguments)
        # single_node runs training inside container with run_dir mounted as /workspace
        # cloud providers run directly inside the pod container -> use actual run dir path
        workspace_env = "/workspace" if self._is_single_node_provider() else self._workspace
        env_vars: dict[str, str] = {
            "LOG_LEVEL": "DEBUG",
            "HELIX_WORKSPACE": workspace_env,
            "PYTHONPATH": workspace_env,
        }

        # HF_TOKEN for gated models
        if self.secrets.hf_token:
            env_vars["HF_TOKEN"] = self.secrets.hf_token
            logger.info("HF_TOKEN will be set via .env file")

        # MLflow configuration for nested runs
        mlflow_config = self.config.experiment_tracking.mlflow
        if mlflow_config:
            resolved_uris = resolve_mlflow_uris(mlflow_config, runtime_role="training")

            # Tracking URI - remote server needs to know where to send runs
            if resolved_uris.effective_remote_tracking_uri:
                env_vars["MLFLOW_TRACKING_URI"] = resolved_uris.effective_remote_tracking_uri
                logger.info(f"📊 MLflow tracking URI: {resolved_uris.effective_remote_tracking_uri}")

            # Parent run ID for nested runs (experiment_name comes from synced config!)
            if context and context.get(PipelineContextKeys.MLFLOW_PARENT_RUN_ID):
                env_vars["MLFLOW_PARENT_RUN_ID"] = context[PipelineContextKeys.MLFLOW_PARENT_RUN_ID]
                logger.info(f"📊 MLflow parent run ID: {context[PipelineContextKeys.MLFLOW_PARENT_RUN_ID]}")

            # Timeout settings
            env_vars["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "15"
            env_vars["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "2"
            logger.debug("📊 MLflow timeout: 15s, max retries: 2")

            if mlflow_config.ca_bundle_path:
                env_vars["REQUESTS_CA_BUNDLE"] = mlflow_config.ca_bundle_path
                env_vars["SSL_CERT_FILE"] = mlflow_config.ca_bundle_path
                logger.info(f"📊 MLflow CA bundle: {mlflow_config.ca_bundle_path}")

        # Provider-specific extras merged last so they can override defaults.
        if extra_env_vars:
            env_vars.update(extra_env_vars)

        env_content = "\n".join(f'export {k}="{v}"' for k, v in env_vars.items())
        env_path = f"{self._workspace}/.env"

        create_cmd = f"cat > {env_path} << 'HELIX_ENV_EOF'\n{env_content}\nHELIX_ENV_EOF"
        success, _, stderr = ssh_client.exec_command(
            command=create_cmd, background=False, timeout=DEPLOYMENT_VERIFY_TIMEOUT
        )
        if not success:
            return Err(ProviderError(message=f"Failed to create .env file: {stderr}", code="ENV_FILE_CREATE_FAILED"))

        # Set restrictive permissions
        # NOTE: Some SSH targets have noticeable handshake overhead; keep this best-effort but not too tight.
        ssh_client.exec_command(
            command=f"chmod 600 {env_path}",
            background=False,
            timeout=DEPLOYMENT_SCRIPT_CHMOD_TIMEOUT,
            silent=True,
        )

        logger.info(f"✅ Created .env file ({len(env_vars)} vars)")
        return Ok(env_path)

    def start_training(
        self,
        ssh_client: SSHClient,
        context: dict[str, Any],
        provider: IGPUProvider | None = None,
    ) -> Result[dict[str, Any], AppError]:
        """
        Start training process on remote.

        Docker-only:
        - single_node: starts a Docker container on the host and runs training inside it
        - cloud providers (RunPod): runs training inside the already-running pod container

        ``provider`` is required for cloud providers (carries lifecycle hooks
        such as the RunPod watchdog). Passing ``None`` is equivalent to a
        provider with no customizations — used by tests and single_node.
        """
        logger.info("Starting training in background...")

        if self._is_single_node_provider():
            return self._start_training_docker(ssh_client, context)

        return self._start_training_cloud(ssh_client, context, provider)

    def _start_training_cloud(
        self,
        ssh_client: SSHClient,
        context: dict[str, Any],
        provider: IGPUProvider | None,
    ) -> Result[dict[str, Any], AppError]:
        """
        Start training inside the current environment (cloud pods).

        Assumptions:
        - SSH is connected inside the pod container
        - Docker is not required/expected inside the container
        """
        # Step 1: ask provider for customizations (env vars, pre/post python hooks).
        if provider is not None:
            hooks_result = provider.prepare_training_script_hooks(ssh_client, context)
            if hooks_result.is_err():
                return Err(hooks_result.unwrap_err())  # type: ignore[union-attr]
            hooks = hooks_result.unwrap()
        else:
            hooks = TrainingScriptHooks.empty()

        # Step 2: Create .env file (merging provider-contributed env vars).
        env_result = self._create_env_file(ssh_client, context, extra_env_vars=hooks.env_vars)
        if env_result.is_err():
            return Err(env_result.unwrap_err())  # type: ignore[union-attr]  # already ProviderError

        env_file = env_result.unwrap()
        log_file = f"{self._workspace}/training.log"

        # Config is ALWAYS uploaded to config/pipeline_config.yaml
        remote_config_path = DEPLOYMENT_CONFIG_PATH
        original_config = context.get("config_path", "unknown")
        logger.info(f"📝 Config: {original_config} → {remote_config_path}")

        # Create start script and run via nohup (SSH returns immediately)
        start_script = f"{self._workspace}/start_training.sh"
        module_args = f"-m src.training.run_training --config {remote_config_path}"

        pre_python = hooks.pre_python
        post_python = hooks.post_python

        script_content = f"""#!/bin/bash
set -euo pipefail
cd {self._workspace}
. {env_file}
exec >{log_file} 2>&1

# --- Crash observability env vars (see src/training/run_training.py:_install_crash_observability) ---
# PYTHONUNBUFFERED=1  → disable Python stdout/stderr block buffering so the tail of
#                       training.log is on disk even if the process dies mid-step.
# PYTHONFAULTHANDLER=1 → activate faulthandler early (before any user import) so native
#                       crashes in C extensions (bitsandbytes, flash-attn, CUDA kernels)
#                       leave a Python + C stack trace.
# PYTHONFAULTHANDLER_PATH → sibling file that monitor can tail post-mortem.
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONFAULTHANDLER_PATH={self._workspace}/training.faulthandler.log

PY_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PY_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PY_BIN=python
elif [ -x /opt/conda/bin/python3 ]; then
  PY_BIN=/opt/conda/bin/python3
elif [ -x /opt/conda/bin/python ]; then
  PY_BIN=/opt/conda/bin/python
else
  echo "PYTHON_NOT_FOUND"
  exit 127
fi

{pre_python}
set +e
"$PY_BIN" {module_args}
exit_code=$?
set -e

# Persist exit code so the monitor's post-mortem probe can distinguish
# signal-kill (128+N) from normal Python exceptions (1).
echo "$exit_code $(date -Iseconds 2>/dev/null || date)" > {self._workspace}/TRAINING_EXIT_CODE || true

# If the Python process crashed before notifiers were initialized (e.g. import error
# or native SEGV), there may be no in-Python marker. Create an enriched
# TRAINING_FAILED marker with exit-code, signal name, and the last 50 lines of
# training.log so the monitor can surface something meaningful.
# NB (python side): this block lives inside a Python f-string - the {{ and }}
# pairs below are escapes that render as single literal braces for the bash
# group command. Keep them doubled.
if [ $exit_code -ne 0 ] && [ ! -f {self._workspace}/TRAINING_FAILED ] && [ ! -f {self._workspace}/TRAINING_COMPLETE ]; then
  {{
    echo "exit_code=$exit_code"
    echo "timestamp=$(date -Iseconds 2>/dev/null || date)"
    if [ $exit_code -gt 128 ]; then
      signal_no=$((exit_code - 128))
      signal_name=$(kill -l $signal_no 2>/dev/null || echo "signal-$signal_no")
      echo "signal=$signal_name (signal_no=$signal_no)"
    fi
    echo "--- last 50 lines of training.log ---"
    tail -n 50 {log_file} 2>/dev/null || echo "(training.log unreadable)"
  }} > {self._workspace}/TRAINING_FAILED || true
fi

{post_python}

exit $exit_code
"""

        logger.info(f"📋 Training command: python3|python {module_args}")

        create_script_cmd = f"cat > {start_script} << 'HELIX_SCRIPT_EOF'\n{script_content}HELIX_SCRIPT_EOF"
        success, _, stderr = ssh_client.exec_command(create_script_cmd, timeout=DEPLOYMENT_VERIFY_TIMEOUT)
        if not success:
            return Err(
                ProviderError(message=f"Failed to create start script: {stderr}", code="TRAINING_SCRIPT_CREATE_FAILED")
            )

        # NOTE: `chmod` itself is fast, but the SSH handshake can exceed a 5s budget in real cloud runs.
        success, _, stderr = ssh_client.exec_command(
            f"chmod +x {start_script}", timeout=DEPLOYMENT_SCRIPT_CHMOD_TIMEOUT
        )
        if not success:
            return Err(
                ProviderError(message=f"Failed to chmod start script: {stderr}", code="TRAINING_SCRIPT_CHMOD_FAILED")
            )

        launch_cmd = f"nohup {start_script} </dev/null >/dev/null 2>&1 & disown"
        success, _, stderr = ssh_client.exec_command(launch_cmd, timeout=DEPLOYMENT_LAUNCH_TIMEOUT)
        if not success:
            return Err(ProviderError(message=f"Failed to start training: {stderr}", code="TRAINING_LAUNCH_FAILED"))

        logger.info("✅ Training command executed")

        # ------------------------------------------------------------------
        # Wait for training start (cloud): retry checks for up to training_start_timeout.
        # ------------------------------------------------------------------
        cloud_cfg = self._get_cloud_training_cfg()
        startup_timeout_seconds = DEPLOYMENT_TRAINING_START_TIMEOUT
        _ = cloud_cfg  # retained for potential future use
        poll_interval_seconds = 5
        poll_interval_seconds = max(1, poll_interval_seconds)

        # SSH handshake overhead in cloud can exceed a minimal 10s budget; keep checks more generous.
        # Also keep it bounded: we do a single "probe" command per poll to avoid N*timeout blowups.
        verify_timeout = max(TRAINING_START_PROBE_MIN_TIMEOUT_SECONDS, int(DEPLOYMENT_VERIFY_TIMEOUT))

        last_timeout_details: list[str] = []

        probe_cmd = (
            f"if [ -f {self._workspace}/TRAINING_COMPLETE ]; then echo 'STATUS=COMPLETE'; exit 0; fi; "
            f"if [ -f {self._workspace}/TRAINING_FAILED ]; then echo 'STATUS=FAILED'; "
            f"tail -n 20 {self._workspace}/TRAINING_FAILED 2>/dev/null || true; exit 0; fi; "
            "if ps aux | grep -E 'python.*train' | grep -v grep >/dev/null 2>&1; then echo 'STATUS=RUNNING'; exit 0; fi; "
            f"if [ -f {log_file} ]; then echo 'STATUS=LOG_EXISTS'; exit 0; fi; "
            "echo 'STATUS=NONE'"
        )

        deadline = time.time() + float(startup_timeout_seconds)
        while True:
            remaining_s = deadline - time.time()
            if remaining_s <= 0:
                break
            probe_timeout = max(1, min(verify_timeout, int(remaining_s) + 1))

            ok_probe, out_probe, err_probe = ssh_client.exec_command(
                command=probe_cmd,
                background=False,
                timeout=probe_timeout,
                silent=True,
            )
            if not ok_probe:
                if "Timeout after" in (err_probe or ""):
                    last_timeout_details.append(f"probe_timeout>{verify_timeout}s")
                remaining_s = deadline - time.time()
                if remaining_s <= 0:
                    break
                time.sleep(min(poll_interval_seconds, max(0, int(remaining_s))))
                continue

            lines = (out_probe or "").splitlines()
            status_line = lines[0].strip() if lines else ""
            status = status_line.split("=", 1)[1].strip() if status_line.startswith("STATUS=") else ""

            if status == "COMPLETE":
                logger.info("✅ Training already completed (fast training scenario)")
                return Ok({DEPLOYMENT_MODE_KEY: DEPLOYMENT_DOCKER_VALUE})

            if status == "FAILED":
                details = "\n".join(ln.strip() for ln in lines[1:] if ln.strip())
                err_snippet = (
                    details[:DEPLOYMENT_ERROR_TRUNCATE] if details else "Training failed early. See training.log."
                )
                logger.error(f"❌ Training failed: {err_snippet}")
                return Err(ProviderError(message=f"Training failed: {err_snippet}", code="TRAINING_FAILED_ON_START"))

            if status in {"RUNNING", "LOG_EXISTS"}:
                if status == "RUNNING":
                    logger.info("✅ Training process confirmed running")
                else:
                    logger.info("✅ Training log file exists - assuming training is running")
                return Ok({DEPLOYMENT_MODE_KEY: DEPLOYMENT_DOCKER_VALUE})

            remaining_s = deadline - time.time()
            if remaining_s <= 0:
                break
            time.sleep(min(poll_interval_seconds, max(0, int(remaining_s))))

        # Final diagnostics (best-effort; never raises)
        error_details_parts: list[str] = []
        if last_timeout_details:
            unique = list(dict.fromkeys(last_timeout_details))
            preview = ", ".join(unique[:5])
            more = "" if len(unique) <= 5 else f" (+{len(unique) - 5} more)"
            error_details_parts.append(f"SSH probe timeouts: {preview}{more}")

        # Workspace listing (helps when run dir is missing / wrong path)
        ls_ok, ls_out, _ls_err = ssh_client.exec_command(
            command=f"ls -la {self._workspace} 2>/dev/null | head -50 || true",
            background=False,
            timeout=verify_timeout,
            silent=True,
        )
        if ls_ok and (ls_out or "").strip():
            error_details_parts.append(f"ls -la workspace:\n{(ls_out or '').strip()}")

        # Log tail (may exist even if existence check raced)
        log_ok, log_out, _log_err = ssh_client.exec_command(
            command=f"tail -n 80 {log_file} 2>/dev/null || true",
            background=False,
            timeout=verify_timeout,
            silent=True,
        )
        if log_ok and (log_out or "").strip():
            error_details_parts.append(f"Log content:\n{(log_out or '').strip()[:DEPLOYMENT_LOG_TRUNCATE]}")

        error_details = ("\n   " + "\n   ".join(error_details_parts)) if error_details_parts else ""
        logger.error("❌ Training process did not start!")
        logger.error(f"   No process, marker, or log found within {startup_timeout_seconds}s{error_details}")
        return Err(
            ProviderError(
                message=f"Training failed to start within {startup_timeout_seconds}s{error_details}",
                code="TRAINING_START_TIMEOUT",
                details={"timeout_seconds": startup_timeout_seconds},
            )
        )

    @staticmethod
    def _sanitize_docker_name(name: str) -> str:
        """Sanitize a string to be safe for Docker container names."""
        import re

        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
        # Docker container name max length is 255, but keep it shorter.
        return safe[:DEPLOYMENT_CONTAINER_NAME_MAX_LEN] if len(safe) > DEPLOYMENT_CONTAINER_NAME_MAX_LEN else safe

    def _start_training_docker(
        self, ssh_client: SSHClient, context: dict[str, Any]
    ) -> Result[dict[str, Any], AppError]:
        """
        Start training inside Docker container on single_node host.

        Contract:
        - Host run directory is mounted into container at /workspace
        - Training is executed with workdir=/workspace so imports resolve from mounted code
        - Markers and training.log are written into /workspace (thus into host run dir)
        """
        logger.info("Starting training in Docker container...")

        # Step 1: Create .env file (in docker-mode HELIX_WORKSPACE/PYTHONPATH point to /workspace)
        env_result = self._create_env_file(ssh_client, context)
        if env_result.is_err():
            return Err(env_result.unwrap_err())  # type: ignore[union-attr]  # already ProviderError
        env_file = env_result.unwrap()  # host path: {run_dir}/.env

        # Step 2: Resolve docker settings
        cfg = self._get_single_node_training_cfg()
        image_val = cfg.get("docker_image")
        if not isinstance(image_val, str) or not image_val.strip():
            return Err(
                ProviderError(
                    message="providers.single_node.training.docker_image is required (no default in docker-only mode)",
                    code="DOCKER_IMAGE_NOT_CONFIGURED",
                )
            )
        image = image_val.strip()

        pull_result = self._ensure_docker_image_present(ssh_client, image=image)
        if pull_result.is_failure():
            return Err(pull_result.unwrap_err())  # type: ignore[union-attr]  # already ProviderError

        shm_size = str(cfg.get("docker_shm_size", "16g")).strip() or "16g"
        prefix = str(cfg.get("docker_container_name_prefix", "ryotenkai_training")).strip() or "ryotenkai_training"

        run_obj = context.get(PipelineContextKeys.RUN)
        run_name = getattr(run_obj, "name", "run")
        container_name = self._sanitize_docker_name(f"{prefix}_{run_name}")
        startup_timeout_seconds = DEPLOYMENT_TRAINING_START_TIMEOUT

        # Config is ALWAYS uploaded to config/pipeline_config.yaml by deploy_files()
        remote_config_path = DEPLOYMENT_CONFIG_PATH
        training_cmd = f"python3 -m src.training.run_training --config {remote_config_path}"

        # Create start script (host-side) and run via nohup
        start_script = f"{self._workspace}/start_training.sh"
        log_file_in_container = "/workspace/training.log"

        script_content = f"""#!/bin/bash
set -euo pipefail
cd {self._workspace}

# Ensure env file exists on host (will be visible as /workspace/.env inside container)
test -f {env_file}

# Ensure no leftover container with the same name
docker rm -f {container_name} >/dev/null 2>&1 || true

# Run training container (runtime image contains all deps)
docker run --rm --detach \\
  --name {container_name} \\
  --gpus all \\
  --shm-size {shm_size} \\
  --user "$(id -u):$(id -g)" \\
  -e HOME=/workspace \\
  -v {self._workspace}:/workspace \\
  -w /workspace \\
  {image} \\
  bash -lc ". /workspace/.env && set +e; {training_cmd} >{log_file_in_container} 2>&1; exit_code=$?; set -e; if [ $exit_code -ne 0 ] && [ ! -f /workspace/TRAINING_FAILED ] && [ ! -f /workspace/TRAINING_COMPLETE ]; then echo \"Training failed early. See training.log.\" > /workspace/TRAINING_FAILED || true; fi; exit $exit_code"
"""

        # Step 1: Create start script
        create_script_cmd = f"cat > {start_script} << 'HELIX_SCRIPT_EOF'\n{script_content}HELIX_SCRIPT_EOF"
        success, _, stderr = ssh_client.exec_command(create_script_cmd, timeout=DEPLOYMENT_VERIFY_TIMEOUT)
        if not success:
            return Err(
                ProviderError(message=f"Failed to create start script: {stderr}", code="TRAINING_SCRIPT_CREATE_FAILED")
            )

        # Step 2: Make executable
        # NOTE: `chmod` itself is fast, but the SSH handshake can exceed a 5s budget in real cloud runs.
        success, _, stderr = ssh_client.exec_command(
            f"chmod +x {start_script}", timeout=DEPLOYMENT_SCRIPT_CHMOD_TIMEOUT
        )
        if not success:
            return Err(
                ProviderError(message=f"Failed to chmod start script: {stderr}", code="TRAINING_SCRIPT_CHMOD_FAILED")
            )

        # Step 3: Launch via nohup (SSH returns immediately)
        launch_cmd = f"nohup {start_script} </dev/null >/dev/null 2>&1 & disown"
        success, _, stderr = ssh_client.exec_command(launch_cmd, timeout=DEPLOYMENT_LAUNCH_TIMEOUT)
        if not success:
            return Err(
                ProviderError(message=f"Failed to start training (docker): {stderr}", code="TRAINING_LAUNCH_FAILED")
            )

        logger.info("✅ Docker training command executed")

        poll_interval_seconds = 1
        attempts = max(1, int((startup_timeout_seconds + poll_interval_seconds - 1) / poll_interval_seconds))
        host_log_file = f"{self._workspace}/training.log"

        for _attempt in range(attempts):
            # Markers in host run dir (written from container into /workspace mount)
            success_marker, marker_out, _ = ssh_client.exec_command(
                command=f"test -f {self._workspace}/TRAINING_COMPLETE && echo 'SUCCESS'",
                background=False,
                timeout=DEPLOYMENT_VERIFY_TIMEOUT,
            )
            if success_marker and "SUCCESS" in marker_out:
                logger.info("✅ Training already completed (fast training scenario)")
                return Ok({DEPLOYMENT_MODE_KEY: DEPLOYMENT_DOCKER_VALUE, "container": container_name})

            fail_marker, fail_out, _ = ssh_client.exec_command(
                command=f"test -f {self._workspace}/TRAINING_FAILED && cat {self._workspace}/TRAINING_FAILED",
                background=False,
                timeout=DEPLOYMENT_VERIFY_TIMEOUT,
            )
            if fail_marker and fail_out.strip():
                raw = fail_out.strip()
                msg = raw
                try:
                    payload = json.loads(raw)
                    if isinstance(payload, dict):
                        msg = str(payload.get("error") or payload.get("message") or raw)
                except Exception:
                    pass

                msg = msg.strip()[:DEPLOYMENT_LOG_TRUNCATE]
                logger.error(f"❌ Training failed: {msg}")
                return Err(ProviderError(message=f"Training failed: {msg}", code="TRAINING_FAILED_ON_START"))

            # Check container is running
            if docker_is_container_running(
                ssh_client, name_filter=container_name, timeout_seconds=DEPLOYMENT_VERIFY_TIMEOUT
            ):
                logger.info("✅ Docker container confirmed running")
                return Ok({DEPLOYMENT_MODE_KEY: DEPLOYMENT_DOCKER_VALUE, "container": container_name})

            # Check log file exists on host (container writes into mount)
            success, stdout, _ = ssh_client.exec_command(
                command=f"test -f {host_log_file} && echo '{DEPLOYMENT_MARKER_EXISTS}'",
                background=False,
                timeout=DEPLOYMENT_VERIFY_TIMEOUT,
                silent=True,
            )
            if success and DEPLOYMENT_MARKER_EXISTS in stdout:
                logger.info("✅ training.log exists - assuming training is running (docker)")
                return Ok({DEPLOYMENT_MODE_KEY: DEPLOYMENT_DOCKER_VALUE, "container": container_name})

            time.sleep(poll_interval_seconds)

        # Nothing found within timeout - show log content for diagnostics
        log_success, log_content, _ = ssh_client.exec_command(
            command=f"cat {host_log_file} 2>/dev/null || echo ''",
            background=False,
            timeout=DEPLOYMENT_VERIFY_TIMEOUT,
            silent=True,
        )
        details = (
            f"\n   Log content: {log_content.strip()[:DEPLOYMENT_LOG_TRUNCATE]}"
            if log_success and log_content.strip()
            else ""
        )
        return Err(ProviderError(message=f"Docker training failed to start{details}", code="TRAINING_START_TIMEOUT"))


__all__ = ["TrainingDeploymentManager"]
