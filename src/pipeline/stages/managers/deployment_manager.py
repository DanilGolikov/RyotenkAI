"""
Training Deployment Manager - Manages training deployment on RunPod.

Handles file uploads, dependency installation, and training execution.
Extracted from RunPodDeployer as part of SOLID refactoring (Phase 4/5).
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from src.constants import PROVIDER_SINGLE_NODE
from src.config.datasets.constants import SOURCE_TYPE_LOCAL
from src.infrastructure.mlflow.uri_resolver import resolve_mlflow_uris
from src.pipeline.constants import (
    DEPLOYMENT_CONFIG_PATH,
    DEPLOYMENT_CONTAINER_NAME_MAX_LEN,
    DEPLOYMENT_DOCKER_PULL_TIMEOUT,
    DEPLOYMENT_DOCKER_VALUE,
    DEPLOYMENT_DOCKER_VERIFY_TIMEOUT,
    DEPLOYMENT_ERROR_TRUNCATE,
    DEPLOYMENT_LAUNCH_TIMEOUT,
    DEPLOYMENT_LOG_TRUNCATE,
    DEPLOYMENT_MARKER_EXISTS,
    DEPLOYMENT_MODE_KEY,
    DEPLOYMENT_PYTHON_VERIFY_TIMEOUT,
    DEPLOYMENT_RSYNC_TIMEOUT,
    DEPLOYMENT_SCRIPT_CHMOD_TIMEOUT,
    DEPLOYMENT_SSH_CMD_TIMEOUT,
    DEPLOYMENT_STDERR_TRUNCATE,
    DEPLOYMENT_STDOUT_LINES,
    DEPLOYMENT_TAR_TIMEOUT,
    DEPLOYMENT_TRAINING_START_TIMEOUT,
    DEPLOYMENT_VERIFY_TIMEOUT,
)
from src.pipeline.stages.constants import PipelineContextKeys
from src.providers.training.interfaces import TrainingScriptHooks
from src.utils.docker import docker_is_container_running, ensure_docker_image
from src.utils.logger import logger
from src.utils.result import AppError, ConfigError, Err, Failure, Ok, ProviderError, Result

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

    # =========================================================================
    # SELECTIVE CODE SYNC - Only upload required modules
    # =========================================================================
    # Modules required for training (relative to project root)
    REQUIRED_MODULES: ClassVar[list[str]] = [
        "src/training",  # Training logic (includes src/training/models)
        "src/infrastructure",  # Infrastructure layer (MLflow gateway, etc.)
        "src/utils",  # Utilities
        "src/config",  # Pydantic config schema (used by src/utils/config facade)
        "src/data",  # Data loaders
        "src/constants.py",  # Shared constants (imported by config schemas, validators, etc.)
        "src/__init__.py",  # Package init
    ]

    # Patterns to exclude from sync
    EXCLUDE_PATTERNS: ClassVar[list[str]] = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".pytest_cache",
        "tests",
        "*.md",
    ]

    # Docker images with pre-installed dependencies (skip pip install)
    PREBUILT_IMAGES: ClassVar[list[str]] = [
        "ryotenkai/ryotenkai-training-runtime",
    ]

    @staticmethod
    def _build_ssh_opts(ssh_client: SSHClient) -> str:
        """Build SSH options string reusing the client's ControlMaster socket.

        When ssh_client has ``ssh_base_opts`` (always the case for a real
        ``SSHClient``), we reuse them so that rsync/tar-over-ssh share the
        persistent TCP connection opened by earlier SSH operations.
        """
        base_opts: list[str] | None = getattr(ssh_client, "ssh_base_opts", None)
        if base_opts:
            key_path = getattr(ssh_client, "key_path", None)
            port = getattr(ssh_client, "port", None)
            parts: list[str] = []
            if isinstance(key_path, str) and key_path:
                parts.extend(["-i", key_path])
            if isinstance(port, int) and port:
                parts.extend(["-p", str(port)])
            parts.extend(base_opts)
            return " ".join(parts)

        # Legacy / mock fallback
        alias_mode_attr = getattr(ssh_client, "is_alias_mode", None)
        if not isinstance(alias_mode_attr, bool):
            alias_mode_attr = getattr(ssh_client, "_is_alias_mode", False)
        is_alias_mode = bool(alias_mode_attr) if isinstance(alias_mode_attr, bool) else False

        if is_alias_mode:
            return "-o StrictHostKeyChecking=no"

        opts_parts: list[str] = []
        key_path = getattr(ssh_client, "key_path", None)
        if isinstance(key_path, str) and key_path:
            opts_parts.append(f"-i {key_path}")
        port = getattr(ssh_client, "port", None)
        if isinstance(port, int) and port:
            opts_parts.append(f"-p {port}")
        opts_parts.append("-o StrictHostKeyChecking=no")
        return " ".join(opts_parts)

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
        logger.debug(f"[DEPLOY] Workspace: {self._workspace}")

    # =========================================================================
    # SOURCE CODE SYNC
    # =========================================================================

    def _sync_source_code(self, ssh_client: SSHClient) -> Result[None, AppError]:
        """Sync required source modules to remote in a single rsync call.

        Directories get ``--delete`` semantics (stale files removed);
        single ``.py`` files are included via ``--include``/``--exclude`` filters.
        Falls back to per-module tar pipes when rsync is unavailable.
        """
        logger.info("📦 Syncing source code (selective)...")

        existing_modules: list[str] = []
        for module in self.REQUIRED_MODULES:
            if Path(module).exists():
                existing_modules.append(module)
            else:
                logger.warning(f"⚠️ Module not found: {module}")

        if not existing_modules:
            logger.warning("⚠️ No modules to sync")
            return Ok(None)

        remote_dirs: list[str] = []
        for module in existing_modules:
            if module.endswith(".py"):
                remote_dir = f"{self._workspace}/{Path(module).parent}"
            else:
                remote_dir = f"{self._workspace}/{module}"
            if remote_dir not in remote_dirs:
                remote_dirs.append(remote_dir)

        if remote_dirs:
            mkdir_targets = " ".join(shlex.quote(d) for d in remote_dirs)
            ssh_client.exec_command(
                command=f"mkdir -p {mkdir_targets}",
                background=False,
                timeout=DEPLOYMENT_VERIFY_TIMEOUT,
                silent=True,
            )

        ssh_opts = self._build_ssh_opts(ssh_client)

        rsync_ok = self._sync_all_modules_rsync(ssh_client, existing_modules, ssh_opts)
        if rsync_ok:
            self._clear_pycache(ssh_client)
            logger.info(f"Source code synced ({len(existing_modules)} modules)")
            return Ok(None)

        logger.warning("⚠️ Batch rsync failed, falling back to per-module tar pipes")
        for module in existing_modules:
            tar_result = self._sync_module_tar(ssh_client, module, ssh_opts)
            if tar_result.is_failure():
                logger.error(f"❌ Failed to sync {module}")
                return tar_result
            logger.debug(f"   ✓ {module}")

        self._clear_pycache(ssh_client)
        logger.info(f"Source code synced ({len(existing_modules)} modules, tar fallback)")
        return Ok(None)

    def _sync_all_modules_rsync(
        self,
        ssh_client: SSHClient,
        modules: list[str],
        ssh_opts: str,
    ) -> bool:
        """Single rsync invocation for all modules. Returns True on success."""
        excludes = " ".join(f"--exclude='{p}'" for p in self.EXCLUDE_PATTERNS)

        dirs = [m for m in modules if not m.endswith(".py")]
        files = [m for m in modules if m.endswith(".py")]

        # rsync --include/--filter to send only the dirs and files we need,
        # rooted at project root so relative paths stay intact on the remote.
        # --relative (-R) preserves the directory structure.
        sources = " ".join(shlex.quote(m + "/" if m in dirs else m) for m in modules)
        rsync_cmd = (
            f"rsync -azR --no-owner --no-group --delete {excludes} "
            f"-e 'ssh {ssh_opts}' "
            f"{sources} {ssh_client.ssh_target}:{self._workspace}/"
        )

        try:
            result = subprocess.run(
                rsync_cmd, shell=True, capture_output=True, text=True, timeout=DEPLOYMENT_RSYNC_TIMEOUT
            )
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ Batch rsync timed out")
            return False

        if result.returncode != 0:
            logger.debug(f"Batch rsync failed (rc={result.returncode}): {result.stderr[:200] if result.stderr else ''}")
            return False

        for m in modules:
            logger.debug(f"   ✓ {m}")
        return True

    def _clear_pycache(self, ssh_client: SSHClient) -> None:
        cache_clear_cmd = (
            f"find {self._workspace}/src -type d -name __pycache__ -exec rm -rf {{}} + 2>/dev/null || true"
        )
        ssh_client.exec_command(
            command=cache_clear_cmd, background=False, timeout=DEPLOYMENT_SSH_CMD_TIMEOUT, silent=True
        )

    def _sync_module_tar(self, ssh_client: SSHClient, module: str, ssh_opts: str) -> Result[None, AppError]:
        """Fallback: sync module using tar pipe."""
        local_path = Path(module)

        if local_path.is_file():
            # For single file, use tar+ssh (more reliable than scp in minimal images).
            remote_parent = f"{self._workspace}/{local_path.parent}"
            tar_cmd = (
                f"tar czf - --no-mac-metadata -C {local_path.parent} {local_path.name} 2>/dev/null | "
                f"ssh {ssh_opts} {ssh_client.ssh_target} "
                f"'mkdir -p {remote_parent} && cd {remote_parent} && tar xzf - 2>/dev/null'"
            )
            result = subprocess.run(tar_cmd, shell=True, capture_output=True, text=True, timeout=DEPLOYMENT_TAR_TIMEOUT)
        else:
            # For directory, use tar
            excludes = " ".join(f"--exclude='{p}'" for p in self.EXCLUDE_PATTERNS)
            tar_cmd = (
                f"tar czf - --no-mac-metadata {excludes} -C {local_path.parent} {local_path.name} 2>/dev/null | "
                f"ssh {ssh_opts} {ssh_client.ssh_target} "
                f"'cd {self._workspace}/{local_path.parent} && tar xzf - 2>/dev/null'"
            )
            result = subprocess.run(
                tar_cmd, shell=True, capture_output=True, text=True, timeout=DEPLOYMENT_RSYNC_TIMEOUT
            )

        if result.returncode != 0:
            # Verify it exists anyway
            verify_cmd = f"test -e {self._workspace}/{module} && echo '{DEPLOYMENT_MARKER_EXISTS}'"
            success, stdout, _ = ssh_client.exec_command(
                command=verify_cmd, background=False, timeout=DEPLOYMENT_VERIFY_TIMEOUT
            )
            if not success or DEPLOYMENT_MARKER_EXISTS not in stdout:
                return Err(ProviderError(message=f"Failed to sync {module}", code="FILE_SYNC_FAILED"))

        return Ok(None)

    def _get_training_path(self, local_path: str, strategy_type: str) -> str:
        """
        Auto-generate training path for dataset file.

        Pattern: data/{strategy_type}/{basename(local_path)}

        Args:
            local_path: Local source path (e.g., "data/datasets/train.jsonl")
            strategy_type: Current strategy type (e.g., "sft", "dpo")

        Returns:
            Relative path for remote workspace (e.g., "data/sft/train.jsonl")

        Example:
            >>> _get_training_path("data/datasets/train.jsonl", "sft")
            "data/sft/train.jsonl"

            >>> _get_training_path("/abs/path/my_dataset.jsonl", "dpo")
            "data/dpo/my_dataset.jsonl"
        """
        basename = Path(local_path).name
        return f"data/{strategy_type}/{basename}"

    def deploy_files(self, ssh_client: SSHClient, context: dict[str, Any]) -> Result[None, AppError]:
        """
        Upload dataset and training scripts to pod.

        Args:
            ssh_client: SSH client connected to pod
            context: Pipeline context with dataset_path

        Returns:
            Result with None on success or error message
        """
        logger.info("📤 Uploading files to pod...")

        try:
            # Get config path from context (set by orchestrator)
            config_path = context.get("config_path", DEPLOYMENT_CONFIG_PATH)
            logger.info(f"📂 Using config: {config_path}")

            # ⚡ MULTI-PHASE FIX: Upload ALL datasets from config, not just the first one
            # This is critical for multi-phase training (SFT → COT → DPO)
            files_to_upload: list[tuple[str, str]] = [
                (config_path, DEPLOYMENT_CONFIG_PATH),  # Original config unchanged
            ]

            # Collect dataset files used by training strategies:
            # - local_path: absolute/real local filesystem path
            # - remote_rel_path: path as referenced in config (relative inside remote workspace)
            dataset_files: list[tuple[str, str]] = []
            missing_datasets: list[str] = []
            resolved_datasets_count = 0
            # Use only datasets that are actually referenced by training strategies
            datasets_to_upload: dict[str, Any] = {}
            strategies = self.config.training.get_strategy_chain()
            if strategies:
                for s in strategies:
                    try:
                        ds_cfg = self.config.get_dataset_for_strategy(s)
                    except (AttributeError, KeyError, TypeError, ValueError):
                        continue
                    ds_name = s.dataset or "__primary__"
                    datasets_to_upload[ds_name] = ds_cfg
            else:
                datasets_to_upload["__primary__"] = self.config.get_primary_dataset()

            for dataset_name, dataset_config in datasets_to_upload.items():
                if not dataset_config:
                    continue
                resolved_datasets_count += 1

                # NEW SCHEMA (v6.0):
                # - local source: source_local.local_paths.* (local fs)
                # - training_paths are AUTO-GENERATED: data/{strategy_type}/{basename}
                # - huggingface source: no uploads
                if dataset_config.get_source_type() != SOURCE_TYPE_LOCAL:
                    continue

                source_local = dataset_config.source_local
                if source_local is None:
                    missing_datasets.append(f"{dataset_name}: missing source_local")
                    logger.warning(f"⚠️ Dataset [{dataset_name}] missing source_local block")
                    continue

                # Find strategy_type for this dataset (for auto-generating training_paths)
                # strategies is never empty in valid config (Pydantic rejects empty chain)
                strategy_type: str | None = None
                for s in strategies:
                    if (s.dataset or "__primary__") == dataset_name:
                        strategy_type = s.strategy_type
                        break
                if strategy_type is None:
                    return Err(
                        ConfigError(
                            message="Dataset not referenced by any strategy. Strategy chain cannot be empty (config validation).",
                            code="DATASET_STRATEGY_NOT_FOUND",
                        )
                    )
                bound_strategy: str = strategy_type

                def add_dataset_file(
                    kind: str,
                    local_ref: str | None,
                    bound_ds_name: str = dataset_name,  # Bind loop variable (renamed to avoid shadowing)
                    bound_strategy_type: str = bound_strategy,  # Bind strategy_type for closure
                ) -> None:
                    if not local_ref:
                        return
                    local_abs = self.config.resolve_path(local_ref)

                    # Auto-generate training path: data/{strategy_type}/{basename}
                    remote_rel = self._get_training_path(local_ref, bound_strategy_type)

                    if local_abs and local_abs.exists():
                        dataset_files.append((str(local_abs), remote_rel))
                        files_to_upload.append((str(local_abs), remote_rel))
                        logger.info(f"📂 Dataset [{bound_ds_name}]: {kind} {local_abs} → {remote_rel}")
                        return
                    missing_datasets.append(str(local_ref))
                    logger.warning(f"⚠️ Dataset [{bound_ds_name}] {kind} not found: {local_ref} (resolved: {local_abs})")

                # Upload train file (required)
                add_dataset_file("train", source_local.local_paths.train)

                # Upload eval file (optional)
                add_dataset_file("eval", source_local.local_paths.eval)

            if not dataset_files:
                if missing_datasets:
                    return Err(
                        ConfigError(
                            message="Dataset file not found: "
                            + ", ".join(missing_datasets)
                            + ". Check 'datasets:' in config and ensure each dataset.source_local.local_paths.* exists.",
                            code="DATASET_FILE_NOT_FOUND",
                            details={"missing": missing_datasets},
                        )
                    )

                if resolved_datasets_count == 0:
                    return Err(
                        ConfigError(
                            message="No datasets configured. Add 'datasets:' section to config.",
                            code="NO_DATASETS_CONFIGURED",
                        )
                    )

                logger.info("📦 No local dataset files to upload; using remote-backed datasets from config only")

            logger.info(f"📦 Uploading {len(files_to_upload)} files ({len(dataset_files)} datasets)...")

            batch_result = self._upload_files_with_transport(ssh_client, files_to_upload, dataset_files, config_path, context)
            if batch_result.is_failure():
                return batch_result

            # Upload source code modules (selective sync for smaller transfers)
            sync_result = self._sync_source_code(ssh_client)
            if sync_result.is_failure():
                return sync_result

            # Note: train.py is already included in src/training/train.py
            # No need to upload it separately - it will be run as a module

            logger.info("✅ All files uploaded successfully")
            return Ok(None)

        except (OSError, subprocess.SubprocessError) as e:
            logger.error(f"File upload error: {e}")
            return Err(ProviderError(message=f"Failed to upload files: {e!s}", code="FILE_UPLOAD_FAILED"))

    def _upload_files_with_transport(
        self,
        ssh_client: SSHClient,
        files_to_upload: list[tuple[str, str]],
        dataset_files: list[tuple[str, str]],
        config_path: str,
        context: dict[str, Any],
    ) -> Result[None, AppError]:
        """Upload files via SSH batch (tar pipe), with individual file fallback."""
        logger.debug("📦 Batch uploading files via tar stream...")
        batch_result = self._upload_files_batch(ssh_client, files_to_upload)
        if batch_result.is_failure():
            assert isinstance(batch_result, Failure)
            err_msg = batch_result.unwrap_err()
            logger.warning(f"⚠️ Batch upload failed: {err_msg}, falling back to individual uploads")
            fallback_result = self._upload_files_individual(ssh_client, dataset_files, config_path)
            if fallback_result.is_failure():
                return fallback_result
        return Ok(None)

    # =========================================================================
    # PROVIDER HELPERS
    # =========================================================================

    def _get_active_provider_name(self) -> str:
        """
        Best-effort active provider name.

        Notes:
        - PipelineConfig provides get_active_provider_name().
        - Some unit tests may pass MagicMock-like configs.
        """
        try:
            return self.config.get_active_provider_name()
        except Exception:
            training = getattr(self.config, "training", None)
            name = getattr(training, "provider", None) if training else None
            if isinstance(name, str) and name:
                return name
            return PROVIDER_SINGLE_NODE

    def _is_single_node_provider(self) -> bool:
        return self._get_active_provider_name() == PROVIDER_SINGLE_NODE

    # =========================================================================
    # DEPENDENCY INSTALLATION
    # =========================================================================

    def install_dependencies(self, ssh_client: SSHClient) -> Result[None, AppError]:
        """
        Docker-only dependency handling.

        - single_node: verify the configured runtime image contains required packages
          (runs `docker run --rm ... python -c "import ..."`) on the host.
        - cloud providers (RunPod): SSH is inside the pod image, so we verify packages in-place.
          If verification fails, we FAIL (image is the single source of truth).
        """
        if self._is_single_node_provider():
            logger.info("🐳 single_node: docker-only deps (no host installs) — verifying runtime image...")
            return self._verify_single_node_docker_runtime(ssh_client)

        # Cloud providers: verify inside the current environment (pod container)
        logger.info("☁️ cloud: docker-only deps — verifying runtime contract inside the current container...")
        verify_result = self._verify_prebuilt_dependencies(ssh_client)
        if verify_result.is_failure():
            orig_err = verify_result.unwrap_err()
            return Err(
                ProviderError(
                    message="Training runtime image missing required packages. Dependencies are docker-only (no fallback install).",
                    code="RUNTIME_DEPS_MISSING",
                    details=orig_err.to_log_dict(),
                )
            )

        return Ok(None)

    @staticmethod
    def _verify_prebuilt_dependencies(ssh_client: SSHClient) -> Result[None, ProviderError]:
        """
        Verify dependencies in prebuilt Docker image (RunPod mode).

        This is for cloud providers (RunPod) where SSH connects INSIDE the pod's
        Docker container, so we can directly check Python packages.

        For single_node Docker mode, this method is NOT called - we skip
        host checks entirely since deps are in the container.

        Only checks that key packages are available, does NOT install anything.
        """
        logger.info("📦 Verifying prebuilt image dependencies (cloud mode)...")

        # Single source of truth: runtime image must contain the contract checker.
        # Python binary may be either `python3` or `python` depending on the image.
        verify_cmd = (
            "if command -v python3 >/dev/null 2>&1; then python3 /opt/helix/runtime_check.py; "
            "elif command -v python >/dev/null 2>&1; then python /opt/helix/runtime_check.py; "
            "elif [ -x /opt/conda/bin/python3 ]; then /opt/conda/bin/python3 /opt/helix/runtime_check.py; "
            "elif [ -x /opt/conda/bin/python ]; then /opt/conda/bin/python /opt/helix/runtime_check.py; "
            'else echo "PYTHON_NOT_FOUND"; exit 127; fi'
        )
        success, stdout, stderr = ssh_client.exec_command(
            command=verify_cmd, background=False, timeout=DEPLOYMENT_PYTHON_VERIFY_TIMEOUT
        )

        if not success or "OK" not in (stdout or ""):
            details = (stderr or stdout or "").strip()[:DEPLOYMENT_STDERR_TRUNCATE]
            logger.error(f"❌ Runtime contract check failed: {details if details else 'unknown'}")
            return Err(
                ProviderError(
                    message="Runtime contract check failed (missing deps or wrong image).",
                    code="RUNTIME_CONTRACT_CHECK_FAILED",
                    details={"output": details},
                )
            )

        # Log versions for debugging (printed by the checker)
        logger.info("✅ Runtime dependencies verified:")
        for line in (stdout or "").strip().split("\n")[:DEPLOYMENT_STDOUT_LINES]:
            logger.info(f"   {line}")

        return Ok(None)

    def _get_single_node_training_cfg(self) -> dict[str, Any]:
        """Best-effort get providers.single_node.training config as raw dict."""
        return self._get_provider_training_cfg("single_node")

    def _get_cloud_training_cfg(self) -> dict[str, Any]:
        """Best-effort get providers.<active_cloud_provider>.training config as raw dict."""
        return self._get_provider_training_cfg(self._get_active_provider_name())

    def _get_provider_training_cfg(self, provider_name: str) -> dict[str, Any]:
        """
        Best-effort get providers.<provider_name>.training config as raw dict.

        Defensive: unit tests may inject MagicMock configs; provider config may be absent.
        Falls back to the default provider config when the named provider is not found.
        """
        provider_cfg_obj: Any
        try:
            provider_cfg_obj = self.config.get_provider_config(provider_name)
        except (AttributeError, KeyError, ValueError, TypeError):
            try:
                provider_cfg_obj = self.config.get_provider_config()
            except (AttributeError, KeyError, ValueError, TypeError):
                provider_cfg_obj = {}

        provider_cfg = provider_cfg_obj if isinstance(provider_cfg_obj, dict) else {}
        training_cfg = provider_cfg.get("training")
        return training_cfg if isinstance(training_cfg, dict) else {}

    def _ensure_docker_image_present(self, ssh_client: SSHClient, *, image: str) -> Result[None, ProviderError]:
        """Ensure Docker image is available on the remote host."""
        return ensure_docker_image(ssh=ssh_client, image=image, pull_timeout_seconds=DEPLOYMENT_DOCKER_PULL_TIMEOUT)

    def _verify_single_node_docker_runtime(self, ssh_client: SSHClient) -> Result[None, AppError]:
        """
        Verify dependencies inside the single_node training Docker runtime image.

        This must run on the host (via SSH) because in single_node docker-mode
        SSH is connected to the host, not inside a container.
        """
        cfg = self._get_single_node_training_cfg()
        image_val = cfg.get("docker_image")
        if not isinstance(image_val, str) or not image_val.strip():
            return Err(
                ConfigError(
                    message="providers.single_node.training.docker_image is required (no default in docker-only mode)",
                    code="DOCKER_IMAGE_NOT_CONFIGURED",
                )
            )
        image = image_val.strip()

        logger.info(f"🐳 Training runtime image: {image}")
        pull_result = self._ensure_docker_image_present(ssh_client, image=image)
        if pull_result.is_failure():
            return Err(pull_result.unwrap_err())  # type: ignore[union-attr]  # already ProviderError

        logger.info("📦 Verifying Docker runtime image dependencies (single_node)...")
        verify_cmd = f"docker run --rm --gpus all {image} python3 /opt/helix/runtime_check.py"
        logger.info(f"🔎 Runtime contract check: {verify_cmd}")
        success, stdout, stderr = ssh_client.exec_command(
            command=verify_cmd, background=False, timeout=DEPLOYMENT_DOCKER_VERIFY_TIMEOUT
        )
        if not success or "OK" not in (stdout or ""):
            return Err(
                ProviderError(
                    message=f"Training runtime image '{image}' missing required packages or failed to start.",
                    code="DOCKER_RUNTIME_CHECK_FAILED",
                    details={"image": image, "stderr": stderr[:DEPLOYMENT_ERROR_TRUNCATE] if stderr else "empty"},
                )
            )

        logger.info("✅ Docker runtime image dependencies verified")
        return Ok(None)

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

    # Private helper methods

    def _upload_files_batch(
        self, ssh_client: SSHClient, files_to_upload: list[tuple[str, str]]
    ) -> Result[None, AppError]:
        """
        Upload multiple files in one tar stream for 3-5x speedup.

        Args:
            ssh_client: SSH client instance
            files_to_upload: List of (local_path, remote_name) tuples

        Returns:
            Result indicating success or failure
        """
        logger.info(f"📦 Batch uploading {len(files_to_upload)} files via tar stream...")

        # Filter existing files and build file mapping
        existing_files = []
        file_mapping = {}  # local_path -> remote_name

        for local_path, remote_name in files_to_upload:
            if Path(local_path).exists():
                existing_files.append(local_path)
                file_mapping[local_path] = remote_name
                logger.info(f"   📄 {local_path} → {self._workspace}/{remote_name}")
            else:
                logger.warning(f"⚠️ File not found, skipping: {local_path}")

        if not existing_files:
            return Err(ProviderError(message="No files to upload", code="NO_FILES_TO_UPLOAD"))

        # Create temporary directory structure that matches remote layout
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy files to temp dir with remote names
            for local_path in existing_files:
                remote_name = file_mapping[local_path]
                # Defense-in-depth: remote_name must be relative, otherwise Path(tmpdir)/remote_name
                # will ignore tmpdir and can cause SameFileError or write outside staging dir.
                if Path(remote_name).is_absolute():
                    return Err(
                        ConfigError(
                            message=f"Invalid remote path (must be relative): {remote_name}. "
                            "Use source_local.training_paths.* for remote-relative paths.",
                            code="INVALID_REMOTE_PATH",
                        )
                    )
                dest_path = Path(tmpdir) / remote_name

                # Create parent directory if needed
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_path, dest_path)

            # Build tar | ssh | tar command
            ssh_opts = self._build_ssh_opts(ssh_client)
            tar_cmd = (
                f"cd {tmpdir} && "
                f"tar czf - . | "
                f"ssh {ssh_opts} {ssh_client.ssh_target} "
                f"'cd {self._workspace} && tar xzf -'"
            )

            logger.info("📤 Streaming files via tar pipe...")
            start_time = time.time()

            result = subprocess.run(
                tar_cmd, shell=True, capture_output=True, text=True, timeout=DEPLOYMENT_RSYNC_TIMEOUT
            )

            elapsed = time.time() - start_time

            if result.returncode != 0:
                # Check for ownership warnings (non-critical)
                if "Cannot change ownership" in result.stderr:
                    logger.warning("⚠️ Ownership warnings (non-critical), files uploaded")
                else:
                    return Err(
                        ProviderError(message=f"Batch upload failed: {result.stderr}", code="BATCH_UPLOAD_FAILED")
                    )

            logger.info(f"✅ Batch upload completed in {elapsed:.1f}s")

            # Verify files were uploaded (check both root and nested files)
            logger.info("✅ Verified files on remote:")
            remote_names = [file_mapping[local_path] for local_path in existing_files]
            verify_results = dict.fromkeys(remote_names, False)
            verify_cmd_parts = []

            for remote_name in remote_names:
                remote_path = f"{self._workspace}/{remote_name}"
                verify_cmd_parts.append(
                    f"if test -f {shlex.quote(remote_path)}; then "
                    f"printf 'OK::%s\\n' {shlex.quote(remote_name)}; "
                    f"else printf 'MISS::%s\\n' {shlex.quote(remote_name)}; fi"
                )

            verify_cmd = "; ".join(verify_cmd_parts)
            success, stdout, _ = ssh_client.exec_command(
                command=verify_cmd, background=False, timeout=DEPLOYMENT_VERIFY_TIMEOUT
            )

            if success:
                for line in (stdout or "").splitlines():
                    if line.startswith("OK::"):
                        remote_name = line.removeprefix("OK::")
                        if remote_name in verify_results:
                            verify_results[remote_name] = True
            else:
                logger.warning("⚠️ Batch verification command failed; proceeding with per-file status as not verified.")

            for remote_name in remote_names:
                if verify_results.get(remote_name, False):
                    logger.info(f"   ✓ {remote_name}")
                else:
                    logger.warning(f"   ✗ {remote_name} NOT FOUND!")

            return Ok(None)

    def _upload_files_individual(  # noqa: WPS231
        self,
        ssh_client: SSHClient,
        dataset_files: list[tuple[str, str]],
        config_path: str = "config/pipeline_config.yaml",
    ) -> Result[None, AppError]:
        """
        Fallback: upload files individually (slower but more reliable).

        Args:
            ssh_client: SSH client instance
            dataset_files: List of (local_path, remote_rel_path) for dataset files
            config_path: Path to config file

        Returns:
            Result indicating success or failure
        """
        logger.info("📦 Uploading files individually (fallback mode)...")

        # Upload all datasets
        for local_path, remote_rel_path in dataset_files:
            logger.info(f"📦 Uploading dataset: {remote_rel_path} (local: {local_path})")

            # Create parent directories on remote
            remote_path = f"{self._workspace}/{remote_rel_path}"
            remote_dir = str(Path(remote_path).parent)
            success, _stdout, stderr = ssh_client.exec_command(
                command=f"mkdir -p {remote_dir}",
                background=False,
                timeout=DEPLOYMENT_VERIFY_TIMEOUT,
            )
            if not success:
                logger.warning(f"⚠️ Failed to create dir {remote_dir}: {stderr}")

            success, error_msg = ssh_client.upload_file(
                local_path=local_path,
                remote_path=remote_path,
                verify=True,
            )

            if not success:
                return Err(
                    ProviderError(
                        message=f"Failed to upload dataset {remote_rel_path}: {error_msg}",
                        code="DATASET_UPLOAD_FAILED",
                        details={"remote_path": remote_rel_path},
                    )
                )

            logger.info(f"✅ Dataset uploaded: {remote_rel_path}")

        # Upload config file (use actual config from parameter)
        config_file = config_path
        if Path(config_file).exists():
            logger.info(f"📦 Uploading configuration: {config_file}")

            # Create config directory first
            success, _stdout, stderr = ssh_client.exec_command(
                command=f"mkdir -p {self._workspace}/config",
                background=False,
                timeout=DEPLOYMENT_VERIFY_TIMEOUT,
            )

            if not success:
                logger.warning(f"⚠️ Failed to create config directory: {stderr}")

            success, error_msg = ssh_client.upload_file(
                local_path=config_file, remote_path=f"{self._workspace}/config/pipeline_config.yaml", verify=True
            )

            if not success:
                logger.warning(f"⚠️ Config upload failed: {error_msg}")
            else:
                logger.info("✅ Configuration uploaded successfully")
        else:
            logger.warning(f"⚠️ Config file not found: {config_file}")

        return Ok(None)


__all__ = ["TrainingDeploymentManager"]
