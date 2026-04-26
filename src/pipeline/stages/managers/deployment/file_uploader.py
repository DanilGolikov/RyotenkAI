"""Upload config + datasets + scripts to the remote training workspace.

Owns the FILES DEPLOY concern of GPU training deployment: collecting
dataset paths from the strategy chain, batch-uploading via tar-pipe
with per-file SCP fallback, and chaining a follow-up code sync via
the injected :class:`CodeSyncer`.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.datasets.constants import SOURCE_TYPE_LOCAL
from src.pipeline.stages.managers.deployment.ssh_helpers import build_ssh_opts
from src.pipeline.stages.managers.deployment_constants import (
    DEPLOYMENT_CONFIG_PATH,
    DEPLOYMENT_RSYNC_TIMEOUT,
    DEPLOYMENT_VERIFY_TIMEOUT,
)
from src.utils.logger import logger
from src.utils.result import AppError, ConfigError, Err, Failure, Ok, ProviderError, Result

if TYPE_CHECKING:
    from src.pipeline.stages.managers.deployment.code_syncer import CodeSyncer
    from src.utils.config import PipelineConfig, Secrets
    from src.utils.ssh_client import SSHClient


DEFAULT_WORKSPACE = "/workspace"


class FileUploader:
    """Push config, datasets and (via CodeSyncer) source modules to remote.

    Cross-component dependency: :class:`CodeSyncer` is injected so
    :meth:`deploy_files` can chain a source-code sync after the upload
    step. Composition is owned by :class:`TrainingDeploymentManager`,
    which constructs both components and wires them.
    """

    def __init__(
        self,
        config: PipelineConfig,
        secrets: Secrets,
        *,
        code_syncer: CodeSyncer,
    ) -> None:
        self.config = config
        self.secrets = secrets
        self._code_syncer = code_syncer
        self._workspace = DEFAULT_WORKSPACE

    @property
    def workspace(self) -> str:
        return self._workspace

    def set_workspace(self, workspace_path: str) -> None:
        self._workspace = workspace_path

    def _get_training_path(self, local_path: str, strategy_type: str) -> str:
        """Auto-generate ``data/{strategy_type}/{basename}`` for a dataset."""
        basename = Path(local_path).name
        return f"data/{strategy_type}/{basename}"

    def deploy_files(self, ssh_client: SSHClient, context: dict[str, Any]) -> Result[None, AppError]:
        """Upload config + dataset files, then sync source modules.

        Iterates the strategy chain to collect every referenced dataset's
        local paths, maps each to ``data/{strategy_type}/{basename}``,
        attempts a single-shot tar-pipe batch upload, and falls back to
        per-file SCP on batch failure. Source modules are pushed last so
        a config/data failure short-circuits before a (slower) rsync.
        """
        logger.info("📤 Uploading files to pod...")

        try:
            config_path = context.get("config_path", DEPLOYMENT_CONFIG_PATH)
            logger.info(f"📂 Using config: {config_path}")

            files_to_upload: list[tuple[str, str]] = [
                (config_path, DEPLOYMENT_CONFIG_PATH),
            ]

            dataset_files: list[tuple[str, str]] = []
            missing_datasets: list[str] = []
            resolved_datasets_count = 0
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

                if dataset_config.get_source_type() != SOURCE_TYPE_LOCAL:
                    continue

                source_local = dataset_config.source_local
                if source_local is None:
                    missing_datasets.append(f"{dataset_name}: missing source_local")
                    logger.warning(f"⚠️ Dataset [{dataset_name}] missing source_local block")
                    continue

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
                    bound_ds_name: str = dataset_name,
                    bound_strategy_type: str = bound_strategy,
                ) -> None:
                    if not local_ref:
                        return
                    local_abs = self.config.resolve_path(local_ref)

                    remote_rel = self._get_training_path(local_ref, bound_strategy_type)

                    if local_abs and local_abs.exists():
                        dataset_files.append((str(local_abs), remote_rel))
                        files_to_upload.append((str(local_abs), remote_rel))
                        logger.info(f"📂 Dataset [{bound_ds_name}]: {kind} {local_abs} → {remote_rel}")
                        return
                    missing_datasets.append(str(local_ref))
                    logger.warning(f"⚠️ Dataset [{bound_ds_name}] {kind} not found: {local_ref} (resolved: {local_abs})")

                add_dataset_file("train", source_local.local_paths.train)
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

            batch_result = self._upload_files_with_transport(ssh_client, files_to_upload, dataset_files, config_path)
            if batch_result.is_failure():
                return batch_result

            sync_result = self._code_syncer.sync(ssh_client)
            if sync_result.is_failure():
                return sync_result

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

    def _upload_files_batch(
        self, ssh_client: SSHClient, files_to_upload: list[tuple[str, str]]
    ) -> Result[None, AppError]:
        """Upload multiple files in one tar stream for 3-5x speedup."""
        logger.info(f"📦 Batch uploading {len(files_to_upload)} files via tar stream...")

        existing_files = []
        file_mapping = {}

        for local_path, remote_name in files_to_upload:
            if Path(local_path).exists():
                existing_files.append(local_path)
                file_mapping[local_path] = remote_name
                logger.info(f"   📄 {local_path} → {self._workspace}/{remote_name}")
            else:
                logger.warning(f"⚠️ File not found, skipping: {local_path}")

        if not existing_files:
            return Err(ProviderError(message="No files to upload", code="NO_FILES_TO_UPLOAD"))

        with tempfile.TemporaryDirectory() as tmpdir:
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
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_path, dest_path)

            ssh_opts = build_ssh_opts(ssh_client)
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
                if "Cannot change ownership" in result.stderr:
                    logger.warning("⚠️ Ownership warnings (non-critical), files uploaded")
                else:
                    return Err(
                        ProviderError(message=f"Batch upload failed: {result.stderr}", code="BATCH_UPLOAD_FAILED")
                    )

            logger.info(f"✅ Batch upload completed in {elapsed:.1f}s")

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
        """Fallback: upload files individually (slower but more reliable)."""
        logger.info("📦 Uploading files individually (fallback mode)...")

        for local_path, remote_rel_path in dataset_files:
            logger.info(f"📦 Uploading dataset: {remote_rel_path} (local: {local_path})")

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

        config_file = config_path
        if Path(config_file).exists():
            logger.info(f"📦 Uploading configuration: {config_file}")

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


__all__ = ["DEFAULT_WORKSPACE", "FileUploader"]
