"""Sync required source modules to the remote training workspace.

Selective rsync of a fixed allow-list of project subtrees, with a
per-module tar-pipe fallback when ``rsync`` is unavailable on the
remote host. Owned by :class:`CodeSyncer`; composed by
``TrainingDeploymentManager``.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from src.pipeline.stages.managers.deployment.ssh_helpers import build_ssh_opts
from src.pipeline.stages.managers.deployment_constants import (
    DEPLOYMENT_MARKER_EXISTS,
    DEPLOYMENT_RSYNC_TIMEOUT,
    DEPLOYMENT_SSH_CMD_TIMEOUT,
    DEPLOYMENT_TAR_TIMEOUT,
    DEPLOYMENT_VERIFY_TIMEOUT,
)
from src.utils.logger import logger
from src.utils.result import AppError, Err, Ok, ProviderError, Result

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig, Secrets
    from src.utils.ssh_client import SSHClient


DEFAULT_WORKSPACE = "/workspace"


class CodeSyncer:
    """Push the required Python modules to the remote workspace.

    Strategy: one ``rsync`` invocation for all modules with shared
    ``--exclude`` filters. If rsync fails or is missing on the remote,
    fall back to per-module tar-over-ssh pipes. Both paths preserve
    the project's package layout under ``<workspace>/src/...``.
    """

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

    def __init__(self, config: PipelineConfig, secrets: Secrets) -> None:
        # config/secrets are accepted to keep the constructor uniform with
        # the other deployment components, even though CodeSyncer does not
        # currently read them — leaves room for, e.g., per-config exclude
        # overrides later without churning callers.
        self.config = config
        self.secrets = secrets
        self._workspace = DEFAULT_WORKSPACE

    @property
    def workspace(self) -> str:
        return self._workspace

    def set_workspace(self, workspace_path: str) -> None:
        self._workspace = workspace_path

    def sync(self, ssh_client: SSHClient) -> Result[None, AppError]:
        """Sync required source modules to remote in a single rsync call.

        Directories get ``--delete`` semantics (stale files removed);
        single ``.py`` files are included via ``--include``/``--exclude``
        filters. Falls back to per-module tar pipes when rsync is
        unavailable.
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

        ssh_opts = build_ssh_opts(ssh_client)

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
            remote_parent = f"{self._workspace}/{local_path.parent}"
            tar_cmd = (
                f"tar czf - --no-mac-metadata -C {local_path.parent} {local_path.name} 2>/dev/null | "
                f"ssh {ssh_opts} {ssh_client.ssh_target} "
                f"'mkdir -p {remote_parent} && cd {remote_parent} && tar xzf - 2>/dev/null'"
            )
            result = subprocess.run(
                tar_cmd, shell=True, capture_output=True, text=True, timeout=DEPLOYMENT_TAR_TIMEOUT
            )
        else:
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
            verify_cmd = f"test -e {self._workspace}/{module} && echo '{DEPLOYMENT_MARKER_EXISTS}'"
            success, stdout, _ = ssh_client.exec_command(
                command=verify_cmd, background=False, timeout=DEPLOYMENT_VERIFY_TIMEOUT
            )
            if not success or DEPLOYMENT_MARKER_EXISTS not in stdout:
                return Err(ProviderError(message=f"Failed to sync {module}", code="FILE_SYNC_FAILED"))

        return Ok(None)


__all__ = ["DEFAULT_WORKSPACE", "CodeSyncer"]
