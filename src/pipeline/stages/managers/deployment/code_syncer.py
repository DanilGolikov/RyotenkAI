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
    DEPLOYMENT_PYTHON_VERIFY_TIMEOUT,
    DEPLOYMENT_RSYNC_TIMEOUT,
    DEPLOYMENT_SSH_CMD_TIMEOUT,
    DEPLOYMENT_STDERR_TRUNCATE,
    DEPLOYMENT_STDOUT_LINES,
    DEPLOYMENT_TAR_TIMEOUT,
    DEPLOYMENT_VERIFY_TIMEOUT,
)
from src.utils.logger import logger
from src.utils.result import AppError, Err, Ok, ProviderError, Result

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig, Secrets
    from src.utils.ssh_client import SSHClient


DEFAULT_WORKSPACE = "/workspace"

# Path to the runtime contract checker baked into the training image.
# Same path used by :class:`DependencyInstaller` for pip-package mode;
# CodeSyncer invokes the same script with ``--check-source`` to verify
# the synced ``src.*`` modules are importable on the pod (PR-A gate).
RUNTIME_CHECK_SCRIPT = "/opt/helix/runtime_check.py"

# rc=2 from runtime_check.py --check-source means "synced source broken"
# (one or more required src.* modules failed to import). Distinct from
# rc=1 which means "image broken / pip packages missing". rc=127 is the
# shell's "command not found" — emitted when /opt/helix/runtime_check.py
# is missing entirely (image too old to support the gate).
_IMPORT_GATE_RC_FAILED = 2
_IMPORT_GATE_RC_MISSING = 127


class CodeSyncer:
    """Push the required Python modules to the remote workspace.

    Strategy: one ``rsync`` invocation for all modules with shared
    ``--exclude`` filters. If rsync fails or is missing on the remote,
    fall back to per-module tar-over-ssh pipes. Both paths preserve
    the project's package layout under ``<workspace>/src/...``.
    """

    # Code shipping policy: ship the entire ``src/`` tree to the pod,
    # filtered by ``EXCLUDE_PATTERNS`` (tests, caches, docs).
    #
    # Why "ship everything" rather than a hand-curated list of submodules:
    #
    #   * The selective whitelist drifts. Phase-1 had 12 explicit entries
    #     (``src/training``, ``src/providers``, ``src/runner``, …) but
    #     missed transitive imports — e.g. ``src/providers/runpod/training/
    #     provider.py`` imports ``src.pipeline``, which was Mac-only and
    #     not whitelisted, causing a ``ModuleNotFoundError`` at trainer
    #     spawn (run_20260502_113553_r8rul, the 16th of a 16-crash chain).
    #     Every new transitive import would re-trigger the same drift
    #     until someone updates the list.
    #
    #   * Cost is negligible. Full ``src/`` is ~3.4 MB after exclusions
    #     (vs ~5.2 MB previously — selective was actually larger because
    #     it shipped full subdirs). rsync ships only changed bytes after
    #     the first run; first-run cost is ~150 ms on a 25 MB/s link.
    #
    #   * Architectural enforcement belongs in static analysis, not in
    #     the shipping list. Phase 3 introduces an importlinter rule that
    #     forbids ``src.providers.* → src.pipeline.*`` (and similar
    #     pod→Mac-only directions) at CI time — which is the right place
    #     to catch boundary violations, not in the deploy step.
    #
    #   * Pod sees Mac-only code (``src/api``, ``src/cli``, ``src/pipeline``,
    #     ``src/reports``) physically on disk but never imports it at
    #     runtime: trainer imports start from ``src.training.run_training``
    #     and pull only what they need. No RAM cost, no security cost
    #     (no secrets in source), no startup cost.
    #
    # NOTE on ``src/community`` vs ``community/``:
    #   * ``src/community`` is the plugin FRAMEWORK (catalog, registry,
    #     manifest, etc.) — shipped with every run as part of the ``src/``
    #     tree. Trainer imports it at module-load time
    #     (``src/training/reward_plugins/factory.py`` etc).
    #   * ``community/`` (the SIBLING dir at repo root) holds the actual
    #     plugin CONTENT (reward / evaluation / validation plugin
    #     packages). Delivered separately through ``PluginPacker`` so we
    #     ship only the plugins a given run declares, not the whole
    #     catalog.
    REQUIRED_MODULES: ClassVar[list[str]] = ["src"]

    # Patterns to exclude from sync — keep the pod-side tree lean.
    #   * tests / *.pyc / __pycache__ / .pytest_cache: dev artefacts
    #   * *.md: docs are git-only, no value on the pod
    #   * src/tests: project tests live under ``src/tests`` not top-level
    #     ``tests/``; the ``tests`` glob below already covers both paths
    #     because rsync ``--exclude`` matches by basename in any depth.
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
            gate_result = self._verify_importability(ssh_client)
            if gate_result.is_failure():
                return gate_result
            logger.info(f"Source code synced ({len(existing_modules)} modules) + importable")
            return Ok(None)

        logger.warning("⚠️ Batch rsync failed, falling back to per-module tar pipes")
        for module in existing_modules:
            tar_result = self._sync_module_tar(ssh_client, module, ssh_opts)
            if tar_result.is_failure():
                logger.error(f"❌ Failed to sync {module}")
                return tar_result
            logger.debug(f"   ✓ {module}")

        self._clear_pycache(ssh_client)
        gate_result = self._verify_importability(ssh_client)
        if gate_result.is_failure():
            return gate_result
        logger.info(f"Source code synced ({len(existing_modules)} modules, tar fallback) + importable")
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

    def _verify_importability(self, ssh_client: SSHClient) -> Result[None, AppError]:
        """Post-sync importability gate (PR-A).

        Runs ``/opt/helix/runtime_check.py --check-source`` on the pod with
        the freshly synced ``src.*`` tree on PYTHONPATH. ``rsync rc=0`` does
        not guarantee that all required modules ended up importable —
        directories can be missing locally (post-rebase, post-stash) or
        deleted on the pod by a previous ``--delete`` pass that did not
        list the same module set. This gate validates the **same import
        path that the trainer will exercise** before we spawn it.

        Failure modes:

        * rc=2 + ``<mod>=NOT_IMPORTABLE`` lines → named modules failed to
          import. Pipeline halts at Stage 1 with the offending list.
        * rc=127 → ``/opt/helix/runtime_check.py`` does not exist on the
          image. Surfaced as an actionable "rebuild image" error.
        * SSH timeout / non-zero unrelated rc → returned as a generic
          gate failure with the raw output for forensics.
        """
        cmd = (
            f"cd {shlex.quote(self._workspace)} && "
            f"PYTHONPATH={shlex.quote(self._workspace)} "
            f"python3 {RUNTIME_CHECK_SCRIPT} --check-source 2>&1 || echo \"GATE_RC=$?\""
        )
        success, stdout, stderr = ssh_client.exec_command(
            command=cmd,
            background=False,
            timeout=DEPLOYMENT_PYTHON_VERIFY_TIMEOUT,
            silent=True,
        )

        output = (stdout or "").strip()
        gate_rc = self._extract_gate_rc(output)

        if success and output.startswith("OK") and gate_rc is None:
            logger.info("✅ Post-sync import check passed:")
            for line in output.split("\n")[:DEPLOYMENT_STDOUT_LINES]:
                logger.info(f"   {line}")
            return Ok(None)

        if gate_rc == _IMPORT_GATE_RC_MISSING:
            error_msg = (
                f"Post-sync import check unavailable: {RUNTIME_CHECK_SCRIPT} "
                "missing on pod. The training image is too old to support the "
                "import gate — rebuild the runtime image (docker/training/) so "
                "it ships the updated runtime_check.py."
            )
            logger.error(f"❌ {error_msg}")
            return Err(
                AppError(
                    message=error_msg,
                    code="IMPORT_GATE_UNAVAILABLE",
                    details={"output": output[:DEPLOYMENT_STDERR_TRUNCATE]},
                )
            )

        if gate_rc == _IMPORT_GATE_RC_FAILED:
            failed_modules = [
                line.split("=", 1)[0]
                for line in output.split("\n")
                if "=NOT_IMPORTABLE" in line
            ]
            error_msg = (
                f"Post-sync import check failed. Modules not importable on "
                f"pod: {failed_modules or '<unknown>'}. Action: ensure these "
                f"modules exist in your local checkout before deploy. Full "
                f"pod output:\n{output[:DEPLOYMENT_STDERR_TRUNCATE]}"
            )
            logger.error(f"❌ {error_msg}")
            return Err(
                AppError(
                    message=error_msg,
                    code="IMPORT_GATE_FAILED",
                    details={"failed_modules": failed_modules, "output": output[:DEPLOYMENT_STDERR_TRUNCATE]},
                )
            )

        # SSH-level failure, timeout, or unexpected rc — surface raw output.
        details = (stderr or output or "<no output>").strip()[:DEPLOYMENT_STDERR_TRUNCATE]
        error_msg = (
            f"Post-sync import check returned an unexpected result "
            f"(success={success}, gate_rc={gate_rc}). Raw: {details}"
        )
        logger.error(f"❌ {error_msg}")
        return Err(
            AppError(
                message=error_msg,
                code="IMPORT_GATE_ERROR",
                details={"output": details, "gate_rc": gate_rc},
            )
        )

    @staticmethod
    def _extract_gate_rc(output: str) -> int | None:
        """Parse ``GATE_RC=<n>`` trailer emitted by the SSH wrapper.

        The wrapper appends ``GATE_RC=$?`` only when ``runtime_check.py``
        exits non-zero (the ``|| echo ...`` short-circuit). Absent trailer
        means the script exited 0 — no failure to report.
        """
        for line in reversed(output.splitlines()):
            stripped = line.strip()
            if stripped.startswith("GATE_RC="):
                try:
                    return int(stripped.split("=", 1)[1])
                except ValueError:
                    return None
        return None

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
