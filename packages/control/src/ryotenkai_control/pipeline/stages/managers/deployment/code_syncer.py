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

from ryotenkai_control.pipeline.stages.managers.deployment.ssh_helpers import build_ssh_opts
from ryotenkai_control.pipeline.stages.managers.deployment_constants import (
    DEPLOYMENT_MARKER_EXISTS,
    DEPLOYMENT_RSYNC_TIMEOUT,
    DEPLOYMENT_SSH_CMD_TIMEOUT,
    DEPLOYMENT_TAR_TIMEOUT,
    DEPLOYMENT_VERIFY_TIMEOUT,
)
from ryotenkai_shared.utils.logger import logger
from ryotenkai_shared.utils.result import AppError, Err, Ok, ProviderError, Result

if TYPE_CHECKING:
    from ryotenkai_shared.config import PipelineConfig, Secrets
    from ryotenkai_shared.utils.ssh_client import SSHClient


DEFAULT_WORKSPACE = "/workspace"

# Phase 3 PR-3.3 (transport-unification-v2): the legacy on-pod
# import gate (``runtime_check.py --check-source`` invoked via SSH)
# is gone. The same modules are now checked over HTTP via
# :meth:`JobClient.check_imports` after ``/healthz`` is green —
# see :data:`REQUIRED_SRC_MODULES` below for the canonical list
# the launcher sends.
REQUIRED_SRC_MODULES: list[str] = [
    "ryotenkai_shared.config",
    "ryotenkai_shared.config.loader",
    "ryotenkai_community.catalog",
    "ryotenkai_providers",
    "ryotenkai_pod.trainer.run_training",
    "ryotenkai_pod.runner.main",
]


class CodeSyncer:
    """Push the four pod-relevant ``ryotenkai_*`` packages to the
    remote workspace.

    Strategy: one ``rsync`` invocation per ``(local, dest)`` pair. If
    rsync fails or is missing on the remote, fall back to tar-over-ssh
    pipes. Both paths drop the local ``packages/<pkg>/src/`` prefix and
    place each package as ``<workspace>/<name>/`` so
    ``PYTHONPATH=<workspace>`` resolves ``ryotenkai_pod.runner.main``,
    ``ryotenkai_shared.config``, etc. directly.
    """

    # Code shipping policy (post Phase B uv-workspace packagization,
    # 2026-05-03): ship the import roots of the four pod-relevant
    # packages so ``PYTHONPATH=<workspace>`` makes them top-level
    # importable. Each pair is ``(local_source_dir, remote_dest_name)``;
    # the rsync wrapper uses an explicit source/dest spelling so we
    # don't reproduce the ``packages/<pkg>/src/`` prefix on the pod.
    #
    # Why each entry is here:
    #   * ``ryotenkai_shared`` — leaf utilities (config / Secrets /
    #     pipeline_context / observability / lifecycle Protocol).
    #     Imported by every other pod-side package.
    #   * ``ryotenkai_community`` — plugin loader / catalog / manifest.
    #     Trainer instantiates reward + dataset-validation plugins
    #     through it.
    #   * ``ryotenkai_providers`` — RunPod / single-node lifecycle
    #     clients resolved at runtime via importlib by
    #     ``ryotenkai_pod.runner.runtime.provider_registry``.
    #   * ``ryotenkai_pod`` — the runner's FastAPI app + the trainer
    #     subprocess code.
    #
    # ``ryotenkai_control`` is intentionally absent — it's Mac-only and
    # the importlinter contract forbids the pod from importing it.
    PROVIDED_PACKAGES: ClassVar[list[tuple[str, str]]] = [
        ("packages/shared/src/ryotenkai_shared", "ryotenkai_shared"),
        ("packages/community/src/ryotenkai_community", "ryotenkai_community"),
        ("packages/providers/src/ryotenkai_providers", "ryotenkai_providers"),
        ("packages/pod/src/ryotenkai_pod", "ryotenkai_pod"),
    ]

    # Patterns to exclude from sync — keep the pod-side tree lean.
    #   * tests / *.pyc / __pycache__ / .pytest_cache: dev artefacts
    #   * *.md: docs are git-only, no value on the pod
    EXCLUDE_PATTERNS: ClassVar[list[str]] = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".pytest_cache",
        "tests",
        "*.md",
    ]

    @staticmethod
    def _repo_root() -> Path:
        """Walk up from this file to the workspace root.

        Identified by an ancestor that has both ``pyproject.toml`` and a
        ``packages/`` directory — the uv workspace marker. Mirrors the
        same shape :func:`ryotenkai_shared.config.secrets.loader.load_secrets`
        uses for repo-root discovery; pinned ``parents[N]`` arithmetic was
        the regression class that broke ``COMMUNITY_ROOT`` and
        ``secrets.env`` lookup post Phase B.
        """
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "pyproject.toml").is_file() and (parent / "packages").is_dir():
                return parent
        return Path.cwd()

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
        """Sync each provided package to ``<workspace>/<dest_name>/``.

        After Phase B the local source layout is
        ``packages/<pkg>/src/<import_name>/``; the pod expects each
        ``<import_name>`` directly under PYTHONPATH=workspace, so the
        sync flattens the ``packages/<pkg>/src/`` prefix per pair. Each
        pair is rsync'd independently (no ``-R``) so the dest can be
        renamed cleanly. Falls back to tar-over-ssh per-pair when rsync
        is unavailable on the remote.
        """
        logger.info("📦 Syncing source code (selective)...")

        # ``PROVIDED_PACKAGES`` paths are relative to the repo root, but
        # the pipeline worker subprocess runs with ``cwd=packages/control/src/``.
        # Resolve each pair against the workspace root (walk up from this
        # file looking for the ``packages/`` + ``pyproject.toml`` marker)
        # so the rsync source spelling works regardless of CWD.
        repo_root = self._repo_root()
        existing_pairs: list[tuple[str, str]] = []
        for local, dest in self.PROVIDED_PACKAGES:
            local_abs = (repo_root / local) if not Path(local).is_absolute() else Path(local)
            if local_abs.is_dir():
                existing_pairs.append((str(local_abs), dest))
            else:
                logger.warning(f"⚠️ Package source not found: {local_abs}")

        if not existing_pairs:
            logger.warning("⚠️ No packages to sync")
            return Ok(None)

        remote_dirs = [f"{self._workspace}/{dest}" for _, dest in existing_pairs]
        mkdir_targets = " ".join(shlex.quote(d) for d in remote_dirs)
        ssh_client.exec_command(
            command=f"mkdir -p {mkdir_targets}",
            background=False,
            timeout=DEPLOYMENT_VERIFY_TIMEOUT,
            silent=True,
        )

        ssh_opts = build_ssh_opts(ssh_client)

        rsync_ok = self._sync_all_modules_rsync(ssh_client, existing_pairs, ssh_opts)
        if rsync_ok:
            self._clear_pycache(ssh_client)
            # Phase 3 PR-3.3 (transport-unification-v2): post-sync
            # importability gate moved out of CodeSyncer. The runner
            # isn't running yet, so the SSH ``runtime_check.py``
            # invocation is gone. ``TrainingLauncher`` now calls
            # ``JobClient.check_imports`` AFTER ``/healthz`` returns
            # green, which validates the SAME synced source AND the
            # runner image — single check covers both.
            logger.info(f"Source code synced ({len(existing_pairs)} packages, rsync)")
            return Ok(None)

        logger.warning("⚠️ Batch rsync failed, falling back to per-package tar pipes")
        for local, dest in existing_pairs:
            tar_result = self._sync_module_tar(ssh_client, local, dest, ssh_opts)
            if tar_result.is_failure():
                logger.error(f"❌ Failed to sync {local}")
                return tar_result
            logger.debug(f"   ✓ {local} → {dest}")

        self._clear_pycache(ssh_client)
        logger.info(f"Source code synced ({len(existing_pairs)} packages, tar fallback)")
        return Ok(None)

    def _sync_all_modules_rsync(
        self,
        ssh_client: SSHClient,
        pairs: list[tuple[str, str]],
        ssh_opts: str,
    ) -> bool:
        """One rsync invocation per (source, dest) pair.

        Each pair is its own rsync call because the pod-side dest name
        (``ryotenkai_shared``, etc.) differs from the local source path
        (``packages/shared/src/ryotenkai_shared``), and ``rsync -R`` would
        replicate the local prefix on the remote. Returns True iff every
        pair rsync'd cleanly.
        """
        excludes = " ".join(f"--exclude='{p}'" for p in self.EXCLUDE_PATTERNS)

        for local, dest in pairs:
            # Trailing slash on source means "copy contents of dir into
            # dest" rather than "copy dir-as-named into dest" — without
            # it we'd get ``<workspace>/ryotenkai_pod/ryotenkai_pod/``.
            source = shlex.quote(local.rstrip("/") + "/")
            target = shlex.quote(f"{self._workspace}/{dest}/")
            rsync_cmd = (
                f"rsync -az --no-owner --no-group --delete {excludes} "
                f"-e 'ssh {ssh_opts}' "
                f"{source} {ssh_client.ssh_target}:{target}"
            )
            try:
                result = subprocess.run(
                    rsync_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=DEPLOYMENT_RSYNC_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️ Rsync timed out for {local}")
                return False
            if result.returncode != 0:
                stderr_snip = (result.stderr or "")[:200]
                logger.debug(
                    f"Rsync failed for {local} (rc={result.returncode}): {stderr_snip}"
                )
                return False
            logger.debug(f"   ✓ {local} → {dest}")
        return True

    def _clear_pycache(self, ssh_client: SSHClient) -> None:
        # Walk every dest dir we just synced; one ``find`` invocation per
        # dest keeps the command short enough for SSH's argv limit even
        # on a heavily-populated workspace.
        targets = " ".join(
            shlex.quote(f"{self._workspace}/{dest}") for _, dest in self.PROVIDED_PACKAGES
        )
        cache_clear_cmd = (
            f"for d in {targets}; do "
            "  find \"$d\" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true; "
            "done"
        )
        ssh_client.exec_command(
            command=cache_clear_cmd, background=False, timeout=DEPLOYMENT_SSH_CMD_TIMEOUT, silent=True
        )

    # Phase 3 PR-3.3 (transport-unification-v2): the legacy
    # ``_verify_importability`` SSH gate is gone. The same modules
    # are now checked via ``JobClient.check_imports`` after
    # ``/healthz`` returns green — see :class:`TrainingLauncher`.
    # The list of canonical modules to check is :data:`REQUIRED_SRC_MODULES`
    # below; runtime_check.py keeps a copy too for image-baked
    # standalone use, but the runtime path goes through HTTP.

    def _sync_module_tar(
        self,
        ssh_client: SSHClient,
        local: str,
        dest: str,
        ssh_opts: str,
    ) -> Result[None, AppError]:
        """Fallback path: pack ``local/`` into a tarball and stream it
        into ``<workspace>/<dest>/`` over ssh. Mirrors the rsync layout
        — local source basename is dropped on the remote so the dest
        can be renamed (e.g. ``packages/pod/src/ryotenkai_pod`` →
        ``ryotenkai_pod``) without leaking the local prefix to the pod.
        """
        local_path = Path(local)
        excludes = " ".join(f"--exclude='{p}'" for p in self.EXCLUDE_PATTERNS)
        remote_dir = f"{self._workspace}/{dest}"
        # tar from inside the source dir so paths in the archive are
        # already rooted at the package contents (no leading
        # ``packages/<pkg>/src/<name>/``).
        tar_cmd = (
            f"tar czf - --no-mac-metadata {excludes} -C {shlex.quote(str(local_path))} . 2>/dev/null | "
            f"ssh {ssh_opts} {ssh_client.ssh_target} "
            f"'mkdir -p {shlex.quote(remote_dir)} && cd {shlex.quote(remote_dir)} && tar xzf - 2>/dev/null'"
        )
        result = subprocess.run(
            tar_cmd, shell=True, capture_output=True, text=True, timeout=DEPLOYMENT_RSYNC_TIMEOUT
        )

        if result.returncode != 0:
            verify_cmd = f"test -d {shlex.quote(remote_dir)} && echo '{DEPLOYMENT_MARKER_EXISTS}'"
            success, stdout, _ = ssh_client.exec_command(
                command=verify_cmd, background=False, timeout=DEPLOYMENT_VERIFY_TIMEOUT
            )
            if not success or DEPLOYMENT_MARKER_EXISTS not in stdout:
                return Err(ProviderError(message=f"Failed to sync {local}", code="FILE_SYNC_FAILED"))

        return Ok(None)


__all__ = ["DEFAULT_WORKSPACE", "CodeSyncer"]
