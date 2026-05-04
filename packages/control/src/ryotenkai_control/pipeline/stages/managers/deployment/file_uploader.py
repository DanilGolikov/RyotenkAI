"""HTTP-based config + dataset uploader.

Phase 3 PR-3.3 of transport-unification-v2: replaces the legacy
SSH tar-pipe + per-file SCP fallback with multipart streaming
through ``JobClient.upload_file`` (PR-2.4 endpoint). The class
preserves the dataset-collection responsibility (walking the
strategy chain → mapping each ``(local_path, strategy_type)`` to
``data/{strategy_type}/{basename}``) but the actual transport is
now HTTP.

Bootstrap flow change: HTTP upload requires uvicorn to be
listening, so file upload moved AFTER ``launch_runner`` in
:class:`TrainingLauncher.start_training`. Code sync (rsync of
``packages/``) stays pre-launch — uvicorn cannot start without
the synced source. See ``docs/architecture/SSH_SURFACE.md``.

Cross-component dependency: :class:`CodeSyncer` is owned by the
deployment-manager facade and called separately as
``deploy_code`` BEFORE ``launch_runner``; this class now does
HTTP file upload only.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ryotenkai_shared.config.datasets.constants import SOURCE_TYPE_LOCAL
from ryotenkai_shared.contracts.runner_api.files import FileUploadTarget
from ryotenkai_control.pipeline.stages.managers.deployment_constants import (
    DEPLOYMENT_CONFIG_PATH,
)
from ryotenkai_shared.utils.logger import logger
from ryotenkai_shared.utils.result import AppError, ConfigError, Err, Ok, ProviderError, Result

if TYPE_CHECKING:
    from ryotenkai_shared.config import PipelineConfig, Secrets
    from ryotenkai_shared.utils.clients.job_client import JobClient


DEFAULT_WORKSPACE = "/workspace"


class FileUploader:
    """HTTP-based uploader for the runner's ``/api/v1/files/upload``.

    Constructed by :class:`TrainingDeploymentManager`. Called from
    :class:`TrainingLauncher.start_training` AFTER the runner's
    ``/healthz`` returns 200 — the upload endpoint cannot serve
    requests before uvicorn binds 8080.

    Public surface:
      * :meth:`upload_via_http(client, context)` — single entry-point.
        Walks the strategy chain to collect dataset files, uploads
        config + datasets through ``JobClient.upload_file``.
      * :meth:`set_workspace(path)` / :attr:`workspace` — kept for
        compatibility with the deployment-manager facade; the actual
        target paths inside the pod are resolved by the runner from
        ``app.state.pod_layout`` (single source of truth — the Mac
        side does NOT need to know the pod-side path).
    """

    def __init__(
        self,
        config: PipelineConfig,
        secrets: Secrets,
    ) -> None:
        self.config = config
        self.secrets = secrets
        self._workspace = DEFAULT_WORKSPACE

    @property
    def workspace(self) -> str:
        return self._workspace

    def set_workspace(self, workspace_path: str) -> None:
        self._workspace = workspace_path

    # ------------------------------------------------------------------
    # Dataset collection — preserved from the SSH-era implementation
    # ------------------------------------------------------------------

    def _collect_local_datasets(self) -> tuple[list[Path], list[str]]:
        """Walk the strategy chain and return ``(files, missing)``.

        ``files``   — list of resolved local paths to upload.
        ``missing`` — list of refs that were referenced but not found
                      on disk. Caller decides whether that's fatal.
        """
        files: list[Path] = []
        missing: list[str] = []

        strategies = self.config.training.get_strategy_chain()
        datasets_to_upload: dict[str, Any] = {}
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

        for ds_name, ds_cfg in datasets_to_upload.items():
            if not ds_cfg or ds_cfg.get_source_type() != SOURCE_TYPE_LOCAL:
                continue
            source_local = ds_cfg.source_local
            if source_local is None:
                missing.append(f"{ds_name}: missing source_local")
                continue

            for kind, ref in (
                ("train", source_local.local_paths.train),
                ("eval", source_local.local_paths.eval),
            ):
                if not ref:
                    continue
                resolved = self.config.resolve_path(ref)
                if resolved and resolved.exists():
                    files.append(Path(resolved))
                    logger.info(
                        f"📂 Dataset [{ds_name}]: {kind} {resolved} → "
                        f"data/.../{Path(ref).name}",
                    )
                else:
                    missing.append(str(ref))
                    logger.warning(
                        f"⚠️ Dataset [{ds_name}] {kind} not found: {ref}",
                    )
        return files, missing

    # ------------------------------------------------------------------
    # HTTP upload — single entry point
    # ------------------------------------------------------------------

    def upload_via_http(
        self,
        client: "JobClient",
        context: dict[str, Any],
    ) -> Result[None, AppError]:
        """Upload config + datasets via ``client.upload_file``.

        Called from :class:`TrainingLauncher.start_training` after
        ``/healthz`` returns 200. Sequential — runs each upload one
        at a time so a slow tunnel doesn't end up with N concurrent
        TCP streams competing for the same SSH ControlMaster.
        """
        config_path = Path(context.get("config_path", DEPLOYMENT_CONFIG_PATH))
        if not config_path.exists():
            return Err(
                ConfigError(
                    message=f"Config file not found: {config_path}",
                    code="CONFIG_FILE_NOT_FOUND",
                )
            )

        dataset_files, missing = self._collect_local_datasets()
        if missing and not dataset_files:
            return Err(
                ConfigError(
                    message=(
                        "Dataset files referenced but not found: "
                        + ", ".join(missing)
                    ),
                    code="DATASET_FILE_NOT_FOUND",
                    details={"missing": missing},
                )
            )

        logger.info(
            f"📤 HTTP upload: 1 config + {len(dataset_files)} dataset file(s)",
        )

        try:
            asyncio.run(
                self._upload_all(client, config_path, dataset_files),
            )
        except Exception as exc:  # noqa: BLE001 — surface uniformly
            return Err(
                ProviderError(
                    message=f"HTTP upload failed: {exc}",
                    code="HTTP_FILE_UPLOAD_FAILED",
                ),
            )

        logger.info("✅ All files uploaded via HTTP")
        return Ok(None)

    async def _upload_all(
        self,
        client: "JobClient",
        config_path: Path,
        dataset_files: list[Path],
    ) -> None:
        """Async core — one POST /files/upload per file."""
        # Config
        await client.upload_file(
            FileUploadTarget.CONFIG.value,
            config_path,
            timeout=120.0,
        )
        logger.info(f"   ✓ config: {config_path.name}")
        # Datasets
        for path in dataset_files:
            await client.upload_file(
                FileUploadTarget.DATASET.value,
                path,
                timeout=300.0,  # large datasets
            )
            logger.info(f"   ✓ dataset: {path.name}")


__all__ = ["DEFAULT_WORKSPACE", "FileUploader"]
