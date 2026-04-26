"""
ModelRetriever — Pipeline Stage 3: Model Retrieval & Upload.

Thin orchestrator that delegates to:
- HFModelUploader  (hf_uploader.py)
- ModelCardGenerator (model_card.py)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.pipeline.stages.model_retriever.constants import MR_SSH_PORT_DEFAULT
from src.pipeline.stages.base import PipelineStage
from src.pipeline.stages.constants import StageNames
from src.utils.logger import logger
from src.utils.result import AppError, Err, ModelError, Ok, Result
from src.utils.ssh_client import SSHClient

from src.pipeline.stages.model_retriever.hf_uploader import HFModelUploader
from src.pipeline.stages.model_retriever.model_card import ModelCardGenerator
from src.pipeline.stages.model_retriever.types import (
    ModelCardContext,
    ModelRetrieverEventCallbacks,
    PhaseMetricsResult,
)

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig, Secrets


class ModelRetriever(PipelineStage):
    """
    Retrieves trained model from any GPU provider.

    Provider-agnostic: uses SSH connection info from context (set by GPUDeployer).
    Uploads model to HuggingFace Hub and handles cleanup.
    """

    def __init__(
        self,
        config: PipelineConfig,
        secrets: Secrets,
        callbacks: ModelRetrieverEventCallbacks | None = None,
    ):
        super().__init__(config, StageNames.MODEL_RETRIEVER)
        self.secrets = secrets
        self._callbacks = callbacks or ModelRetrieverEventCallbacks()

        self._provider_name = config.get_active_provider_name()
        self._provider_config = config.get_provider_config()

        training_cfg_obj = None
        get_train_cfg = getattr(config, "get_provider_training_config", None)
        if callable(get_train_cfg):
            try:
                training_cfg_obj = get_train_cfg()
            except Exception:
                training_cfg_obj = None
        if not isinstance(training_cfg_obj, dict):
            training_cfg_obj = self._provider_config
        self._provider_training_cfg: dict[str, Any] = (
            training_cfg_obj if isinstance(training_cfg_obj, dict) else {}
        )
        self.mock_mode = bool(self._provider_training_cfg.get("mock_mode", False))

        # Sub-components
        self._uploader = HFModelUploader(config, secrets)
        self._card_generator = ModelCardGenerator()

        # Expose HF properties for backward compat
        hf_config = config.experiment_tracking.huggingface
        self._hf_api = self._uploader.hf_api
        self.hf_enabled = bool(hf_config and hf_config.integration)
        self._hf_repo_id: str | None = hf_config.repo_id if hf_config else None
        self._hf_private: bool | None = hf_config.private if hf_config else None

        # SSH connection info (set in execute from context)
        self.__ssh_client: SSHClient | None = None  # noqa: E501
        self.__workspace_path: str = "/workspace"
        self._resource_id: str | None = None
        # Staging area for card content generated in _upload_to_hf_from_remote delegate
        self._card_content: str = ""

    # ------------------------------------------------------------------
    # hf_api property — propagates to uploader so test mocks work
    # ------------------------------------------------------------------

    @property
    def hf_api(self) -> Any:
        return self._hf_api

    @hf_api.setter
    def hf_api(self, value: Any) -> None:
        self._hf_api = value
        self._uploader.hf_api = value

    @property
    def hf_repo_id(self) -> str | None:
        return self._hf_repo_id

    @hf_repo_id.setter
    def hf_repo_id(self, value: str | None) -> None:
        self._hf_repo_id = value
        self._uploader.hf_repo_id = value

    @property
    def hf_private(self) -> bool | None:
        return self._hf_private

    @hf_private.setter
    def hf_private(self, value: bool | None) -> None:
        self._hf_private = value
        self._uploader.hf_private = value

    @property
    def _ssh_client(self) -> SSHClient | None:
        return self.__ssh_client

    @_ssh_client.setter
    def _ssh_client(self, value: SSHClient | None) -> None:
        self.__ssh_client = value
        self._uploader._ssh_client = value

    @property
    def _workspace_path(self) -> str:
        return self.__workspace_path

    @_workspace_path.setter
    def _workspace_path(self, value: str) -> None:
        self.__workspace_path = value
        self._uploader._workspace_path = value

    # ------------------------------------------------------------------
    # Stage entry point
    # ------------------------------------------------------------------

    def execute(self, context: dict[str, Any]) -> Result[dict[str, Any], AppError]:
        """Retrieve model and clean up."""
        deployer_context = context.get(StageNames.GPU_DEPLOYER, {})

        self._resource_id = deployer_context.get("resource_id")

        ssh_host = deployer_context.get("ssh_host")
        ssh_port = deployer_context.get("ssh_port", MR_SSH_PORT_DEFAULT)
        ssh_user = deployer_context.get("ssh_user", "root")
        ssh_key_path = deployer_context.get("ssh_key_path", "")
        is_alias_mode = deployer_context.get("is_alias_mode", False)
        self._workspace_path = deployer_context.get("workspace_path", "/workspace")

        provider_info = deployer_context.get("provider_info", {})

        if not ssh_host:
            return Err(
                ModelError(
                    message="No SSH connection info found in context (GPU Deployer stage may have failed)",
                    code="MISSING_SSH_INFO",
                )
            )

        logger.info(
            f"Retrieving model from {self._provider_name}: {self._resource_id or 'unknown'}"
        )

        if provider_info.get("mock") or self.mock_mode:
            logger.info("[MOCK] Running in MOCK MODE - simulating model retrieval")
            return self._execute_mock(context, self._resource_id or "mock")

        effective_username = None if is_alias_mode else (ssh_user if ssh_user else None)
        import sys as _sys
        _pkg = _sys.modules.get("src.pipeline.stages.model_retriever")
        _SSHClientCls = getattr(_pkg, "SSHClient", SSHClient) if _pkg else SSHClient
        self._ssh_client = _SSHClientCls(
            host=str(ssh_host),
            port=int(ssh_port) if ssh_port else MR_SSH_PORT_DEFAULT,
            username=effective_username,
            key_path=ssh_key_path if ssh_key_path else None,
        )
        self._uploader.set_ssh_context(self._ssh_client, self._workspace_path)

        try:
            return self._execute_retrieval(context)
        finally:
            try:
                self._ssh_client.close_master()
            except Exception as e:
                logger.debug(f"[RETRIEVER] Failed to close SSH ControlMaster: {e}")

    def _execute_retrieval(self, context: dict[str, Any]) -> Result[dict[str, Any], AppError]:
        """Run model retrieval logic (HF upload + optional local download)."""
        local_model_path = None
        hf_uploaded = False

        model_size_mb: float | None = None
        size_result = self._get_model_size()
        if size_result.is_success():
            size_value = size_result.unwrap()  # type: ignore[union-attr]
            model_size_mb = float(size_value) if size_value is not None else None
            if model_size_mb is not None:
                logger.info(f"Model size: {model_size_mb:.1f} MB")
        else:
            size_err = size_result.unwrap_err()  # type: ignore[union-attr]
            logger.warning(f"Model size check unavailable: {size_err}. Proceeding with download.")

        upload_duration = 0.0
        if self.hf_enabled and self.hf_repo_id:
            hf_repo_id_str = self.hf_repo_id
            logger.info(f"Uploading to HF Hub: {hf_repo_id_str}...")
            if self._callbacks.on_hf_upload_started:
                self._callbacks.on_hf_upload_started(hf_repo_id_str)
            upload_start = time.time()
            try:
                upload_result = self._upload_to_hf_from_remote(context=context)
            except TypeError:
                upload_result = self._upload_to_hf_from_remote()
            upload_duration = time.time() - upload_start
            if upload_result.is_success():
                logger.info(f"Model uploaded to HF Hub: {hf_repo_id_str} ({upload_duration:.1f}s)")
                hf_uploaded = True
                if self._callbacks.on_hf_upload_completed:
                    self._callbacks.on_hf_upload_completed(hf_repo_id_str, upload_duration)
            else:
                err_msg_raw = upload_result.unwrap_err()  # type: ignore[union-attr]
                err_msg = str(err_msg_raw) if err_msg_raw is not None else "Unknown error"
                logger.warning(f"HF upload failed: {err_msg}")
                if self._callbacks.on_hf_upload_failed:
                    self._callbacks.on_hf_upload_failed(hf_repo_id_str, err_msg)
        elif not self.hf_enabled:
            logger.info("HF Hub upload disabled (huggingface.integration not set)")
        else:
            logger.warning("HF repo_id not configured, skipping HF upload")

        if not hf_uploaded:
            logger.info("Checking model size for fallback...")
            max_size_mb = 1024

            if model_size_mb is not None and model_size_mb > max_size_mb:
                logger.error(f"Model too large ({model_size_mb:.1f}MB > {max_size_mb}MB)")
                return Err(
                    ModelError(
                        message=(
                            f"Model retrieval failed: HF upload failed, "
                            f"model too large for local download ({model_size_mb:.1f}MB > {max_size_mb}MB). "
                            "TODO: Add backup storage support."
                        ),
                        code="MODEL_TOO_LARGE_FOR_LOCAL",
                        details={"model_size_mb": model_size_mb, "max_size_mb": max_size_mb},
                    )
                )

            size_info = f"{model_size_mb:.1f}MB" if model_size_mb is not None else "size unknown"
            logger.info(f"Fallback: Downloading locally ({size_info})...")
            if self._callbacks.on_local_download_started:
                self._callbacks.on_local_download_started(model_size_mb or 0.0)
            download_result = self._download_model()
            if download_result.is_failure():
                err_msg_raw = download_result.unwrap_err()  # type: ignore[union-attr]
                err_msg = str(err_msg_raw) if err_msg_raw is not None else "Unknown error"
                logger.error(f"Download failed: {err_msg}")
                if self._callbacks.on_local_download_failed:
                    self._callbacks.on_local_download_failed(err_msg)
                return Err(
                    ModelError(
                        message="Model retrieval failed: HF upload failed, local download failed",
                        code="MODEL_RETRIEVAL_FAILED",
                    )
                )
            else:
                local_model_path = download_result.unwrap()
                logger.info(f"Model saved locally: {local_model_path}")
                if self._callbacks.on_local_download_completed:
                    self._callbacks.on_local_download_completed(str(local_model_path))

        logger.info(f"Cleanup will be handled by {self._provider_name} provider")

        if hf_uploaded:
            logger.info(f"Model available on HF: https://huggingface.co/{self.hf_repo_id}")
        if local_model_path:
            logger.info(f"Model available locally: {local_model_path}")

        if self._callbacks.on_retrieval_completed:
            self._callbacks.on_retrieval_completed(
                hf_uploaded,
                str(local_model_path) if local_model_path else None,
            )

        return Ok(
            self.update_context(
                context,
                {
                    "local_model_path": str(local_model_path) if local_model_path else None,
                    "hf_repo_id": self.hf_repo_id if hf_uploaded else None,
                    "hf_uploaded": hf_uploaded,
                    "provider_name": self._provider_name,
                    "model_size_mb": model_size_mb,
                    "upload_duration_seconds": upload_duration if hf_uploaded else None,
                },
            )
        )
    def _execute_mock(
        self, context: dict[str, Any], resource_id: str
    ) -> Result[dict[str, Any], AppError]:
        """Mock execution for testing without real model retrieval."""
        logger.info(f"[MOCK] Downloading model from {self._provider_name}: {resource_id}")
        time.sleep(1)

        mock_model_path = Path("outputs/models/mock-model-checkpoint")
        mock_model_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"[MOCK] Model downloaded to: {mock_model_path}")
        logger.info(f"[MOCK] Uploading model to HF Hub: {self.hf_repo_id}")
        time.sleep(1)
        logger.info(f"[MOCK] Model uploaded to HF Hub: {self.hf_repo_id}")
        logger.info(f"[MOCK] Cleanup handled by {self._provider_name} provider")

        return Ok(
            self.update_context(
                context,
                {
                    "local_model_path": str(mock_model_path),
                    "hf_repo_id": self.hf_repo_id,
                    "provider_name": self._provider_name,
                    "mock": True,
                },
            )
        )

    # ------------------------------------------------------------------
    # Backward-compat delegates (tests call these directly on retriever)
    # ------------------------------------------------------------------

    def _upload_to_hf_from_remote(
        self,
        context: dict[str, Any] | None = None,
    ) -> Result[None, ModelError]:
        """Backward-compat delegate → HFModelUploader.upload_to_hf_from_remote.

        Calls self._ensure_hf_repo_ready() first so tests can mock it.
        Generates the model card here so test mocks bypass card generation.
        Wraps in try/except to satisfy tests expecting "direct upload failed" on exceptions.
        """
        try:
            # Repo ready check goes through the retriever method so tests can mock it
            ready_result = self._ensure_hf_repo_ready()
            if ready_result.is_failure():
                return Err(ready_result.unwrap_err())  # type: ignore[union-attr]

            remote_output_dir = f"{self._workspace_path}/output"
            phase_result = self._uploader.extract_phase_metrics(
                context=context or {},
                remote_output_dir=remote_output_dir,
            )
            card_ctx = ModelCardContext(
                phase_metrics=phase_result.phase_metrics,
                datasets=self._uploader.extract_datasets_for_readme(),
                dataset_source_type=self._uploader.extract_dataset_source_type_for_readme(),
                training_started_at=phase_result.training_started_at,
                training_completed_at=phase_result.training_completed_at,
            )
            card_content = self._card_generator.generate(
                card_ctx,
                config=self.config,
                provider_name=self._provider_name,
                provider_training_cfg=self._provider_training_cfg,
            )
            # Bypass internal repo-ready check since we already ran it above
            return self._uploader._upload_files(
                context=context, card_content=card_content
            )
        except Exception as e:
            return Err(
                ModelError(
                    message=f"Direct upload failed: {e!s}",
                    code="HF_UPLOAD_FAILED",
                )
            )

    def _download_model(self) -> Result[Path, ModelError]:
        """Backward-compat delegate → HFModelUploader.download_model."""
        return self._uploader.download_model()

    def _get_model_size(self) -> Result[float, ModelError]:
        """Backward-compat delegate → HFModelUploader.get_model_size."""
        return self._uploader.get_model_size()

    def _ensure_hf_repo_ready(self) -> Result[None, ModelError]:
        """Backward-compat delegate → HFModelUploader.ensure_hf_repo_ready."""
        return self._uploader.ensure_hf_repo_ready()

    def _resolve_checkpoint(self, remote_output_dir: str) -> str:
        """Backward-compat delegate → HFModelUploader.resolve_checkpoint."""
        return self._uploader.resolve_checkpoint(remote_output_dir)

    def _extract_phase_metrics(
        self, *, context: dict[str, Any], remote_output_dir: str
    ) -> PhaseMetricsResult:
        """Backward-compat delegate → HFModelUploader.extract_phase_metrics."""
        return self._uploader.extract_phase_metrics(
            context=context, remote_output_dir=remote_output_dir
        )

    def _generate_model_card(self, ctx: ModelCardContext | None = None) -> str:
        """Backward-compat delegate → ModelCardGenerator.generate."""
        return self._card_generator.generate(
            ctx,
            config=self.config,
            provider_name=self._provider_name,
            provider_training_cfg=self._provider_training_cfg,
        )

    def _extract_datasets_for_readme(self) -> list[str]:
        """Backward-compat delegate → HFModelUploader.extract_datasets_for_readme."""
        return self._uploader.extract_datasets_for_readme()

    def _extract_dataset_source_type_for_readme(self) -> str | None:
        """Backward-compat delegate → HFModelUploader.extract_dataset_source_type_for_readme."""
        return self._uploader.extract_dataset_source_type_for_readme()

    # Static helpers kept on class for backward compat
    @staticmethod
    def _format_strategies(cfg: Any) -> str:
        return ModelCardGenerator._format_strategies(cfg)

    @staticmethod
    def _get_lora_param(cfg: Any, param: str) -> str:
        return ModelCardGenerator._get_lora_param(cfg, param)

    @staticmethod
    def _basename(path_str: str) -> str:
        return ModelCardGenerator._basename(path_str)

    @staticmethod
    def _sha12(text: str) -> str:
        return HFModelUploader._sha12(text)


__all__ = ["ModelRetriever"]
