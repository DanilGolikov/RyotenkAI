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

from ryotenkai_control.pipeline.stages.model_retriever.constants import MR_SSH_PORT_DEFAULT
from ryotenkai_control.pipeline.stages.base import PipelineStage
from ryotenkai_control.pipeline.stages.constants import PipelineContextKeys, StageNames
from ryotenkai_shared.errors import (
    ModelLoadFailedError,
    RyotenkAIError,
)
from ryotenkai_shared.utils.logger import logger
from ryotenkai_shared.utils.ssh_client import SSHClient

from ryotenkai_control.pipeline.stages.model_retriever.hf_uploader import HFModelUploader
from ryotenkai_control.pipeline.stages.model_retriever.metrics_buffer_retriever import (
    MetricsBufferRetriever,
)
from ryotenkai_control.pipeline.stages.model_retriever.metrics_replay import (
    BufferedMetricsReplay,
)
from ryotenkai_control.pipeline.stages.model_retriever.model_card import ModelCardGenerator
from ryotenkai_control.pipeline.stages.model_retriever.types import (
    ModelCardContext,
    ModelRetrieverEventCallbacks,
    PhaseMetricsResult,
)

if TYPE_CHECKING:
    from ryotenkai_shared.config import PipelineConfig, Secrets


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

        # Mock mode lookup tolerates both the typed Pydantic schema
        # (post PipelineConfig validator) and legacy dict blocks.
        from pydantic import BaseModel

        training_cfg_obj: Any = None
        get_train_cfg = getattr(config, "get_provider_training_config", None)
        if callable(get_train_cfg):
            try:
                training_cfg_obj = get_train_cfg()
            except Exception:
                training_cfg_obj = None
        if training_cfg_obj is None:
            training_cfg_obj = self._provider_config

        mock_mode = False
        if isinstance(training_cfg_obj, BaseModel):
            mock_mode = bool(getattr(training_cfg_obj, "mock_mode", False))
        elif isinstance(training_cfg_obj, dict):
            mock_mode = bool(training_cfg_obj.get("mock_mode", False))
        self.mock_mode = mock_mode

        # Kept for stages that still want the raw dict view; preserves
        # the legacy public attribute even when training_cfg_obj is a
        # typed schema.
        if isinstance(training_cfg_obj, BaseModel):
            self._provider_training_cfg: dict[str, Any] = training_cfg_obj.model_dump(mode="json")
        elif isinstance(training_cfg_obj, dict):
            self._provider_training_cfg = training_cfg_obj
        else:
            self._provider_training_cfg = {}

        # Sub-components
        self._uploader = HFModelUploader(config, secrets)
        self._card_generator = ModelCardGenerator()

        # Expose HF properties for backward compat
        hf_config = config.integrations.huggingface
        self._hf_api = self._uploader.hf_api
        self.hf_enabled = bool(hf_config and hf_config.repo_id)
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

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Retrieve model and clean up.

        Returns:
            Updated pipeline context dict on success.

        Raises:
            ModelLoadFailedError / HFAuthFailedError / HFNotFoundError:
                per failure mode in HF upload / local download.
        """
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
            raise ModelLoadFailedError(
                detail="No SSH connection info found in context (GPU Deployer stage may have failed)",
                context={"legacy_code": "MISSING_SSH_INFO"},
            )

        logger.info(
            f"Retrieving model from {self._provider_name}: {self._resource_id or 'unknown'}"
        )

        if provider_info.get("mock") or self.mock_mode:
            logger.info("[MOCK] Running in MOCK MODE - simulating model retrieval")
            return self._execute_mock(context, self._resource_id or "mock")

        effective_username = None if is_alias_mode else (ssh_user if ssh_user else None)
        import sys as _sys
        _pkg = _sys.modules.get("ryotenkai_control.pipeline.stages.model_retriever")
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

    def _execute_retrieval(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run model retrieval logic (HF upload + optional local download).

        Raises:
            ModelLoadFailedError: when both HF upload and local download fail,
                or the model is too large to fall back to local.
        """
        local_model_path = None
        hf_uploaded = False

        model_size_mb: float | None = None
        try:
            model_size_mb = float(self._get_model_size())
            if model_size_mb is not None:
                logger.info(f"Model size: {model_size_mb:.1f} MB")
        except RyotenkAIError as size_err:
            logger.warning(f"Model size check unavailable: {size_err}. Proceeding with download.")

        upload_duration = 0.0
        if self.hf_enabled and self.hf_repo_id:
            hf_repo_id_str = self.hf_repo_id
            logger.info(f"Uploading to HF Hub: {hf_repo_id_str}...")
            if self._callbacks.on_hf_upload_started:
                self._callbacks.on_hf_upload_started(hf_repo_id_str)
            upload_start = time.time()
            try:
                try:
                    self._upload_to_hf_from_remote(context=context)
                except TypeError:
                    self._upload_to_hf_from_remote()
                upload_duration = time.time() - upload_start
                logger.info(f"Model uploaded to HF Hub: {hf_repo_id_str} ({upload_duration:.1f}s)")
                hf_uploaded = True
                if self._callbacks.on_hf_upload_completed:
                    self._callbacks.on_hf_upload_completed(hf_repo_id_str, upload_duration)
            except RyotenkAIError as upload_exc:
                upload_duration = time.time() - upload_start
                err_msg = upload_exc.detail or str(upload_exc)
                logger.warning(f"HF upload failed: {err_msg}")
                if self._callbacks.on_hf_upload_failed:
                    self._callbacks.on_hf_upload_failed(hf_repo_id_str, err_msg)
        elif not self.hf_enabled:
            logger.info("HF Hub upload disabled (integrations.huggingface.repo_id not set)")
        else:
            logger.warning("HF repo_id not configured, skipping HF upload")

        if not hf_uploaded:
            logger.info("Checking model size for fallback...")
            max_size_mb = 1024

            if model_size_mb is not None and model_size_mb > max_size_mb:
                logger.error(f"Model too large ({model_size_mb:.1f}MB > {max_size_mb}MB)")
                raise ModelLoadFailedError(
                    detail=(
                        f"Model retrieval failed: HF upload failed, "
                        f"model too large for local download "
                        f"({model_size_mb:.1f}MB > {max_size_mb}MB). "
                        "TODO: Add backup storage support."
                    ),
                    context={
                        "legacy_code": "MODEL_TOO_LARGE_FOR_LOCAL",
                        "model_size_mb": model_size_mb,
                        "max_size_mb": max_size_mb,
                    },
                )

            size_info = f"{model_size_mb:.1f}MB" if model_size_mb is not None else "size unknown"
            logger.info(f"Fallback: Downloading locally ({size_info})...")
            if self._callbacks.on_local_download_started:
                self._callbacks.on_local_download_started(model_size_mb or 0.0)
            try:
                local_model_path = self._download_model()
                logger.info(f"Model saved locally: {local_model_path}")
                if self._callbacks.on_local_download_completed:
                    self._callbacks.on_local_download_completed(str(local_model_path))
            except RyotenkAIError as dl_exc:
                err_msg = dl_exc.detail or str(dl_exc)
                logger.error(f"Download failed: {err_msg}")
                if self._callbacks.on_local_download_failed:
                    self._callbacks.on_local_download_failed(err_msg)
                raise ModelLoadFailedError(
                    detail="Model retrieval failed: HF upload failed, local download failed",
                    context={"legacy_code": "MODEL_RETRIEVAL_FAILED"},
                    cause=dl_exc,
                ) from dl_exc

        # Phase 12.A.1 — best-effort buffered MLflow metrics replay.
        # Runs AFTER HF upload / local download (so adapters are safe)
        # but BEFORE provider.cleanup_pod (so /workspace still exists).
        # Failure of this step is intentionally non-fatal — the run
        # already shipped its model artefacts; replay is opportunistic
        # data recovery for the metrics dimension.
        try:
            self._retrieve_and_replay_metrics_buffer(context)
        except Exception as exc:  # noqa: BLE001 — fail-open by contract
            logger.warning(
                "[METRICS_REPLAY] retrieval+replay raised unexpectedly: %s. "
                "Continuing — model artefacts already secured.", exc,
            )

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

        return self.update_context(
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
    # ------------------------------------------------------------------
    # Phase 12.A.1 — metrics buffer retrieval + replay
    # ------------------------------------------------------------------

    def _retrieve_and_replay_metrics_buffer(
        self, context: dict[str, Any]
    ) -> None:
        """Retrieve ``metrics_buffer.jsonl`` from pod and replay into
        MLflow.

        Best-effort. Never raises — caller wraps in defensive try/except
        as a second line of defence. Logs every outcome (missing,
        oversized, fetch error, replay error, success) so operator
        sees exactly what happened in pipeline.log.

        Skipped silently when:
        * SSH client is None (mock mode / no-pod paths).
        * No MLflow run id available on context (MLflow tracking
          disabled in this run).
        """
        if self._ssh_client is None:
            logger.debug(
                "[METRICS_REPLAY] no SSH client — skipping (mock mode?)"
            )
            return

        run_id = self._resolve_mlflow_run_id(context)
        attempt_dir = self._resolve_attempt_directory(context)
        if attempt_dir is None:
            logger.info(
                "[METRICS_REPLAY] no attempt directory in context — skipping"
            )
            return

        # 1. Retrieve the buffer file.
        retriever = MetricsBufferRetriever(
            self._ssh_client,
            workspace_path=self._workspace_path,
        )
        fetch = retriever.fetch(local_dir=attempt_dir)

        # Healthy case: trainer's drain succeeded — buffer absent.
        if fetch.missing:
            if fetch.error:
                logger.info(
                    "[METRICS_REPLAY] buffer probe failed (%s) — skipping",
                    fetch.error,
                )
            else:
                logger.info(
                    "[METRICS_REPLAY] no buffered metrics on pod — "
                    "trainer drain already succeeded"
                )
            if self._callbacks.on_metrics_buffer_retrieved:
                self._callbacks.on_metrics_buffer_retrieved(
                    0, 0, 0, True, False,
                )
            return

        # Operator-visible: buffer existed but was deliberately skipped.
        if fetch.oversized:
            logger.warning(
                "[METRICS_REPLAY] buffer oversized (%d bytes) — skipped",
                fetch.size_bytes,
            )
            if self._callbacks.on_metrics_buffer_retrieved:
                self._callbacks.on_metrics_buffer_retrieved(
                    0, 0, fetch.size_bytes, False, True,
                )
            return

        # Download failed for unexpected reason.
        if fetch.local_path is None:
            logger.warning(
                "[METRICS_REPLAY] fetch failed: %s",
                fetch.error or "unknown error",
            )
            if self._callbacks.on_metrics_buffer_retrieved:
                self._callbacks.on_metrics_buffer_retrieved(
                    0, 0, fetch.size_bytes, False, False,
                )
            return

        # 2. Buffer is on Mac. Replay into MLflow.
        if not run_id:
            # We have data but no run to write into. Keep the file
            # around for forensics — the operator can manually replay
            # it later via `mlflow.tracking.MlflowClient.log_metric`.
            logger.warning(
                "[METRICS_REPLAY] %d buffered records retrieved but no "
                "MLflow run_id available — preserved at %s",
                fetch.line_count, fetch.local_path,
            )
            if self._callbacks.on_metrics_buffer_retrieved:
                self._callbacks.on_metrics_buffer_retrieved(
                    0, fetch.line_count, fetch.size_bytes, False, False,
                )
            return

        client = self._build_mlflow_client()
        if client is None:
            logger.warning(
                "[METRICS_REPLAY] MlflowClient unavailable; %d records "
                "preserved locally at %s for manual replay",
                fetch.line_count, fetch.local_path,
            )
            if self._callbacks.on_metrics_buffer_retrieved:
                self._callbacks.on_metrics_buffer_retrieved(
                    0, fetch.line_count, fetch.size_bytes, False, False,
                )
            return

        replayer = BufferedMetricsReplay(client)
        result = replayer.replay(
            buffer_path=fetch.local_path,
            run_id=run_id,
        )

        if self._callbacks.on_metrics_buffer_retrieved:
            self._callbacks.on_metrics_buffer_retrieved(
                result.replayed,
                fetch.line_count,
                fetch.size_bytes,
                False,
                False,
            )

        logger.info(
            "[METRICS_REPLAY] replayed %d/%d metrics (run=%s, %dms)",
            result.replayed, fetch.line_count, run_id, result.duration_ms,
        )

    @staticmethod
    def _resolve_mlflow_run_id(context: dict[str, Any]) -> str | None:
        """Pick the best MLflow run id to replay into.

        Phase 12.A.1 ships with a single source of truth: the parent
        run id (``MLFLOW_PARENT_RUN_ID``). Multi-phase nested metrics
        all land in the parent, which is the same place HF Trainer's
        own MLflow callback writes them — Mac-side replay matches
        that semantic. A future Phase 12.A.3 can extend the buffer
        format with a per-record ``run_id`` if nested-run granularity
        becomes important.
        """
        run_id = context.get(PipelineContextKeys.MLFLOW_PARENT_RUN_ID)
        if isinstance(run_id, str) and run_id.strip():
            return run_id.strip()
        # Fallback: some test contexts use the explicit string key.
        run_id_alt = context.get("mlflow_parent_run_id") or context.get(
            "mlflow_run_id"
        )
        if isinstance(run_id_alt, str) and run_id_alt.strip():
            return run_id_alt.strip()
        return None

    @staticmethod
    def _resolve_attempt_directory(
        context: dict[str, Any],
    ) -> Path | None:
        raw = context.get(PipelineContextKeys.ATTEMPT_DIRECTORY) or context.get(
            "attempt_directory"
        )
        if not raw:
            return None
        try:
            return Path(str(raw))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_mlflow_client() -> Any | None:
        """Construct an :class:`mlflow.tracking.MlflowClient` if MLflow
        is importable. Returns ``None`` when the package is absent or
        construction fails (e.g. malformed tracking URI). Replay then
        gracefully no-ops with the buffer file preserved locally.
        """
        try:
            from mlflow.tracking import MlflowClient
        except ImportError:
            return None
        try:
            return MlflowClient()
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning(
                "[METRICS_REPLAY] MlflowClient construction failed: %s", exc,
            )
            return None

    def _execute_mock(
        self, context: dict[str, Any], resource_id: str
    ) -> dict[str, Any]:
        """Mock execution for testing without real model retrieval."""
        logger.info(f"[MOCK] Downloading model from {self._provider_name}: {resource_id}")
        time.sleep(1)

        from ryotenkai_shared.config.runtime import workspace_root
        mock_model_path = workspace_root() / "outputs/models/mock-model-checkpoint"
        mock_model_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"[MOCK] Model downloaded to: {mock_model_path}")
        logger.info(f"[MOCK] Uploading model to HF Hub: {self.hf_repo_id}")
        time.sleep(1)
        logger.info(f"[MOCK] Model uploaded to HF Hub: {self.hf_repo_id}")
        logger.info(f"[MOCK] Cleanup handled by {self._provider_name} provider")

        return self.update_context(
            context,
            {
                "local_model_path": str(mock_model_path),
                "hf_repo_id": self.hf_repo_id,
                "provider_name": self._provider_name,
                "mock": True,
            },
        )

    # ------------------------------------------------------------------
    # Backward-compat delegates (tests call these directly on retriever)
    # ------------------------------------------------------------------

    def _upload_to_hf_from_remote(
        self,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Backward-compat delegate → HFModelUploader.upload_to_hf_from_remote.

        Calls self._ensure_hf_repo_ready() first so tests can mock it.
        Generates the model card here so test mocks bypass card generation.

        Raises:
            HFAuthFailedError / HFNotFoundError / ModelLoadFailedError per failure mode.
        """
        try:
            # Repo ready check goes through the retriever method so tests can mock it
            self._ensure_hf_repo_ready()

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
            self._uploader._upload_files(
                context=context, card_content=card_content
            )
            return None
        except RyotenkAIError:
            raise
        except Exception as e:
            raise ModelLoadFailedError(
                detail=f"Direct upload failed: {e!s}",
                context={"legacy_code": "HF_UPLOAD_FAILED"},
                cause=e,
            ) from e

    def _download_model(self) -> Path:
        """Backward-compat delegate → HFModelUploader.download_model.

        Raises:
            ModelLoadFailedError: on download failure.
        """
        return self._uploader.download_model()

    def _get_model_size(self) -> float:
        """Backward-compat delegate → HFModelUploader.get_model_size.

        Raises:
            ModelLoadFailedError: when size probe fails.
        """
        return self._uploader.get_model_size()

    def _ensure_hf_repo_ready(self) -> None:
        """Backward-compat delegate → HFModelUploader.ensure_hf_repo_ready.

        Raises:
            HFAuthFailedError / HFNotFoundError / ModelLoadFailedError per failure mode.
        """
        self._uploader.ensure_hf_repo_ready()

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
