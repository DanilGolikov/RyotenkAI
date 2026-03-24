"""
Stage 3: Model Retriever & Cleanup

Provider-agnostic model retrieval from any GPU provider (RunPod, SingleNode, etc.)
Uploads model to HuggingFace Hub and handles cleanup.

Uses SSH connection info from context (set by GPUDeployer stage).
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import HfApi

from src.pipeline.constants import (
    HTTP_STATUS_NOT_FOUND,
    HTTP_STATUS_UNAUTHORIZED,
    MR_SHA12_LENGTH,
    MR_SSH_CMD_TIMEOUT,
    MR_SSH_PORT_DEFAULT,
    MR_UPLOAD_TIMEOUT,
)
from src.pipeline.stages.base import PipelineStage
from src.pipeline.stages.constants import StageNames
from src.utils.logger import logger
from src.utils.result import AppError, Err, ModelError, Ok, Result
from src.utils.ssh_client import SSHClient

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.utils.config import PipelineConfig, Secrets


# Keys in DataBuffer `pipeline_state.json` used for README generation.
_PIPELINE_STATE_STARTED_AT = "started_at"
_PIPELINE_STATE_COMPLETED_AT = "completed_at"
_PHASE_IDX = "phase_idx"
_STRATEGY_TYPE = "strategy_type"
_STATUS = "status"
_METRICS = "metrics"


# =============================================================================
# EVENT CALLBACKS
# =============================================================================


@dataclass
class ModelRetrieverEventCallbacks:
    """
    Callbacks for ModelRetriever events (SOLID-compliant event collection).

    Used to integrate ModelRetriever with MLflow or other logging systems.
    """

    # HF upload started event
    on_hf_upload_started: Callable[[str], None] | None = None
    # Args: repo_id

    # HF upload completed event
    on_hf_upload_completed: Callable[[str, float], None] | None = None
    # Args: repo_id, duration_seconds

    # HF upload failed event
    on_hf_upload_failed: Callable[[str, str], None] | None = None
    # Args: repo_id, error

    # Local download started event
    on_local_download_started: Callable[[float], None] | None = None
    # Args: size_mb

    # Local download completed event
    on_local_download_completed: Callable[[str], None] | None = None
    # Args: local_path

    # Local download failed event
    on_local_download_failed: Callable[[str], None] | None = None
    # Args: error

    # Retrieval completed event
    on_retrieval_completed: Callable[[bool, str | None], None] | None = None
    # Args: hf_uploaded, local_path


@dataclass(frozen=True)
class ModelCardContext:
    """
    Data required to generate a HuggingFace model card (README.md).

    This is intentionally a small, explicit interface: ModelRetriever should NOT re-run any
    expensive computations while generating the model card.
    """

    phase_metrics: list[dict[str, Any]]
    datasets: list[str]
    dataset_source_type: str | None = None
    training_started_at: str | None = None
    training_completed_at: str | None = None


@dataclass(frozen=True)
class PhaseMetricsResult:
    """Return value of _extract_phase_metrics."""

    phase_metrics: list[dict[str, Any]]
    training_started_at: str | None
    training_completed_at: str | None


class ModelRetriever(PipelineStage):
    """
    Retrieves trained model from any GPU provider.

    Provider-agnostic: uses SSH connection info from context.
    Uploads model to HuggingFace Hub and handles cleanup.

    TODO: Add support for alternative adapter export targets (local path, S3, custom SSH host)
          to allow users to download adapters to locations other than HuggingFace Hub.
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

        # Get provider config and name
        self._provider_name = config.get_active_provider_name()
        self._provider_config = config.get_provider_config()
        # Prefer training-scoped provider view (new schemas), fallback to legacy flat provider dict.
        # NOTE: MagicMock returns "hasattr=True" for everything, so we check for dict result.
        training_cfg_obj = None
        get_train_cfg = getattr(config, "get_provider_training_config", None)
        if callable(get_train_cfg):
            try:
                training_cfg_obj = get_train_cfg()
            except Exception:
                training_cfg_obj = None
        if not isinstance(training_cfg_obj, dict):
            training_cfg_obj = self._provider_config
        self._provider_training_cfg: dict[str, Any] = training_cfg_obj if isinstance(training_cfg_obj, dict) else {}
        self.mock_mode = bool(self._provider_training_cfg.get("mock_mode", False))

        # HuggingFace API
        self.hf_api = HfApi(token=secrets.hf_token)
        hf_config = config.experiment_tracking.huggingface
        self.hf_enabled = hf_config.enabled if hf_config else False
        self.hf_repo_id = hf_config.repo_id if hf_config else None
        self.hf_private = hf_config.private if hf_config else None

        # SSH connection info (set in execute from context)
        self._ssh_client: SSHClient | None = None
        self._workspace_path: str = "/workspace"
        self._resource_id: str | None = None

    @staticmethod
    def _format_strategies(cfg: PipelineConfig) -> str:
        """Format strategy chain for display."""
        strategies = cfg.training.get_strategy_chain()
        if not strategies:
            return "SFT (default)"
        parts: list[str] = []
        for phase in strategies:
            # NOTE: after hyperparams unification, epochs live in `phase.hyperparams.epochs`.
            # Fall back to global defaults (training.hyperparams.epochs) if phase override is missing.
            epochs = phase.hyperparams.epochs
            if epochs is None:
                epochs = cfg.training.hyperparams.epochs
            parts.append(
                f"{phase.strategy_type.upper()} ({epochs}ep)" if epochs is not None else phase.strategy_type.upper()
            )
        return " → ".join(parts)

    @staticmethod
    def _get_lora_param(cfg: PipelineConfig, param: str) -> str:
        """Get LoRA/QLoRA parameter safely based on training type."""
        try:
            adapter_config = cfg.get_adapter_config()
            return str(getattr(adapter_config, param, "N/A"))
        except (AttributeError, ValueError):
            return "N/A"

    @staticmethod
    def _basename(path_str: str) -> str:
        """Return the filename portion of a path string, handling both POSIX and Windows separators."""
        s = path_str.strip()
        if not s:
            return ""
        name = Path(s).name
        if "/" in name or "\\" in name:
            name = name.split("/")[-1].split("\\")[-1]
        if not name:
            s2 = s.rstrip("/\\")
            name = s2.split("/")[-1].split("\\")[-1] if s2 else ""
        return name

    def _extract_dataset_source_type_for_readme(self) -> str | None:
        """
        Extract dataset source type for README rendering.

        Returns:
            - "huggingface" | "local" when available
            - Any non-empty string returned by dataset config (future-proof)
            - None on errors
        """
        try:
            default_ds = self.config.get_primary_dataset()
        except (AttributeError, KeyError, ValueError):
            return None

        try:
            source_type = default_ds.get_source_type()
        except (AttributeError, TypeError):
            return None

        if not (isinstance(source_type, str) and source_type.strip()):
            return None
        return source_type.strip()

    def _extract_datasets_for_readme(self) -> list[str]:
        """
        Extract dataset identifiers suitable for HuggingFace model card metadata.

        Policy:
        - If the primary dataset source is HuggingFace → return its train_id (and eval_id if present).
        - If dataset is local files → return only the local filename(s) (train + optional eval).
        """
        try:
            default_ds = self.config.get_primary_dataset()
        except (AttributeError, KeyError, ValueError):
            return []

        try:
            source_type = default_ds.get_source_type()
            out: list[str] = []

            if source_type == "huggingface" and default_ds.source_hf is not None:
                train_id = getattr(default_ds.source_hf, "train_id", None)
                eval_id = getattr(default_ds.source_hf, "eval_id", None)
                if isinstance(train_id, str) and train_id.strip():
                    out.append(train_id.strip())
                if isinstance(eval_id, str) and eval_id.strip() and eval_id.strip() not in out:
                    out.append(eval_id.strip())

            elif source_type == "local" and default_ds.source_local is not None:
                local_paths = getattr(default_ds.source_local, "local_paths", None)
                train_path = getattr(local_paths, "train", None) if local_paths is not None else None
                eval_path = getattr(local_paths, "eval", None) if local_paths is not None else None

                if isinstance(train_path, str) and train_path.strip():
                    name = self._basename(train_path)
                    if name:
                        out.append(name)
                if isinstance(eval_path, str) and eval_path.strip():
                    name = self._basename(eval_path)
                    if name and name not in out:
                        out.append(name)

            return out
        except (AttributeError, TypeError, KeyError):
            return []

    def _extract_phase_metrics(
        self,
        *,
        context: dict[str, Any],
        remote_output_dir: str,
    ) -> PhaseMetricsResult:
        """
        Extract training phase metrics for README generation.

        Primary source: remote `pipeline_state.json` (DataBuffer state) created during training.
        Fallbacks: context-supplied lists (if provided by future stages / integrations).
        """
        # Fallback 1: context (future-proof)
        for key in ("phase_metrics", "training_phase_metrics", "phase_runs_metrics"):
            v = context.get(key)
            if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                return PhaseMetricsResult(
                    phase_metrics=[dict(x) for x in v],
                    training_started_at=None,
                    training_completed_at=None,
                )

        # Primary: remote pipeline_state.json
        if self._ssh_client:
            state_path = f"{remote_output_dir}/pipeline_state.json"
            ok, stdout, _stderr = self._ssh_client.exec_command(
                command=f"cat {state_path} 2>/dev/null || true",
                background=False,
                timeout=MR_SSH_CMD_TIMEOUT,
            )
            raw = stdout.strip() if ok and isinstance(stdout, str) else ""
            if raw:
                try:
                    state = json.loads(raw)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(f"[RETRIEVER] Failed to parse pipeline_state.json: {e}")
                    state = None

                if isinstance(state, dict):
                    started_at_raw = state.get(_PIPELINE_STATE_STARTED_AT)
                    started_at = started_at_raw if isinstance(started_at_raw, str) else None
                    completed_at_raw = state.get(_PIPELINE_STATE_COMPLETED_AT)
                    completed_at = completed_at_raw if isinstance(completed_at_raw, str) else None

                    phase_metrics: list[dict[str, Any]] = []
                    phases = state.get("phases")
                    if isinstance(phases, list):
                        for p in phases:
                            if not isinstance(p, dict):
                                continue
                            row: dict[str, Any] = {}

                            phase_idx = p.get(_PHASE_IDX)
                            if isinstance(phase_idx, int):
                                row[_PHASE_IDX] = phase_idx

                            strategy_type = p.get(_STRATEGY_TYPE)
                            if isinstance(strategy_type, str):
                                row[_STRATEGY_TYPE] = strategy_type

                            status = p.get(_STATUS)
                            if isinstance(status, str):
                                row[_STATUS] = status

                            started_at_phase = p.get(_PIPELINE_STATE_STARTED_AT)
                            if isinstance(started_at_phase, str):
                                row[_PIPELINE_STATE_STARTED_AT] = started_at_phase

                            completed_at_phase = p.get(_PIPELINE_STATE_COMPLETED_AT)
                            if isinstance(completed_at_phase, str):
                                row[_PIPELINE_STATE_COMPLETED_AT] = completed_at_phase

                            m = p.get(_METRICS)
                            if isinstance(m, dict):
                                row.update(m)
                            phase_metrics.append(row)

                    phase_metrics.sort(key=lambda x: int(x.get(_PHASE_IDX, 0) or 0))
                    return PhaseMetricsResult(
                        phase_metrics=phase_metrics,
                        training_started_at=started_at,
                        training_completed_at=completed_at,
                    )

        return PhaseMetricsResult(phase_metrics=[], training_started_at=None, training_completed_at=None)

    def _resolve_checkpoint(self, remote_output_dir: str) -> str:
        """
        Resolve the best checkpoint directory under remote_output_dir.

        Priority:
        1. `checkpoint-final` (explicit final marker)
        2. Latest `checkpoint-*` by version sort
        3. remote_output_dir itself as last resort
        """
        if not self._ssh_client:
            return remote_output_dir

        ok, out, _ = self._ssh_client.exec_command(
            command=f"find {remote_output_dir} -type d -name 'checkpoint-final' 2>/dev/null | head -1",
            background=False,
            timeout=MR_SSH_CMD_TIMEOUT,
        )
        if ok and out.strip():
            logger.info(f"Found checkpoint: {out.strip()}")
            return out.strip()

        ok, out, _ = self._ssh_client.exec_command(
            command=f"find {remote_output_dir} -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1",
            background=False,
            timeout=MR_SSH_CMD_TIMEOUT,
        )
        candidate = out.strip() if ok and out.strip() else remote_output_dir
        logger.warning(f"checkpoint-final not found, using: {candidate}")
        return candidate

    def _ensure_hf_repo_ready(self) -> Result[None, ModelError]:
        """
        Ensure HF repo exists and matches requested visibility.

        IMPORTANT:
        - `create_repo(..., exist_ok=True)` does NOT change visibility of an existing repo.
        - To enforce `private: true/false`, we must call `update_repo_settings(private=...)`.
        """
        hf_cfg = self.config.experiment_tracking.huggingface
        if not hf_cfg or not hf_cfg.enabled:
            return Err(ModelError(message="HuggingFace upload disabled", code="HF_UPLOAD_DISABLED"))
        if not self.hf_repo_id:
            return Err(ModelError(message="HF repo_id not configured", code="HF_REPO_ID_MISSING"))

        try:
            # Avoid requiring "create repo" permissions when repo already exists.
            # Some orgs allow pushing to an existing repo but disallow creating new repos.
            # `create_repo(..., exist_ok=True)` still hits `/api/repos/create` and can fail with 401/403
            # even if the repo exists, so we check existence first.
            repo_exists = True
            try:
                _ = self.hf_api.repo_info(repo_id=self.hf_repo_id, repo_type="model")
                # Repo exists, skip create
            except Exception as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == HTTP_STATUS_NOT_FOUND:
                    # Repo doesn't exist, create it
                    repo_exists = False
                else:
                    raise

            if not repo_exists:
                # Create if missing
                self.hf_api.create_repo(
                    repo_id=self.hf_repo_id,
                    private=hf_cfg.private,
                    exist_ok=True,
                    repo_type="model",
                )

            # Best-effort visibility enforcement:
            # updating settings may require admin rights in orgs; do not block upload.
            try:
                self.hf_api.update_repo_settings(
                    repo_id=self.hf_repo_id,
                    private=hf_cfg.private,
                )
            except Exception as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                hint = ""
                if status == HTTP_STATUS_UNAUTHORIZED:
                    hint = (
                        " | 401 Unauthorized: HF_TOKEN was not accepted by the Hub. "
                        "Common cause: HF_TOKEN in the environment overrides secrets.env. "
                        "Fix: `unset HF_TOKEN` before running the pipeline, or export a valid token."
                    )
                logger.warning(
                    f"⚠️ Failed to update HF repo settings for '{self.hf_repo_id}': {e!s}{hint} (continuing upload)"
                )

            return Ok(None)
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            hint_parts: list[str] = []

            if status in (HTTP_STATUS_UNAUTHORIZED, 403) and self.hf_repo_id:
                namespace = self.hf_repo_id.split("/", 1)[0]
                if status == HTTP_STATUS_UNAUTHORIZED:
                    hint_parts.append(
                        "401 Unauthorized: token was not accepted by the Hub. "
                        "Verify HF_TOKEN is correct, with no quotes/spaces/newlines, "
                        "and that the token has access to this namespace (org/user)."
                    )
                else:
                    hint_parts.append(
                        "403 Forbidden: token is valid but lacks permission for the repo/namespace. "
                        "Check your role in the organization and token policies."
                    )

                # Best-effort diagnostics (no secrets in logs):
                try:
                    who = self.hf_api.whoami()
                    username = who.get("name") or who.get("username") or who.get("fullname") or "unknown"
                    orgs_raw = who.get("orgs") or []
                    orgs = [orgs_raw] if isinstance(orgs_raw, str) else list(orgs_raw)
                    hint_parts.append(f"whoami={username}, orgs={orgs}")
                    if namespace and namespace not in {username, *orgs}:
                        hint_parts.append(
                            f"repo_id namespace '{namespace}' does not match token whoami/orgs "
                            "(org/username may be wrong)."
                        )
                except Exception as who_err:
                    hint_parts.append(f"whoami_check_failed: {who_err!s}")

            hint = f" | {' ; '.join(hint_parts)}" if hint_parts else ""
            return Err(
                ModelError(
                    message=f"Failed to prepare HF repo '{self.hf_repo_id}': {e!s}{hint}", code="HF_REPO_PREPARE_FAILED"
                )
            )

    @staticmethod
    def _sha12(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:MR_SHA12_LENGTH]

    def execute(self, context: dict[str, Any]) -> Result[dict[str, Any], AppError]:
        """
        Retrieve model and clean up.

        Provider-agnostic: uses SSH from context (set by GPUDeployer).

        Strategy:
        1. Try upload to HF directly from remote
        2. If upload failed:
           - If size < 1GB → download to local machine (fallback)
           - If size > 1GB → TODO: need backup storage

        Args:
            context: Pipeline context with SSH connection info

        Returns:
            Result with model location info or error message
        """
        # Get connection info from GPU Deployer stage context
        deployer_context = context.get(StageNames.GPU_DEPLOYER, {})

        # Resource ID (pod_id for RunPod, run_dir for SingleNode)
        self._resource_id = deployer_context.get("resource_id")

        # SSH connection info
        ssh_host = deployer_context.get("ssh_host")
        ssh_port = deployer_context.get("ssh_port", MR_SSH_PORT_DEFAULT)
        ssh_user = deployer_context.get("ssh_user", "root")
        ssh_key_path = deployer_context.get("ssh_key_path", "")
        is_alias_mode = deployer_context.get("is_alias_mode", False)
        self._workspace_path = deployer_context.get("workspace_path", "/workspace")

        # Provider info for mock mode check
        provider_info = deployer_context.get("provider_info", {})

        # Validate we have connection info
        if not ssh_host:
            return Err(
                ModelError(
                    message="No SSH connection info found in context (GPU Deployer stage may have failed)",
                    code="MISSING_SSH_INFO",
                )
            )

        logger.info(f"Retrieving model from {self._provider_name}: {self._resource_id or 'unknown'}")

        # Mock mode for testing
        if provider_info.get("mock") or self.mock_mode:
            logger.info("[MOCK] Running in MOCK MODE - simulating model retrieval")
            return self._execute_mock(context, self._resource_id or "mock")

        # Create SSH client from context
        # For alias mode, username should be None (SSH will use ~/.ssh/config)
        effective_username = None if is_alias_mode else (ssh_user if ssh_user else None)
        self._ssh_client = SSHClient(
            host=str(ssh_host),
            port=int(ssh_port) if ssh_port else MR_SSH_PORT_DEFAULT,
            username=effective_username,
            key_path=ssh_key_path if ssh_key_path else None,
        )

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

        # Get model size first (for logging and context)
        model_size_mb: float = 0.0
        size_result = self._get_model_size()
        if size_result.is_success():
            size_value = size_result.unwrap()  # type: ignore[union-attr]
            model_size_mb = float(size_value) if size_value is not None else 0.0
            logger.info(f"Model size: {model_size_mb:.1f} MB")

        # Step 1: Try upload to HF directly from remote
        upload_duration = 0.0
        if self.hf_enabled and self.hf_repo_id:
            hf_repo_id_str = self.hf_repo_id  # Narrow type for callbacks
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
            logger.info("HF Hub upload disabled (huggingface.enabled: false)")
        else:
            logger.warning("HF repo_id not configured, skipping HF upload")

        # Step 2: If HF upload failed → fallback based on size
        if not hf_uploaded:
            logger.info("Checking model size for fallback...")
            max_size_mb = 1024  # 1GB limit

            if model_size_mb <= max_size_mb:
                # Fallback: Download to local machine
                logger.info(f"Fallback: Downloading locally ({model_size_mb:.1f}MB < {max_size_mb}MB)...")
                if self._callbacks.on_local_download_started:
                    self._callbacks.on_local_download_started(model_size_mb)
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
            else:
                # TODO: Need backup storage for large models
                logger.error(f"Model too large ({model_size_mb:.1f}MB > {max_size_mb}MB)")
                logger.error("TODO: Implement backup storage (S3, GCS, etc.) for large models")
                return Err(
                    ModelError(
                        message=f"Model retrieval failed: HF upload failed, "
                        f"model too large for local download ({model_size_mb:.1f}MB > {max_size_mb}MB). "
                        f"TODO: Add backup storage support.",
                        code="MODEL_TOO_LARGE_FOR_LOCAL",
                        details={"model_size_mb": model_size_mb, "max_size_mb": max_size_mb},
                    )
                )

        # Step 3: Cleanup is handled by GPUDeployer.cleanup() / Orchestrator
        # No need to cleanup here - provider handles it based on config
        logger.info(f"Cleanup will be handled by {self._provider_name} provider")

        # Summary
        if hf_uploaded:
            logger.info(f"Model available on HF: https://huggingface.co/{self.hf_repo_id}")
        if local_model_path:
            logger.info(f"Model available locally: {local_model_path}")

        # Fire callback
        if self._callbacks.on_retrieval_completed:
            self._callbacks.on_retrieval_completed(
                hf_uploaded,
                str(local_model_path) if local_model_path else None,
            )

        # Update context
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

    def _upload_to_hf_from_remote(self, context: dict[str, Any] | None = None) -> Result[None, ModelError]:
        """
        Upload model directly from remote server to HuggingFace Hub.

        Provider-agnostic: uses self._ssh_client from execute().
        This is faster than downloading locally first.

        Uploads ONLY:
        - Final model files (adapter_model.safetensors, adapter_config.json)
        - Tokenizer files
        - README.md (model card)

        Does NOT upload:
        - checkpoint-* directories (intermediate checkpoints)
        - runs/ directory (TensorBoard logs - kept locally)
        - run_* directories (internal state)
        """
        try:
            context = context or {}
            repo_ready = self._ensure_hf_repo_ready()
            if repo_ready.is_failure():
                return Err(repo_ready.unwrap_err())  # type: ignore[union-attr]  # already ModelError

            if not self._ssh_client:
                return Err(ModelError(message="SSH client not initialized", code="SSH_CLIENT_NOT_INITIALIZED"))

            ssh_client = self._ssh_client
            # Output path is hardcoded: checkpoints live under {workspace}/output/
            remote_output_dir = f"{self._workspace_path}/output"
            hf_token = self.secrets.hf_token

            # Step 0: Find best checkpoint (checkpoint-final → checkpoint-* → output dir)
            upload_dir = f"{self._workspace_path}/hf_upload"
            logger.info("Finding final model checkpoint...")
            final_checkpoint = self._resolve_checkpoint(remote_output_dir)

            # Debug: show checkpoint contents
            debug_cmd = f"ls -la {final_checkpoint}/ 2>/dev/null || echo 'DIR_NOT_FOUND'"
            success, stdout, _ = ssh_client.exec_command(
                command=debug_cmd, background=False, timeout=MR_SSH_CMD_TIMEOUT
            )
            if success:
                logger.info(f"Checkpoint contents:\n{stdout}")

            # Clean up and create fresh upload dir
            cleanup_cmds = [
                f"rm -rf {upload_dir}",
                f"mkdir -p {upload_dir}",
            ]

            for cmd in cleanup_cmds:
                ssh_client.exec_command(command=cmd, background=False, timeout=60)

            # Copy model files from checkpoint
            copy_cmds = [
                f"cp -r {final_checkpoint}/adapter_* {upload_dir}/ 2>/dev/null || true",
                f"cp -r {final_checkpoint}/tokenizer* {upload_dir}/ 2>/dev/null || true",
                f"cp -r {final_checkpoint}/special_tokens* {upload_dir}/ 2>/dev/null || true",
                f"cp -r {final_checkpoint}/config.json {upload_dir}/ 2>/dev/null || true",
                f"cp -r {final_checkpoint}/*.json {upload_dir}/ 2>/dev/null || true",
                f"cp -r {final_checkpoint}/*.model {upload_dir}/ 2>/dev/null || true",
                f"cp -r {final_checkpoint}/*.safetensors {upload_dir}/ 2>/dev/null || true",
                f"cp -r {final_checkpoint}/model.safetensors {upload_dir}/ 2>/dev/null || true",
            ]

            for cmd in copy_cmds:
                ssh_client.exec_command(command=cmd, background=False, timeout=60)

            # NOTE: training_config.yaml is now uploaded to MLflow artifacts instead of HF
            # This keeps HF repo clean (only model files) and centralizes configs in MLflow

            # Step 1: Generate README.md with model card (using base64 to avoid shell escaping issues)
            phase_result = self._extract_phase_metrics(
                context=context,
                remote_output_dir=remote_output_dir,
            )
            card_ctx = ModelCardContext(
                phase_metrics=phase_result.phase_metrics,
                datasets=self._extract_datasets_for_readme(),
                dataset_source_type=self._extract_dataset_source_type_for_readme(),
                training_started_at=phase_result.training_started_at,
                training_completed_at=phase_result.training_completed_at,
            )
            readme_content = self._generate_model_card(card_ctx)
            readme_b64 = base64.b64encode(memoryview(readme_content.encode())).decode()
            readme_cmd = f"echo '{readme_b64}' | base64 -d > {upload_dir}/README.md"
            success, _, stderr = ssh_client.exec_command(command=readme_cmd, background=False, timeout=60)
            if success:
                logger.info("README.md created with model card")
            else:
                logger.warning(f"README.md creation failed: {stderr}")

            # Step 3: List what we're uploading (for debug)
            list_cmd = f"ls -la {upload_dir}/"
            success, stdout, _ = ssh_client.exec_command(command=list_cmd, background=False, timeout=MR_SSH_CMD_TIMEOUT)
            if success and stdout:
                logger.info(f"Files to upload:\n{stdout}")

            # Step 4: Upload using `huggingface-cli`.
            #
            # IMPORTANT:
            # In some SSH environments PATH can be sanitized (missing /opt/conda/bin),
            # while `huggingface-cli` is installed into the conda bin directory by our training runtime.
            #
            # Policy:
            # - We keep using `huggingface-cli` (requested UX).
            # - We "probe" the binary into a stable PATH location (best-effort symlink to /usr/local/bin).
            commit_msg = "Upload from RyotenkAI pipeline (final model only)"
            upload_cmd = (
                "set -euo pipefail && "
                'export PATH="/usr/local/bin:/opt/conda/bin:$HOME/.local/bin:$PATH" && '
                "if ! command -v huggingface-cli >/dev/null 2>&1; then "
                "  if [ -x /opt/conda/bin/huggingface-cli ]; then "
                "    ln -sf /opt/conda/bin/huggingface-cli /usr/local/bin/huggingface-cli; "
                "  fi; "
                "fi && "
                f'HF_TOKEN="{hf_token}" '
                f"huggingface-cli upload {self.hf_repo_id} {upload_dir} . "
                f'--commit-message "{commit_msg}"'
            )

            logger.info(f"Uploading from {self._provider_name} to {self.hf_repo_id}...")
            success, _stdout, stderr = ssh_client.exec_command(
                command=upload_cmd,
                background=False,
                timeout=MR_UPLOAD_TIMEOUT,
            )

            if not success:
                return Err(ModelError(message=f"Upload command failed: {stderr}", code="HF_UPLOAD_COMMAND_FAILED"))

            # Step 5: Cleanup temp upload dir
            ssh_client.exec_command(command=f"rm -rf {upload_dir}", background=False, timeout=MR_SSH_CMD_TIMEOUT)

            return Ok(None)

        except Exception as e:
            return Err(ModelError(message=f"Direct upload failed: {e!s}", code="HF_UPLOAD_FAILED"))

    def _generate_model_card(self, ctx: ModelCardContext | None = None) -> str:
        """Generate README.md content for HuggingFace model card."""
        cfg = self.config
        if ctx is None:
            ctx = ModelCardContext(phase_metrics=[], datasets=[])

        repo_id = self.hf_repo_id or "unknown"
        model_name = repo_id.split("/")[-1] if repo_id else "model"
        base_model = cfg.model.name or "Unknown"

        def _pick(row: dict[str, Any], *keys: str) -> Any:
            for k in keys:
                if k in row and row[k] is not None:
                    return row[k]
            return None

        def _fmt(v: Any, *, digits: int | None = None) -> str:
            if v is None:
                return "—"
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, int | str):
                return str(v)
            if isinstance(v, float):
                if digits is None:
                    return str(v)
                return f"{v:.{digits}f}"
            return str(v)

        tags: list[str] = []
        for t in ("fine-tuned", "adapter", "peft", "lora", cfg.training.type, "trl", "ryotenkai"):
            if isinstance(t, str) and t and t not in tags:
                tags.append(t)

        yaml_lines: list[str] = [
            "---",
            "license: apache-2.0",
            f"base_model: {base_model}",
            "base_model_relation: adapter",
            "library_name: transformers",
            "pipeline_tag: text-generation",
        ]

        if ctx.datasets:
            yaml_lines.append("datasets:")
            for ds in ctx.datasets:
                if isinstance(ds, str) and ds.strip():
                    yaml_lines.append(f"  - {ds.strip()}")

        yaml_lines.append("tags:")
        for t in tags:
            yaml_lines.append(f"  - {t}")
        yaml_lines.append("---")

        datasets_md = ", ".join(f"`{d}`" for d in ctx.datasets) if ctx.datasets else "—"

        # Training results table (best-effort; never raises)
        results_lines: list[str] = []
        if ctx.training_started_at or ctx.training_completed_at:
            results_lines.append("### Run timeline")
            if ctx.training_started_at:
                results_lines.append(f"- **Started at**: `{ctx.training_started_at}`")
            if ctx.training_completed_at:
                results_lines.append(f"- **Completed at**: `{ctx.training_completed_at}`")
            results_lines.append("")

        if ctx.phase_metrics:
            results_lines.extend(
                [
                    "| Phase | Strategy | Status | train_loss | eval_loss | global_step | epoch | runtime_s | peak_mem_gb |",
                    "|---:|---|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in ctx.phase_metrics:
                phase_idx = _fmt(_pick(row, _PHASE_IDX))
                strategy = _fmt(_pick(row, _STRATEGY_TYPE))
                status = _fmt(_pick(row, _STATUS))
                train_loss = _fmt(_pick(row, "train_loss", "loss"), digits=4)
                eval_loss = _fmt(_pick(row, "eval_loss"), digits=4)
                global_step = _fmt(_pick(row, "global_step"))
                epoch = _fmt(_pick(row, "epoch"), digits=2)
                runtime_s = _fmt(_pick(row, "train_runtime"), digits=1)
                peak_mem = _fmt(_pick(row, "peak_memory_gb"), digits=2)
                results_lines.append(
                    f"| {phase_idx} | {strategy} | {status} | {train_loss} | {eval_loss} | "
                    f"{global_step} | {epoch} | {runtime_s} | {peak_mem} |"
                )
        else:
            results_lines.append("No per-phase metrics were found for this run.")

        trust_remote_code = bool(getattr(cfg.model, "trust_remote_code", False))

        hyper = cfg.training.hyperparams
        per_device_bs = getattr(hyper, "per_device_train_batch_size", None)
        grad_acc = getattr(hyper, "gradient_accumulation_steps", None)
        effective_bs: int | None = None
        if isinstance(per_device_bs, int) and isinstance(grad_acc, int):
            effective_bs = per_device_bs * grad_acc

        md: list[str] = []
        md.extend(yaml_lines)
        md.append("")
        md.append(f"# {model_name}")
        md.append("")
        md.append(f"This repository contains a **PEFT LoRA adapter** for `{base_model}`.")
        md.append("")
        md.append("## Model Details")
        md.append("")
        md.extend(
            [
                "| Parameter | Value |",
                "|---|---|",
                f"| **Base model** | `{base_model}` |",
                f"| **Adapter repo** | `{repo_id}` |",
                f"| **Training type** | `{cfg.training.type}` |",
                f"| **Strategy chain** | {self._format_strategies(cfg)} |",
                f"| **Dataset source** | `{_fmt(ctx.dataset_source_type)}` |",
                f"| **Datasets** | {datasets_md} |",
                f"| **Batch Size** | {hyper.per_device_train_batch_size} |",
                f"| **LoRA r** | `{self._get_lora_param(cfg, 'r')}` |",
                f"| **LoRA alpha** | `{self._get_lora_param(cfg, 'lora_alpha')}` |",
                f"| **LoRA dropout** | `{self._get_lora_param(cfg, 'lora_dropout')}` |",
                f"| **LoRA bias** | `{self._get_lora_param(cfg, 'bias')}` |",
                f"| **Target modules** | `{self._get_lora_param(cfg, 'target_modules')}` |",
                f"| **DoRA** | `{self._get_lora_param(cfg, 'use_dora')}` |",
                f"| **rsLoRA** | `{self._get_lora_param(cfg, 'use_rslora')}` |",
                f"| **Init** | `{self._get_lora_param(cfg, 'init_lora_weights')}` |",
            ]
        )

        md.append("")
        md.append("## Training Details")
        md.append("")
        md.extend(
            [
                "| Hyperparameter | Value |",
                "|---|---|",
                f"| **epochs** | `{hyper.epochs}` |",
                f"| **learning_rate** | `{hyper.learning_rate}` |",
                f"| **warmup_ratio** | `{hyper.warmup_ratio}` |",
                f"| **per_device_train_batch_size** | `{hyper.per_device_train_batch_size}` |",
                f"| **gradient_accumulation_steps** | `{hyper.gradient_accumulation_steps}` |",
                f"| **effective_batch_size** | `{_fmt(effective_bs)}` |",
                f"| **optimizer** | `{cfg.training.get_effective_optimizer()}` |",
                f"| **lr_scheduler** | `{hyper.lr_scheduler_type}` |",
            ]
        )

        md.append("")
        md.append("## Training Results")
        md.append("")
        md.extend(results_lines)

        md.append("")
        md.append("## Usage")
        md.append("")
        md.append("### Load as a PEFT adapter (recommended)")
        md.append("")
        md.append("```python")
        md.append("from transformers import AutoModelForCausalLM, AutoTokenizer")
        md.append("from peft import PeftModel")
        md.append("")
        md.append(f'base_model_id = "{base_model}"')
        md.append(f'adapter_id = "{repo_id}"')
        md.append("")
        md.append(f"tokenizer = AutoTokenizer.from_pretrained(adapter_id, trust_remote_code={trust_remote_code})")
        md.append(
            "model = AutoModelForCausalLM.from_pretrained("
            'base_model_id, device_map="auto", torch_dtype="auto", trust_remote_code='
            f"{trust_remote_code})"
        )
        md.append("model = PeftModel.from_pretrained(model, adapter_id)")
        md.append("model.eval()")
        md.append("")
        md.append('prompt = "Hello!"')
        md.append('inputs = tokenizer(prompt, return_tensors="pt").to(model.device)')
        md.append("outputs = model.generate(**inputs, max_new_tokens=256)")
        md.append("print(tokenizer.decode(outputs[0], skip_special_tokens=True))")
        md.append("```")

        md.append("")
        md.append("### Merge adapter into base model (optional)")
        md.append("")
        md.append("```python")
        md.append("merged = model.merge_and_unload()")
        md.append('merged.save_pretrained("merged-model")')
        md.append('tokenizer.save_pretrained("merged-model")')
        md.append("```")

        md.append("")
        md.append("## Training Infrastructure")
        md.append("")
        md.append(f"- **Platform**: {self._provider_name}")
        md.append(f"- **GPU**: {self._provider_training_cfg.get('gpu_type', 'auto-detect')}")

        return "\n".join(md) + "\n"

    def _get_model_size(self) -> Result[float, ModelError]:
        """
        Get model size on remote server in MB.

        Provider-agnostic: uses self._ssh_client from execute().
        """
        try:
            if not self._ssh_client:
                return Err(ModelError(message="SSH client not initialized", code="SSH_CLIENT_NOT_INITIALIZED"))

            # Output path is hardcoded: checkpoints live under {workspace}/output/
            remote_output_dir = f"{self._workspace_path}/output"

            # Get size in bytes
            size_cmd = f"du -sb {remote_output_dir} 2>/dev/null | cut -f1"
            success, stdout, stderr = self._ssh_client.exec_command(
                command=size_cmd, background=False, timeout=MR_SSH_CMD_TIMEOUT
            )

            if not success:
                return Err(ModelError(message=f"Size command failed: {stderr}", code="MODEL_SIZE_CHECK_FAILED"))

            size_bytes = int(stdout.strip())
            size_mb = size_bytes / (1024 * 1024)

            return Ok(size_mb)

        except Exception as e:
            return Err(ModelError(message=f"Failed to get model size: {e!s}", code="MODEL_SIZE_CHECK_FAILED"))

    def _download_model(self) -> Result[Path, ModelError]:
        """
        Download trained model from remote server using SSHClient.

        Provider-agnostic: uses self._ssh_client from execute().
        """
        logger.info("Downloading model files...")

        try:
            if not self._ssh_client:
                return Err(ModelError(message="SSH client not initialized", code="SSH_CLIENT_NOT_INITIALIZED"))

            # Create local directory for model
            local_model_dir = Path("models") / f"model_{int(time.time())}"
            local_model_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Local model directory: {local_model_dir}")

            # Download model using tar+ssh
            logger.info(f"Downloading model files from {self._provider_name}...")

            remote_output_dir = f"{self._workspace_path}/output"
            logger.info(f"Remote output directory: {remote_output_dir}")

            try:
                remote_download_dir = self._resolve_checkpoint(remote_output_dir)
            except Exception as e:
                logger.warning(f"Failed to resolve final checkpoint for download, using output dir: {e!s}")
                remote_download_dir = remote_output_dir

            logger.info(f"Remote download directory: {remote_download_dir}")

            download_result = self._ssh_client.download_directory(
                remote_path=remote_download_dir,
                local_path=local_model_dir,
            )

            if download_result.is_failure():
                dl_err = download_result.unwrap_err()  # type: ignore[union-attr]  # str from ssh_client
                return Err(ModelError(message=f"Failed to download model: {dl_err}", code="MODEL_DOWNLOAD_FAILED"))

            logger.info("Model downloaded to local PC")
            return Ok(local_model_dir)

        except Exception as e:
            logger.error(f"Download error: {e}")
            return Err(ModelError(message=f"Failed to download model: {e!s}", code="MODEL_DOWNLOAD_FAILED"))

    def _execute_mock(self, context: dict[str, Any], resource_id: str) -> Result[dict[str, Any], AppError]:
        """
        Mock execution for testing without real model retrieval.
        Simulates downloading model and uploading to HF Hub.
        """
        logger.info(f"[MOCK] Downloading model from {self._provider_name}: {resource_id}")
        time.sleep(1)

        # Create mock model directory
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
