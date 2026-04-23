"""
HFModelUploader — HuggingFace Hub upload and SSH infrastructure for ModelRetriever.

Handles:
- HF repo lifecycle (create, visibility enforcement)
- Checkpoint discovery on remote
- Model upload via huggingface-cli over SSH (with retry)
- Model size check and local download fallback
- Phase metrics extraction from remote pipeline_state.json
- Dataset info extraction for model card
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import HfApi

from src.constants import LORA_CHECKPOINT_PATTERNS
from src.config.datasets.constants import SOURCE_TYPE_HUGGINGFACE, SOURCE_TYPE_LOCAL
from src.pipeline.constants import (
    HTTP_STATUS_NOT_FOUND,
    HTTP_STATUS_UNAUTHORIZED,
    MR_SHA12_LENGTH,
    MR_SSH_CMD_TIMEOUT,
    MR_UPLOAD_TIMEOUT,
)
from src.utils.logger import logger
from src.utils.result import Err, ModelError, Ok, Result

from src.pipeline.stages.model_retriever.types import (
    PhaseMetricsResult,
    _METRICS,
    _PHASE_IDX,
    _PIPELINE_STATE_COMPLETED_AT,
    _PIPELINE_STATE_STARTED_AT,
    _STATUS,
    _STRATEGY_TYPE,
)

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig, Secrets
    from src.utils.ssh_client import SSHClient


class HFModelUploader:
    """
    Infrastructure layer for HF Hub upload and remote SSH operations.

    Responsibilities:
    - Ensure HF repo is ready (create if missing, enforce visibility)
    - Upload model from remote server via huggingface-cli + SSH (with retry)
    - Discover best checkpoint on remote
    - Extract phase metrics from remote pipeline_state.json
    - Extract dataset identifiers for model card metadata
    - Check and download model size
    """

    def __init__(
        self,
        config: PipelineConfig,
        secrets: Secrets,
        *,
        hf_api: HfApi | None = None,
    ) -> None:
        self.config = config
        self.secrets = secrets

        hf_cfg = config.experiment_tracking.huggingface
        self.hf_repo_id: str | None = hf_cfg.repo_id if hf_cfg else None
        self.hf_private: bool | None = hf_cfg.private if hf_cfg else None

        # PR4: prefer per-integration token; fall back to HF_TOKEN env.
        integration_id = hf_cfg.integration if hf_cfg else None
        token = secrets.get_hf_token(integration_id)
        self.hf_api: HfApi = hf_api or HfApi(token=token)

        # Set by caller after SSH connection is established
        self._ssh_client: SSHClient | None = None
        self._workspace_path: str = "/workspace"

    def set_ssh_context(self, ssh_client: SSHClient, workspace_path: str) -> None:
        """Set the SSH client and workspace path after connection."""
        self._ssh_client = ssh_client
        self._workspace_path = workspace_path

    # ------------------------------------------------------------------
    # HF repo lifecycle
    # ------------------------------------------------------------------

    def ensure_hf_repo_ready(self) -> Result[None, ModelError]:
        """
        Ensure HF repo exists and matches requested visibility.

        IMPORTANT:
        - `create_repo(..., exist_ok=True)` does NOT change visibility of an existing repo.
        - To enforce `private: true/false`, we must call `update_repo_settings(private=...)`.
        """
        hf_cfg = self.config.experiment_tracking.huggingface
        if not hf_cfg or not hf_cfg.integration:
            return Err(ModelError(message="HuggingFace upload disabled", code="HF_UPLOAD_DISABLED"))
        if not self.hf_repo_id:
            return Err(ModelError(message="HF repo_id not configured", code="HF_REPO_ID_MISSING"))

        try:
            repo_exists = True
            try:
                _ = self.hf_api.repo_info(repo_id=self.hf_repo_id, repo_type="model")
            except Exception as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == HTTP_STATUS_NOT_FOUND:
                    repo_exists = False
                else:
                    raise

            if not repo_exists:
                self.hf_api.create_repo(
                    repo_id=self.hf_repo_id,
                    private=hf_cfg.private,
                    exist_ok=True,
                    repo_type="model",
                )

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
                    f"\u26a0\ufe0f Failed to update HF repo settings for '{self.hf_repo_id}': "
                    f"{e!s}{hint} (continuing upload)"
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
                    message=f"Failed to prepare HF repo '{self.hf_repo_id}': {e!s}{hint}",
                    code="HF_REPO_PREPARE_FAILED",
                )
            )

    # ------------------------------------------------------------------
    # Checkpoint discovery
    # ------------------------------------------------------------------

    def resolve_checkpoint(self, remote_output_dir: str) -> str:
        """
        Resolve the best checkpoint directory under remote_output_dir.

        Priority:
        1. checkpoint-final
        2. Latest checkpoint-* by version sort
        3. remote_output_dir itself
        """
        if not self._ssh_client:
            return remote_output_dir

        ok_final, out_final, _ = self._ssh_client.exec_command(
            command=f"find {remote_output_dir} -type d -name 'checkpoint-final' 2>/dev/null | head -1",
            background=False,
            timeout=MR_SSH_CMD_TIMEOUT,
        )
        if ok_final and out_final.strip():
            logger.info(f"Found checkpoint: {out_final.strip()}")
            return out_final.strip()

        ok_latest, out_latest, _ = self._ssh_client.exec_command(
            command=f"find {remote_output_dir} -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1",
            background=False,
            timeout=MR_SSH_CMD_TIMEOUT,
        )

        if ok_latest and out_latest.strip():
            logger.warning(f"checkpoint-final not found, using: {out_latest.strip()}")
            return out_latest.strip()

        if not ok_final or not ok_latest:
            logger.warning(
                f"Checkpoint discovery unavailable (SSH exec error), using: {remote_output_dir}"
            )
        else:
            logger.warning(f"No checkpoints found, using: {remote_output_dir}")
        return remote_output_dir

    # ------------------------------------------------------------------
    # Phase metrics extraction
    # ------------------------------------------------------------------

    def extract_phase_metrics(
        self,
        *,
        context: dict[str, Any],
        remote_output_dir: str,
    ) -> PhaseMetricsResult:
        """
        Extract training phase metrics for README generation.

        Primary source: remote `pipeline_state.json` (DataBuffer state).
        Fallbacks: context-supplied lists.
        """
        for key in ("phase_metrics", "training_phase_metrics", "phase_runs_metrics"):
            v = context.get(key)
            if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                return PhaseMetricsResult(
                    phase_metrics=[dict(x) for x in v],
                    training_started_at=None,
                    training_completed_at=None,
                )

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

        return PhaseMetricsResult(
            phase_metrics=[], training_started_at=None, training_completed_at=None
        )

    # ------------------------------------------------------------------
    # Dataset info extraction
    # ------------------------------------------------------------------

    def extract_dataset_source_type_for_readme(self) -> str | None:
        """Extract dataset source type for README rendering."""
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

    def extract_datasets_for_readme(self, *, basename_fn: Any = None) -> list[str]:
        """Extract dataset identifiers suitable for HF model card metadata."""
        from src.pipeline.stages.model_retriever.model_card import ModelCardGenerator

        _basename = basename_fn or ModelCardGenerator._basename

        try:
            default_ds = self.config.get_primary_dataset()
        except (AttributeError, KeyError, ValueError):
            return []

        try:
            source_type = default_ds.get_source_type()
            out: list[str] = []

            if source_type == SOURCE_TYPE_HUGGINGFACE and default_ds.source_hf is not None:
                train_id = getattr(default_ds.source_hf, "train_id", None)
                eval_id = getattr(default_ds.source_hf, "eval_id", None)
                if isinstance(train_id, str) and train_id.strip():
                    out.append(train_id.strip())
                if isinstance(eval_id, str) and eval_id.strip() and eval_id.strip() not in out:
                    out.append(eval_id.strip())

            elif source_type == SOURCE_TYPE_LOCAL and default_ds.source_local is not None:
                local_paths = getattr(default_ds.source_local, "local_paths", None)
                train_path = getattr(local_paths, "train", None) if local_paths is not None else None
                eval_path = getattr(local_paths, "eval", None) if local_paths is not None else None

                if isinstance(train_path, str) and train_path.strip():
                    name = _basename(train_path)
                    if name:
                        out.append(name)
                if isinstance(eval_path, str) and eval_path.strip():
                    name = _basename(eval_path)
                    if name and name not in out:
                        out.append(name)

            return out
        except (AttributeError, TypeError, KeyError):
            return []

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_to_hf_from_remote(
        self,
        context: dict[str, Any] | None = None,
        *,
        card_content: str,
    ) -> Result[None, ModelError]:
        """
        Upload model directly from remote server to HuggingFace Hub.

        Uses SSH + huggingface-cli. Includes 3-attempt retry for transient failures.
        """
        try:
            context = context or {}
            repo_ready = self.ensure_hf_repo_ready()
            if repo_ready.is_failure():
                return Err(repo_ready.unwrap_err())  # type: ignore[union-attr]

            return self._upload_files(context=context, card_content=card_content)
        except Exception as e:
            return Err(ModelError(message=f"Direct upload failed: {e!s}", code="HF_UPLOAD_FAILED"))

    def _upload_files(
        self,
        context: dict[str, Any] | None = None,
        *,
        card_content: str = "",
    ) -> Result[None, ModelError]:
        """Internal upload (skips repo-ready check, called after ensure_hf_repo_ready)."""
        try:
            context = context or {}

            if not self._ssh_client:
                return Err(
                    ModelError(
                        message="SSH client not initialized",
                        code="SSH_CLIENT_NOT_INITIALIZED",
                    )
                )

            ssh_client = self._ssh_client
            remote_output_dir = f"{self._workspace_path}/output"
            hf_token = self.secrets.hf_token

            upload_dir = f"{self._workspace_path}/hf_upload"
            logger.info("Finding final model checkpoint...")
            final_checkpoint = self.resolve_checkpoint(remote_output_dir)

            debug_cmd = f"ls -la {final_checkpoint}/ 2>/dev/null || echo 'DIR_NOT_FOUND'"
            success, stdout, _ = ssh_client.exec_command(
                command=debug_cmd, background=False, timeout=MR_SSH_CMD_TIMEOUT
            )
            if success:
                logger.info(f"Checkpoint contents:\n{stdout}")

            for cmd in [f"rm -rf {upload_dir}", f"mkdir -p {upload_dir}"]:
                ssh_client.exec_command(command=cmd, background=False, timeout=60)

            copy_cmds = [
                f"cp -r {final_checkpoint}/{pattern} {upload_dir}/ 2>/dev/null || true"
                for pattern in LORA_CHECKPOINT_PATTERNS
            ]
            # full-model specific patterns (dense weights, not LoRA-specific):
            copy_cmds += [
                f"cp -r {final_checkpoint}/*.safetensors {upload_dir}/ 2>/dev/null || true",
                f"cp -r {final_checkpoint}/model.safetensors {upload_dir}/ 2>/dev/null || true",
            ]
            for cmd in copy_cmds:
                ssh_client.exec_command(command=cmd, background=False, timeout=60)

            # Write README.md via base64 to avoid shell escaping issues
            readme_b64 = base64.b64encode(memoryview(card_content.encode())).decode()
            readme_cmd = f"echo '{readme_b64}' | base64 -d > {upload_dir}/README.md"
            success, _, stderr = ssh_client.exec_command(
                command=readme_cmd, background=False, timeout=60
            )
            if success:
                logger.info("README.md created with model card")
            else:
                logger.warning(f"README.md creation failed: {stderr}")

            list_cmd = f"ls -la {upload_dir}/"
            success, stdout, _ = ssh_client.exec_command(
                command=list_cmd, background=False, timeout=MR_SSH_CMD_TIMEOUT
            )
            if success and stdout:
                logger.info(f"Files to upload:\n{stdout}")

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

            logger.info(f"Uploading to {self.hf_repo_id}...")

            _upload_last_err: str = ""
            _upload_success = False
            for _attempt in range(1, 4):
                success, _stdout, stderr = ssh_client.exec_command(
                    command=upload_cmd,
                    background=False,
                    timeout=MR_UPLOAD_TIMEOUT,
                )
                if success:
                    _upload_success = True
                    break
                _upload_last_err = stderr
                if _attempt < 3:
                    logger.warning(
                        f"[MR:UPLOAD_RETRY {_attempt}/3] Upload attempt failed: {stderr}. "
                        "Retrying in 10s..."
                    )
                    time.sleep(10)

            if not _upload_success:
                return Err(
                    ModelError(
                        message=f"Upload command failed after 3 attempts: {_upload_last_err}",
                        code="HF_UPLOAD_COMMAND_FAILED",
                    )
                )

            ssh_client.exec_command(
                command=f"rm -rf {upload_dir}", background=False, timeout=MR_SSH_CMD_TIMEOUT
            )

            return Ok(None)

        except Exception as e:
            return Err(ModelError(message=f"Direct upload failed: {e!s}", code="HF_UPLOAD_FAILED"))

    # ------------------------------------------------------------------
    # Model size & download
    # ------------------------------------------------------------------

    def get_model_size(self) -> Result[float, ModelError]:
        """Get model size on remote server in MB."""
        try:
            if not self._ssh_client:
                return Err(
                    ModelError(
                        message="SSH client not initialized",
                        code="SSH_CLIENT_NOT_INITIALIZED",
                    )
                )

            remote_output_dir = f"{self._workspace_path}/output"
            size_cmd = f"du -sb {remote_output_dir} 2>/dev/null | cut -f1"
            success, stdout, stderr = self._ssh_client.exec_command(
                command=size_cmd, background=False, timeout=MR_SSH_CMD_TIMEOUT
            )

            if not success:
                return Err(
                    ModelError(
                        message=f"Size command failed: {stderr}",
                        code="MODEL_SIZE_CHECK_FAILED",
                    )
                )

            size_bytes = int(stdout.strip())
            size_mb = size_bytes / (1024 * 1024)
            return Ok(size_mb)

        except Exception as e:
            return Err(
                ModelError(
                    message=f"Failed to get model size: {e!s}",
                    code="MODEL_SIZE_CHECK_FAILED",
                )
            )

    def download_model(self) -> Result[Path, ModelError]:
        """Download trained model from remote server using SSHClient."""
        logger.info("Downloading model files...")

        try:
            if not self._ssh_client:
                return Err(
                    ModelError(
                        message="SSH client not initialized",
                        code="SSH_CLIENT_NOT_INITIALIZED",
                    )
                )

            local_model_dir = Path("models") / f"model_{int(time.time())}"
            local_model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Local model directory: {local_model_dir}")

            remote_output_dir = f"{self._workspace_path}/output"
            logger.info(f"Remote output directory: {remote_output_dir}")

            try:
                remote_download_dir = self.resolve_checkpoint(remote_output_dir)
            except Exception as e:
                logger.warning(f"Failed to resolve final checkpoint for download: {e!s}")
                remote_download_dir = remote_output_dir

            logger.info(f"Remote download directory: {remote_download_dir}")

            download_result = self._ssh_client.download_directory(
                remote_path=remote_download_dir,
                local_path=local_model_dir,
            )

            if download_result.is_failure():
                dl_err = download_result.unwrap_err()  # type: ignore[union-attr]
                return Err(
                    ModelError(
                        message=f"Failed to download model: {dl_err}",
                        code="MODEL_DOWNLOAD_FAILED",
                    )
                )

            logger.info("Model downloaded to local PC")
            return Ok(local_model_dir)

        except Exception as e:
            logger.error(f"Download error: {e}")
            return Err(
                ModelError(
                    message=f"Failed to download model: {e!s}",
                    code="MODEL_DOWNLOAD_FAILED",
                )
            )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @staticmethod
    def _sha12(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:MR_SHA12_LENGTH]


__all__ = ["HFModelUploader"]
