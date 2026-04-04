"""
AdapterCacheManager — Hugging Face Hub adapter caching for phase-based training.

Handles:
- Dataset fingerprint computation (local file metadata, HF dataset commit sha)
- Cache hit detection via HF repo tags
- Post-training adapter upload with retry logic
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.utils.logger import logger
from src.utils.result import Err, Ok, Result, TrainingError

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from src.training.managers.data_buffer import DataBuffer
    from src.utils.config import PipelineConfig, StrategyPhaseConfig


_HF_UPLOAD_RETRIES = 3
_HF_UPLOAD_RETRY_DELAY_S = 10


def _retry_call(fn: Any, retries: int = 3, delay_s: int = 10, label: str = "") -> Any:
    """
    Call fn() with retry logic. Raises the last exception after exhausting retries.
    """
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < retries:
                logger.warning(
                    f"[PE:RETRY {attempt}/{retries}] {label}: {e}. Retrying in {delay_s}s..."
                )
                time.sleep(delay_s)
    raise last_err  # type: ignore[misc]


class AdapterCacheManager:
    """
    Manages adapter caching on Hugging Face Hub for phase-based training.

    Responsibilities:
    - Compute dataset fingerprints for drift detection
    - Check for cached adapters by tag
    - Upload trained adapters after successful training (soft-fail)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Fingerprinting
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_dataset_fingerprint(dataset_config: Any) -> str:
        """
        Compute a short fingerprint for a dataset config.

        Local file:  sha256(resolved_path + mtime + size)[:10]
        HF dataset:  sha256(train_id + commit_sha)[:10]  (one network call)
        """
        source_type = dataset_config.get_source_type()
        if source_type == "local":
            train_path = Path(dataset_config.source_local.local_paths.train).resolve()
            if train_path.exists():
                stat = train_path.stat()
                raw = f"{train_path}\x00{stat.st_mtime}\x00{stat.st_size}"
            else:
                raw = str(train_path)
        else:
            train_id = dataset_config.source_hf.train_id
            try:
                from huggingface_hub import dataset_info as hf_dataset_info

                info = hf_dataset_info(train_id)
                commit_sha = info.sha or ""
            except Exception as e:
                logger.debug(
                    f"[PE:FINGERPRINT_HF_WARN] could not fetch dataset_info for {train_id}: {e}"
                )
                commit_sha = ""
            raw = f"{train_id}\x00{commit_sha}"

        return hashlib.sha256(raw.encode()).hexdigest()[:10]

    def compute_fingerprint_safe(self, phase_idx: int, phase: StrategyPhaseConfig) -> str | None:
        """
        Compute dataset fingerprint, returning None on any error (treat as cache miss).
        """
        try:
            dataset_config = self.config.get_dataset_for_strategy(phase)
            return self._compute_dataset_fingerprint(dataset_config)
        except Exception as e:
            logger.warning(
                f"[PE:FINGERPRINT_FAILED] phase={phase_idx} ({phase.strategy_type}): {e}. "
                "Treating as cache miss — will train normally."
            )
            return None

    # ------------------------------------------------------------------
    # Cache hit detection
    # ------------------------------------------------------------------

    def try_hit(
        self,
        phase_idx: int,
        phase: StrategyPhaseConfig,
        model: PreTrainedModel,
        buffer: DataBuffer,
        fingerprint: str,
    ) -> Result[PreTrainedModel, TrainingError] | None:
        """
        Try to load a cached adapter from HF Hub.

        Returns:
            Ok(model)  if cache hit  — training is skipped.
            None       if cache miss — caller should proceed to training.
        """
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError

        cache = phase.adapter_cache
        repo_id = cache.repo_id
        expected_tag = f"phase-{phase_idx}-{phase.strategy_type}-ds{fingerprint}"

        logger.debug(
            f"[PE:ADAPTER_CACHE_CHECK] phase={phase_idx}, repo={repo_id}, tag={expected_tag}"
        )

        try:
            api = HfApi()
            refs = api.list_repo_refs(repo_id, repo_type="model")
            tag_exists = any(t.name == expected_tag for t in refs.tags)
        except RepositoryNotFoundError:
            logger.info(
                f"   Cache miss (repo not found): phase {phase_idx} ({phase.strategy_type}) — will train"
            )
            return None
        except Exception as e:
            logger.warning(
                f"[PE:ADAPTER_CACHE_CHECK_WARN] phase={phase_idx}: could not list refs for "
                f"{repo_id}: {e}. Treating as cache miss."
            )
            return None

        if not tag_exists:
            logger.info(
                f"   Cache miss: phase {phase_idx} ({phase.strategy_type}), "
                f"tag '{expected_tag}' not found in '{repo_id}' — will train"
            )
            return None

        logger.info(
            f"   \U0001f3af Adapter cache hit: phase {phase_idx} ({phase.strategy_type}), "
            f"loading from {repo_id}@{expected_tag}"
        )
        try:
            from peft import PeftModel

            loaded_model = PeftModel.from_pretrained(model, repo_id, revision=expected_tag)
        except Exception as e:
            logger.warning(
                f"[PE:ADAPTER_CACHE_LOAD_WARN] phase={phase_idx}: failed to load adapter "
                f"from {repo_id}@{expected_tag}: {e}. Treating as cache miss — will train."
            )
            return None

        buffer.state.phases[phase_idx].adapter_cache_hit = True
        buffer.state.phases[phase_idx].adapter_cache_tag = expected_tag
        buffer.mark_phase_skipped(
            phase_idx,
            reason=f"adapter_cache_hit: {repo_id}@{expected_tag}",
        )
        logger.debug(f"[PE:ADAPTER_CACHE_HIT_DONE] phase={phase_idx}, tag={expected_tag}")
        return Ok(loaded_model)

    # ------------------------------------------------------------------
    # Cache upload
    # ------------------------------------------------------------------

    def upload(
        self,
        phase_idx: int,
        phase: StrategyPhaseConfig,
        checkpoint_path: Path,
        buffer: DataBuffer,
        fingerprint: str,
    ) -> None:
        """
        Upload trained adapter to HF Hub cache after successful training. Soft-fail.

        Creates the repo if it doesn't exist.
        Tags the uploaded commit with the dataset fingerprint tag.
        On failure: logs error in PhaseState, does NOT stop the pipeline.
        """
        from huggingface_hub import HfApi

        cache = phase.adapter_cache
        repo_id = cache.repo_id
        expected_tag = f"phase-{phase_idx}-{phase.strategy_type}-ds{fingerprint}"
        api = HfApi()

        logger.info(f"   \u2b06\ufe0f  Uploading adapter to cache: {repo_id}@{expected_tag} ...")

        def do_upload() -> None:
            api.create_repo(repo_id, private=cache.private, exist_ok=True, repo_type="model")
            api.upload_folder(
                folder_path=str(checkpoint_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message=(
                    f"RyotenkAI: phase-{phase_idx} {phase.strategy_type}, tag={expected_tag}"
                ),
            )
            api.create_tag(
                repo_id=repo_id,
                tag=expected_tag,
                exist_ok=True,
                repo_type="model",
            )

        try:
            _retry_call(
                do_upload,
                retries=_HF_UPLOAD_RETRIES,
                delay_s=_HF_UPLOAD_RETRY_DELAY_S,
                label=f"upload adapter phase {phase_idx} to {repo_id}",
            )
            buffer.state.phases[phase_idx].adapter_cache_tag = expected_tag
            buffer.save_state()
            logger.info(f"   \u2705 Adapter cached: {repo_id}@{expected_tag}")
        except Exception as e:
            err_msg = str(e)
            buffer.state.phases[phase_idx].adapter_cache_upload_error = err_msg
            buffer.save_state()
            logger.error(
                f"[PE:ADAPTER_CACHE_UPLOAD_FAILED] phase={phase_idx}, repo={repo_id}, "
                f"tag={expected_tag}: {err_msg}"
            )
            logger.warning(
                f"   \u26a0\ufe0f  Adapter upload failed after {_HF_UPLOAD_RETRIES} attempts (soft-fail). "
                f"Next run will retrain phase {phase_idx} due to cache miss."
            )


__all__ = ["AdapterCacheManager", "_retry_call", "_HF_UPLOAD_RETRIES", "_HF_UPLOAD_RETRY_DELAY_S"]
