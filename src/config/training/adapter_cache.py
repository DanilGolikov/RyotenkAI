from __future__ import annotations

from pydantic import Field

from ..base import StrictBaseModel


class AdapterCacheConfig(StrictBaseModel):
    """
    Adapter caching configuration for HuggingFace Hub.

    When enabled, the system acts as a cache layer for trained adapters:
    - Before training: checks HF Hub for a tag matching the current dataset fingerprint.
      Cache hit  → adapter loaded, training skipped.
      Cache miss → training runs normally.
    - After training: adapter is uploaded to HF Hub and tagged with the dataset fingerprint.

    Tag format: phase-{idx}-{strategy}-ds{fingerprint[:10]}
    Example:    phase-0-sft-dsA1B2C3D4E5

    Dataset fingerprint:
      - Local file:   sha256(resolved_path + mtime + size)[:10]
      - HF dataset:   sha256(train_id + commit_sha)[:10]  (one network call)

    Cascade invalidation: if a phase is retrained (cache miss), all subsequent
    phases are also forced to retrain regardless of their own fingerprint.

    Upload failure: soft-fail — error is logged in PhaseState, pipeline continues.
    Next run will encounter a cache miss for that phase and retrain.

    Limitations (documented):
      - Local files copied with --preserve-timestamps are not detected as changed.
      - HF dataset fingerprint requires one network call to dataset_info() per run.
    """

    enabled: bool = Field(default=False, description="Enable adapter caching on HF Hub")
    repo_id: str | None = Field(
        default=None,
        description=(
            "HF Hub repository for intermediate adapters. "
            "Required when enabled=true. "
            "Must differ from experiment_tracking.huggingface.repo_id (final model)."
        ),
    )
    private: bool = Field(default=True, description="Create private HF repository")


__all__ = ["AdapterCacheConfig"]
