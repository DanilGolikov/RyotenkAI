"""Project-scoped HuggingFace push target.

The block carries only project-local push parameters: ``repo_id`` and
``private``. The auth token is resolved at runtime from the ``HF_TOKEN``
env var (loaded from ``secrets.env`` by the Pydantic ``Secrets`` model).

The block is "active" iff ``repo_id`` is set. To opt out of HF Hub
upload, omit the whole ``integrations.huggingface`` block.

History: the previous schema carried an ``integration: <id>`` reference
pointing at a Settings → Integrations entry that held the token. That
indirection was a frontend-only convenience; the backend now reads the
token directly from env, eliminating the bridge layer.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from ..base import StrictBaseModel


class HuggingFaceHubConfig(StrictBaseModel):
    """Project-scoped HF push target."""

    repo_id: str | None = Field(
        None,
        description="Full repo ID: username/model-name. Block is active iff this is set.",
    )
    private: bool = Field(True, description="Make repo private.")

    @field_validator("repo_id", mode="before")
    @classmethod
    def _normalize(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @property
    def enabled(self) -> bool:
        """``True`` iff the block has a push target configured."""
        return bool(self.repo_id)


# Back-compat alias (used in various places across the codebase).
HuggingFaceConfig = HuggingFaceHubConfig


__all__ = [
    "HuggingFaceConfig",
    "HuggingFaceHubConfig",
]
