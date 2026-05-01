"""Project-scoped HuggingFace reference.

Starting with PR3, ``integrations.huggingface`` in the project YAML
is a *reference* to a reusable HF integration managed in Settings →
Integrations. The integration carries the token; the project keeps the
project-local push target (``repo_id``, ``private``) alongside the
``integration`` id.

Field ``enabled`` is **removed**. The block is "active" iff
``integration`` is set; runtime consumers should check ``cfg.huggingface
is not None and cfg.huggingface.integration`` instead of a separate flag.

Legacy YAML — ``enabled``/``repo_id``/``private`` at the previous layout
— is rejected with a user-friendly error from
``IntegrationsConfig._reject_legacy_keys``.
"""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from ..base import StrictBaseModel


class HuggingFaceHubConfig(StrictBaseModel):
    """Project-scoped HF push target + reference to a Settings integration.

    Kept under the legacy class name for minimum-churn compatibility
    with existing call-sites (``cfg.huggingface.repo_id`` / ``.private``
    keep working). The class is now conceptually an "HF ref", not a full
    integration block.
    """

    integration: str | None = Field(
        None,
        description=(
            "Id of the HuggingFace integration in Settings → Integrations. "
            "When empty, HF push is disabled for this project."
        ),
    )
    repo_id: str | None = Field(
        None,
        description="Full repo ID: username/model-name. Required when integration is set.",
    )
    private: bool = Field(True, description="Make repo private.")

    @field_validator("integration", "repo_id", mode="before")
    @classmethod
    def _normalize(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @model_validator(mode="after")
    def _require_both_or_neither(self) -> HuggingFaceHubConfig:
        # The two fields are bound together: an integration without a
        # repo_id has no push target; a repo_id without an integration
        # has no auth token. Either both are set (HF active) or both
        # are absent (HF disabled — drop the whole block to opt out).
        # Half-configured blocks were the #1 source of silent
        # "HF Hub upload disabled" surprises in production runs.
        if self.integration and not self.repo_id:
            raise ValueError(
                "integrations.huggingface.repo_id is required when "
                "integrations.huggingface.integration is set."
            )
        if self.repo_id and not self.integration:
            raise ValueError(
                "integrations.huggingface.integration is required when "
                "integrations.huggingface.repo_id is set. Add "
                "``integration: <id>`` referencing a Settings → "
                "Integrations entry, or remove the whole "
                "integrations.huggingface block to disable HF Hub upload."
            )
        return self

    @property
    def enabled(self) -> bool:
        """Back-compat accessor.

        Runtime code that still reads ``hf.enabled`` keeps working — the
        flag is now derived from ``integration`` truthiness. New code
        should check ``hf.integration`` directly.
        """
        return bool(self.integration)


# Back-compat alias (used in various places across the codebase).
HuggingFaceConfig = HuggingFaceHubConfig


__all__ = [
    "HuggingFaceConfig",
    "HuggingFaceHubConfig",
]
