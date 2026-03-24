from __future__ import annotations

from pydantic import Field, model_validator

from ..base import StrictBaseModel
from .constants import (
    CHAT_UI_PORT_DEFAULT,
    CHAT_UI_PORT_MAX,
    CHAT_UI_PORT_MIN,
    HEALTH_CHECK_INTERVAL_DEFAULT,
    HEALTH_CHECK_INTERVAL_MAX,
    HEALTH_CHECK_INTERVAL_MIN,
    HEALTH_CHECK_RETRIES_DEFAULT,
    HEALTH_CHECK_RETRIES_MAX,
    HEALTH_CHECK_RETRIES_MIN,
    HEALTH_CHECK_TIMEOUT_DEFAULT,
    HEALTH_CHECK_TIMEOUT_MAX,
    HEALTH_CHECK_TIMEOUT_MIN,
)


class InferenceHealthCheckConfig(StrictBaseModel):
    """Health check policy for inference endpoint readiness."""

    enabled: bool = True
    timeout_seconds: int = Field(
        HEALTH_CHECK_TIMEOUT_DEFAULT,
        ge=HEALTH_CHECK_TIMEOUT_MIN,
        le=HEALTH_CHECK_TIMEOUT_MAX,
    )
    interval_seconds: int = Field(
        HEALTH_CHECK_INTERVAL_DEFAULT,
        ge=HEALTH_CHECK_INTERVAL_MIN,
        le=HEALTH_CHECK_INTERVAL_MAX,
    )
    retries: int = Field(
        HEALTH_CHECK_RETRIES_DEFAULT,
        ge=HEALTH_CHECK_RETRIES_MIN,
        le=HEALTH_CHECK_RETRIES_MAX,
    )


class InferenceLoRAConfig(StrictBaseModel):
    """LoRA handling policy for inference deployment."""

    merge_before_deploy: bool = Field(True, description="Default: merge adapter into base model before serving.")
    adapter_path: str = Field("auto", description="'auto' or explicit adapter path/ref.")


class InferenceChatUIConfig(StrictBaseModel):
    """Chat UI config (Phase 3+, MVP disabled)."""

    enabled: bool = False
    type: str = Field("gradio", description="gradio | streamlit | none")
    port: int = Field(CHAT_UI_PORT_DEFAULT, ge=CHAT_UI_PORT_MIN, le=CHAT_UI_PORT_MAX)
    share: bool = False


class InferenceLLMConfig(StrictBaseModel):
    """LLM execution settings (system prompts, generation params, etc.)."""

    system_prompt_path: str | None = Field(
        default=None,
        description=(
            "Path to a text file containing the system prompt for the LLM. "
            "Mutually exclusive with system_prompt_mlflow_name."
        ),
    )
    system_prompt_mlflow_name: str | None = Field(
        default=None,
        description=(
            "Prompt name or URI in MLflow Prompt Registry. "
            "Supported formats: "
            "'my-prompt' (latest version), "
            "'prompts:/my-prompt/3' (specific version, immutable), "
            "'prompts:/my-prompt@production' (alias, mutable). "
            "Requires experiment_tracking.mlflow to be configured. "
            "Mutually exclusive with system_prompt_path."
        ),
    )

    @model_validator(mode="after")
    def _validate_single_source(self) -> InferenceLLMConfig:
        if self.system_prompt_path and self.system_prompt_mlflow_name:
            raise ValueError(
                "inference.llm: specify either system_prompt_path or " "system_prompt_mlflow_name, not both."
            )
        return self


class InferenceCommonConfig(StrictBaseModel):
    """Common inference settings."""

    model_source: str = Field("auto", description="'auto' or explicit ref/path for model artifact.")
    keep_inference_after_eval: bool = Field(
        False,
        description="Keep inference runtime alive after evaluation completes successfully.",
    )
    health_check: InferenceHealthCheckConfig = Field(default_factory=InferenceHealthCheckConfig)  # pyright: ignore[reportArgumentType]
    lora: InferenceLoRAConfig = Field(default_factory=InferenceLoRAConfig)  # pyright: ignore[reportArgumentType]
    chat_ui: InferenceChatUIConfig = Field(default_factory=InferenceChatUIConfig)  # pyright: ignore[reportArgumentType]


__all__ = [
    "InferenceChatUIConfig",
    "InferenceCommonConfig",
    "InferenceHealthCheckConfig",
    "InferenceLLMConfig",
    "InferenceLoRAConfig",
]
