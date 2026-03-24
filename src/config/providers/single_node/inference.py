from __future__ import annotations

from pydantic import Field

from ...base import StrictBaseModel
from ...inference import InferenceSingleNodeServeConfig
from ...inference.common import InferenceLLMConfig


class SingleNodeInferenceConfig(StrictBaseModel):
    """
    Inference-specific settings for single_node provider.

    Example:
        inference:
          serve:
            host: "127.0.0.1"
            port: 8000
            workspace: "/home/user/inference"
    """

    serve: InferenceSingleNodeServeConfig = Field(
        default_factory=InferenceSingleNodeServeConfig,  # type: ignore[arg-type]
        description="Inference server settings",
    )
    llm: InferenceLLMConfig = Field(
        default_factory=InferenceLLMConfig,  # type: ignore[arg-type]
        description="LLM execution settings (system prompt, etc.).",
    )


__all__ = [
    "SingleNodeInferenceConfig",
]
