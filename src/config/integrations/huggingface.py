from __future__ import annotations

from pydantic import Field

from ..base import StrictBaseModel


class HuggingFaceHubConfig(StrictBaseModel):
    """
    HuggingFace Hub - model hosting and sharing.

    Setup:
        1. Create account at huggingface.co
        2. Get token from Settings > Access Tokens
        3. Set env: HF_TOKEN=hf_xxxxx

    Env vars (required):
        HF_TOKEN - your access token (write permission)
    """

    enabled: bool = Field(..., description="Enable HF Hub upload")
    repo_id: str = Field(..., description="Full repo ID: username/model-name")
    private: bool = Field(..., description="Make repo private")


# Backward compatibility alias
HuggingFaceConfig = HuggingFaceHubConfig


__all__ = [
    "HuggingFaceConfig",
    "HuggingFaceHubConfig",
]
