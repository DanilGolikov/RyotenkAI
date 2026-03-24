from __future__ import annotations

from pydantic import Field, field_validator

from ..base import StrictBaseModel
from .constants import (
    SINGLE_NODE_PORT_DEFAULT,
    SINGLE_NODE_PORT_MAX,
    SINGLE_NODE_PORT_MIN,
)


class InferenceSingleNodeServeConfig(StrictBaseModel):
    """How to run the inference server on the node."""

    host: str = Field("127.0.0.1", description="Bind host on inference node (MVP default: localhost only).")
    port: int = Field(
        SINGLE_NODE_PORT_DEFAULT,
        ge=SINGLE_NODE_PORT_MIN,
        le=SINGLE_NODE_PORT_MAX,
        description="Bind port for inference server.",
    )
    workspace: str = Field(
        "/home/ml/inference",
        description="Remote workspace directory for inference artifacts/cache (absolute path).",
    )

    @field_validator("workspace")
    @classmethod
    def _validate_abs_workspace(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError("inference.providers.single_node.serve.workspace must be an absolute path")
        return v


__all__ = [
    "InferenceSingleNodeServeConfig",
]
