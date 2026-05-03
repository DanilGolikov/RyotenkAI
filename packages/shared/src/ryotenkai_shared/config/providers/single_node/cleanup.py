from __future__ import annotations

from pydantic import Field

from ...base import StrictBaseModel


class SingleNodeCleanupConfig(StrictBaseModel):
    """Cleanup policy for single_node provider."""

    cleanup_workspace: bool = Field(False, description="Delete run directory after training")
    keep_on_error: bool = Field(True, description="Keep workspace on error for debugging")
    on_interrupt: bool = Field(
        True,
        description="Apply cleanup policy when pipeline is interrupted via Ctrl+C (SIGINT).",
    )


__all__ = [
    "SingleNodeCleanupConfig",
]
