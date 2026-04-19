"""Project workspace primitives — an experiment-scoped directory with its own
config history, plugin selection, and runs output. Sibling to the file-based
``PipelineStateStore`` layout.
"""

from __future__ import annotations

from src.pipeline.project.models import (
    ProjectConfigVersion,
    ProjectMetadata,
    ProjectRegistryEntry,
)
from src.pipeline.project.registry import ProjectRegistry
from src.pipeline.project.store import ProjectStore

__all__ = [
    "ProjectConfigVersion",
    "ProjectMetadata",
    "ProjectRegistry",
    "ProjectRegistryEntry",
    "ProjectStore",
]
