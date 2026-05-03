"""Project workspace primitives — an experiment-scoped directory with its own
config history, plugin selection, and runs output. Sibling to the file-based
``PipelineStateStore`` layout.
"""

from __future__ import annotations

from src.workspace.projects.models import (
    ProjectConfigVersion,
    ProjectMetadata,
    ProjectRegistryEntry,
)
from src.workspace.projects.registry import ProjectRegistry
from src.workspace.projects.store import ProjectStore

__all__ = [
    "ProjectConfigVersion",
    "ProjectMetadata",
    "ProjectRegistry",
    "ProjectRegistryEntry",
    "ProjectStore",
]
