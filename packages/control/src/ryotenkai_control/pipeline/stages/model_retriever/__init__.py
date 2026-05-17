"""
model_retriever — backward-compatible package facade.

Phase 4 (event-system unification, 2026-05-16): the legacy
``ModelRetrieverEventCallbacks`` dataclass was removed from the public
surface.

Internal structure:
    types.py        ← ModelCardContext, PhaseMetricsResult
    model_card.py   ← ModelCardGenerator
    hf_uploader.py  ← HFModelUploader
    retriever.py    ← ModelRetriever (thin orchestrator)
"""

from __future__ import annotations

# Re-export SSHClient so that `patch("ryotenkai_control.pipeline.stages.model_retriever.SSHClient")`
# keeps working for tests that patch at the package level.
from ryotenkai_shared.utils.ssh_client import SSHClient  # noqa: F401

from ryotenkai_control.pipeline.stages.model_retriever.hf_uploader import HFModelUploader
from ryotenkai_control.pipeline.stages.model_retriever.metrics_buffer_retriever import (
    FetchResult,
    MetricsBufferRetriever,
)
from ryotenkai_control.pipeline.stages.model_retriever.metrics_replay import (
    BufferedMetricsReplay,
    ReplayResult,
)
from ryotenkai_control.pipeline.stages.model_retriever.model_card import ModelCardGenerator
from ryotenkai_control.pipeline.stages.model_retriever.retriever import ModelRetriever
from ryotenkai_control.pipeline.stages.model_retriever.types import (
    ModelCardContext,
    PhaseMetricsResult,
)

__all__ = [
    "ModelRetriever",
    "ModelCardContext",
    "PhaseMetricsResult",
    "HFModelUploader",
    "ModelCardGenerator",
    # Phase 12.A.1 — metrics buffer retrieval + replay
    "BufferedMetricsReplay",
    "FetchResult",
    "MetricsBufferRetriever",
    "ReplayResult",
    # SSHClient re-exported for test patch compatibility
    "SSHClient",
]
