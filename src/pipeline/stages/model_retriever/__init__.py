"""
model_retriever — backward-compatible package facade.

All names that were previously importable from
`src.pipeline.stages.model_retriever` remain importable unchanged.

Internal structure:
    types.py        ← ModelRetrieverEventCallbacks, ModelCardContext, PhaseMetricsResult
    model_card.py   ← ModelCardGenerator
    hf_uploader.py  ← HFModelUploader
    retriever.py    ← ModelRetriever (thin orchestrator)
"""

from __future__ import annotations

# Re-export SSHClient so that `patch("src.pipeline.stages.model_retriever.SSHClient")`
# keeps working for tests that patch at the package level.
from src.utils.ssh_client import SSHClient  # noqa: F401

from src.pipeline.stages.model_retriever.hf_uploader import HFModelUploader
from src.pipeline.stages.model_retriever.metrics_buffer_retriever import (
    FetchResult,
    MetricsBufferRetriever,
)
from src.pipeline.stages.model_retriever.metrics_replay import (
    BufferedMetricsReplay,
    ReplayResult,
)
from src.pipeline.stages.model_retriever.model_card import ModelCardGenerator
from src.pipeline.stages.model_retriever.retriever import ModelRetriever
from src.pipeline.stages.model_retriever.types import (
    ModelCardContext,
    ModelRetrieverEventCallbacks,
    PhaseMetricsResult,
)

__all__ = [
    "ModelRetriever",
    "ModelRetrieverEventCallbacks",
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
