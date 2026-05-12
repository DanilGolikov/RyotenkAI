"""Pod-cleanup helpers for the control plane.

Phase 1 (greenfield testing) — small, fully-DI'd helpers that
demonstrate the canonical-fakes pattern. They depend on the
provider-agnostic Protocols (``IPodLifecycleClient`` /
``IRunPodAPI``) so the same code paths exercise both real
production transports and the in-memory fakes used by L2 component
tests.
"""

from __future__ import annotations

from ryotenkai_control.cleanup.batch_terminator import BatchPodTerminator, BatchTerminationReport
from ryotenkai_control.cleanup.hibernation_detector import HibernatedPodInfo, HibernationDetector

__all__ = [
    "BatchPodTerminator",
    "BatchTerminationReport",
    "HibernatedPodInfo",
    "HibernationDetector",
]
