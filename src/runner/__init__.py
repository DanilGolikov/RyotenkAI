"""RyotenkAI Job Server — in-pod control plane for remote training.

Public surface:
- :func:`create_app` — build the FastAPI app for ``uvicorn``.
- :data:`RUNTIME_IMAGE` — pinned docker image tag the Mac control plane
  uses when provisioning a pod.

Phase 0 status: skeleton only. Real lifecycle / event-bus / supervisor
implementations land in Phase 1+. See ``docs/plans/harmonic-rolling-crayon.md``.
"""

from __future__ import annotations

from src.runner.__about__ import RUNTIME_IMAGE
from src.runner.main import create_app

__all__ = ["RUNTIME_IMAGE", "create_app"]
