"""Loopback endpoints used by the trainer subprocess — Phase 0 placeholder.

The trainer subprocess (``python -m src.training.run_training``) runs
inside the same container as the job server. It pushes structured
progress events through a HuggingFace ``TrainerCallback`` to:

    POST http://127.0.0.1:8080/internal/events

The ``RunnerEventCallback`` (Phase 3) buffers locally and flushes
every N steps. This router must remain bound to ``127.0.0.1`` only —
never accept events from the SSH tunnel side.

Phase 0 mounts an empty router so the FastAPI app boots cleanly.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/internal", tags=["internal"])
