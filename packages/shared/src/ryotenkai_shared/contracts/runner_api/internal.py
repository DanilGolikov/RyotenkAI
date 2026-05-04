"""Trainer → runner loopback DTO (POST /api/v1/internal/events)."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ._strict import _StrictModel


class InternalEventRequest(_StrictModel):
    """Body of ``POST /internal/events`` — published by the trainer
    subprocess via a HuggingFace ``TrainerCallback`` (Phase 3).

    Loopback only — uvicorn binds to ``127.0.0.1`` so the pod's
    SSH side cannot reach this endpoint.
    """

    kind: str = Field(min_length=1, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)


__all__ = ["InternalEventRequest"]
