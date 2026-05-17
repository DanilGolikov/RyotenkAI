"""Trainer → runner loopback DTO (POST /api/v1/internal/events).

Phase 2 (ethereal-tumbling-patterson) — the wire body is now a full
typed-envelope JSON dict, validated downstream via
:data:`ryotenkai_shared.events.EVENT_ADAPTER`. The model is a
:class:`pydantic.RootModel` over ``dict[str, Any]`` so the JSON layer
accepts the envelope as-is; the route handler then runs the
discriminator-aware validation. Keeping the DTO schema-free here keeps
the wire-format contract single-sourced inside ``shared.events`` and
avoids duplicating envelope field constraints in two places.

The pre-Phase-2 ``{"kind": str, "payload": dict}`` shape is no longer
accepted — trainers MUST build a typed envelope via the shared codec
before POSTing.
"""

from __future__ import annotations

from typing import Any

from pydantic import RootModel


class InternalEventRequest(RootModel[dict[str, Any]]):
    """Full envelope JSON, validated downstream via ``EVENT_ADAPTER``.

    Loopback only — uvicorn binds to ``127.0.0.1`` so the pod's SSH
    side cannot reach this endpoint.
    """


__all__ = ["InternalEventRequest"]
