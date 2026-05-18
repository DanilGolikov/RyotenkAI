"""Loopback endpoints used by the trainer subprocess.

The trainer subprocess (``python -m ryotenkai_pod.trainer.run_training``)
runs inside the same container as the job server. It pushes typed event
envelopes through a HuggingFace ``TrainerCallback``
(:class:`RunnerEventCallback`) to:

    POST http://127.0.0.1:8080/api/v1/internal/events

The router lives behind the same ``API_V1_PREFIX`` as the public
``/jobs`` surface. Security relies on the uvicorn bind being
``127.0.0.1`` only — confirmed in ``docker/training/entrypoint.sh``.

Phase 2 contract (ethereal-tumbling-patterson):

* Request body is the **full envelope JSON** (per the shared event
  taxonomy). The handler validates via
  :data:`ryotenkai_shared.events.EVENT_ADAPTER`; ``ValidationError``
  surfaces as HTTP 422 with the codec's diagnostic detail.
* The handler delegates to :class:`EventBus.publish` so the offset is
  assigned by the bus (R-05: per-source lock); journal append and
  subscriber fan-out are downstream of that single call.
* Response body :class:`EventResponse` carries the assigned offset so
  the trainer-side callback can correlate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request
from pydantic import ValidationError

from ryotenkai_pod.runner.api.deps import get_bus, get_fsm
from ryotenkai_pod.runner.event_bus import envelope_to_wire
from ryotenkai_shared.contracts.runner_api import EventResponse, InternalEventRequest
from ryotenkai_shared.errors import (
    JobSpecInvalidError,
    LoopbackRequiredError,
    NoActiveJobError,
)
from ryotenkai_shared.events import EVENT_ADAPTER, UnknownEvent

if TYPE_CHECKING:
    from ryotenkai_pod.runner.event_bus import EventBus
    from ryotenkai_pod.runner.state import JobLifecycleFSM

router = APIRouter(prefix="/internal", tags=["internal"])


_TRUSTED_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1", "testclient"})


def _require_loopback(request: Request) -> None:
    """Reject requests that didn't arrive over loopback."""
    if request.client is None:
        raise LoopbackRequiredError(
            detail="missing client address (server bind misconfigured)",
        )
    if request.client.host not in _TRUSTED_HOSTS:
        raise LoopbackRequiredError(
            detail=f"non-loopback client refused: {request.client.host}",
            context={"client_host": request.client.host},
        )


@router.post(
    "/events",
    status_code=202,
    response_model=EventResponse,
    summary="Trainer pushes a typed event envelope (loopback only)",
)
def push_event(
    body: InternalEventRequest,
    request: Request,
    fsm: JobLifecycleFSM = Depends(get_fsm),
    bus: EventBus = Depends(get_bus),
) -> EventResponse:
    _require_loopback(request)

    snap = fsm.current()
    if snap is None:
        # Trainer pushing events without a submitted job is a server-
        # side bug (the supervisor would not have spawned the
        # subprocess). 409 is more appropriate than 404 — the
        # endpoint exists, the precondition does not.
        raise NoActiveJobError(
            detail="trainer pushed event but FSM has no active job",
        )

    raw = body.root
    try:
        envelope = EVENT_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        # Wire the codec's structured error through the typed-error
        # contract so the runner emits problem+json (not bare 422).
        raise JobSpecInvalidError(
            detail=f"event envelope failed validation: {exc}",
            cause=exc,
        ) from exc

    # Surface the legacy ``kind`` field on the response — it matches the
    # envelope's typed discriminator. Falls back to the original_type
    # for UnknownEvent so consumers can still see what the trainer
    # tried to emit.
    assigned_offset = bus.publish(envelope)
    wire = envelope_to_wire(envelope.model_copy(update={"offset": assigned_offset}))

    if isinstance(envelope, UnknownEvent):
        kind = envelope.original_type
        relay_payload = envelope.raw_payload
    else:
        kind = envelope.kind
        payload_obj = getattr(envelope, "payload", None)
        if hasattr(payload_obj, "model_dump"):
            relay_payload = payload_obj.model_dump()
        elif isinstance(payload_obj, dict):
            relay_payload = dict(payload_obj)
        else:
            relay_payload = {}

    # MLflow relay was deleted in M7 — Pattern A has HF MLflowCallback
    # adopt MLFLOW_RUN_ID directly inside the trainer process; no
    # cross-process forwarding required.

    return EventResponse(
        offset=assigned_offset,
        timestamp=str(wire.get("timestamp", "")),
        kind=kind,
        payload=relay_payload,
    )
