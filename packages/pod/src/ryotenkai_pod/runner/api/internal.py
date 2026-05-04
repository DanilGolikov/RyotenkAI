"""Loopback endpoints used by the trainer subprocess.

The trainer subprocess (``python -m src.training.run_training``)
runs inside the same container as the job server. It pushes
structured progress events through a HuggingFace ``TrainerCallback``
(:class:`RunnerEventCallback`, Phase 3) to:

    POST http://127.0.0.1:8080/api/v1/internal/events

The router lives behind the same ``API_V1_PREFIX`` as the public
``/jobs`` surface. Security relies on the uvicorn bind being
``127.0.0.1`` only — confirmed in ``docker/training/entrypoint.sh``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request

from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.contracts.runner_api import EventResponse, InternalEventRequest

from ryotenkai_pod.runner.api.deps import get_bus, get_fsm, get_mlflow_relay
from ryotenkai_pod.runner.api.errors import APIError
from ryotenkai_pod.runner.mlflow_relay import MLFLOW_EVENT_KINDS

if TYPE_CHECKING:
    from ryotenkai_pod.runner.event_bus import EventBus
    from ryotenkai_pod.runner.mlflow_relay import MLflowRelay
    from ryotenkai_pod.runner.state import JobLifecycleFSM

router = APIRouter(prefix="/internal", tags=["internal"])


# IPv4 loopback. Trusted clients reach the server via this address
# (set by uvicorn ``--host 127.0.0.1`` in entrypoint.sh) — anyone
# else either is on the same loopback (the trainer subprocess) or
# is the SSH ``-L`` tunnel terminating at the loopback port from
# the Mac side. The check below is belt-and-suspenders for the
# unlikely case the server is misconfigured to bind 0.0.0.0.
#
# ``testclient`` is the synthetic peer ``fastapi.testclient.TestClient``
# uses for in-process requests — never reachable from a real network,
# so it's safe to whitelist for the test surface.
_TRUSTED_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1", "testclient"})


def _require_loopback(request: Request) -> None:
    """Reject requests that didn't arrive over loopback.

    FastAPI's ``request.client.host`` is the peer address — for
    a uvicorn bound on 127.0.0.1 it is always 127.0.0.1 by
    construction. The check is cheap and surfaces a misconfigured
    bind during integration tests (where someone may inadvertently
    set ``--host 0.0.0.0``).
    """
    if request.client is None:
        raise APIError(
            ErrorCode.LOOPBACK_REQUIRED, status=403,
            detail="missing client address (server bind misconfigured)",
        )
    if request.client.host not in _TRUSTED_HOSTS:
        raise APIError(
            ErrorCode.LOOPBACK_REQUIRED, status=403,
            detail=f"non-loopback client refused: {request.client.host}",
        )


@router.post(
    "/events",
    status_code=202,
    response_model=EventResponse,
    summary="Trainer pushes a progress event (loopback only)",
)
def push_event(
    body: InternalEventRequest,
    request: Request,
    fsm: "JobLifecycleFSM" = Depends(get_fsm),
    bus: "EventBus" = Depends(get_bus),
    mlflow_relay: "MLflowRelay" = Depends(get_mlflow_relay),
) -> EventResponse:
    _require_loopback(request)

    snap = fsm.current()
    if snap is None:
        # Trainer pushing events without a submitted job is a server-
        # side bug (the supervisor would not have spawned the
        # subprocess). 409 is more appropriate than 404 — the
        # endpoint exists, the precondition does not.
        raise APIError(
            ErrorCode.NO_ACTIVE_JOB, status=409,
            detail="trainer pushed event but FSM has no active job",
        )

    event = bus.publish(body.kind, body.payload)

    # MLflow relay (Phase 4.3) — forward MLflow-shaped events to the
    # configured upstream when enabled. Disabled relay returns False
    # and we move on. Non-blocking: ``submit`` is sync, work happens
    # in the relay's worker task.
    if body.kind in MLFLOW_EVENT_KINDS:
        mlflow_relay.submit({"kind": body.kind, "payload": body.payload})

    return EventResponse(
        offset=event.offset,
        timestamp=event.timestamp,
        kind=event.kind,
        payload=dict(event.payload),
    )
