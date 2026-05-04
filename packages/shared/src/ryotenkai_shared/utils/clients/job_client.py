"""HTTP + WebSocket client for the in-pod runner — Phase 5.

The runner serves a small REST + WS surface on ``127.0.0.1:8080``
inside the pod. The Mac side opens an ``ssh -L`` tunnel from a free
local port (chosen by :class:`~src.utils.clients.ssh_tunnel.SSHTunnelManager`)
and points :class:`JobClient` at that local URL. The client knows
nothing about SSH — it just speaks HTTP/WS to whatever ``base_url``
it was constructed with, which makes mock-transport tests trivial.

Public surface:

- :meth:`JobClient.health_check` — ``GET /healthz``
- :meth:`JobClient.submit_job`  — ``POST /api/v1/jobs`` (multipart;
  job spec JSON + optional plugins payload ZIP)
- :meth:`JobClient.get_status`  — ``GET /api/v1/jobs/{id}``
- :meth:`JobClient.request_stop` — ``POST /api/v1/jobs/{id}/stop``
- :meth:`JobClient.subscribe_events` — async iterator over the WS
  ``/api/v1/jobs/{id}/events?since=N`` stream with auto-reconnect

Transport choice: ``httpx.AsyncClient`` for HTTP (already a transitive
dep via FastAPI's TestClient) and ``websockets.connect`` for the WS
stream. We deliberately do NOT use ``httpx-ws`` — the ``websockets``
package is already in :file:`pyproject.toml` and gives us the
reconnect-with-backoff primitives we need.

Reconnect semantics:
The WS stream is lossy at the socket layer but lossless at the bus
layer — every event has a monotonic ``offset`` (see
:class:`src.runner.event_bus.EventBus`). On a connection drop, the
client reconnects with ``?since=<last_offset+1>`` to replay missed
events from the ring buffer. If the buffer rolled past that offset
the server closes with custom code 4410; the client surfaces this
as :class:`ReplayTruncatedError` so the caller can decide whether to
restart from the latest snapshot or give up.

Backoff: 1 s → 2 s → 4 s → 8 s → 16 s → 30 s (capped). Each attempt
adds ±25 % jitter to spread reconnect storms when many clients drop
at once (e.g. tunnel breaks because Mac sleeps).
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import TYPE_CHECKING, Any

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


__all__ = [
    "DEFAULT_RECONNECT_MAX_DELAY",
    "DEFAULT_REQUEST_TIMEOUT",
    "JobClient",
    "JobClientError",
    "JobNotFoundError",
    "ReplayTruncatedError",
]


# Defaults match the runner's expected wire shape. Tests inject
# tighter values to keep the suite fast; production trusts the
# constants below.

DEFAULT_REQUEST_TIMEOUT = 30.0  # seconds — matches httpx default,
# but spelled out so the override path is obvious.
DEFAULT_RECONNECT_INITIAL_DELAY = 1.0
DEFAULT_RECONNECT_MAX_DELAY = 30.0
DEFAULT_RECONNECT_MULTIPLIER = 2.0
DEFAULT_RECONNECT_JITTER = 0.25  # ±25 % around the computed delay

# Custom WebSocket close codes from :mod:`src.runner.api.events` —
# duplicated here as constants so the client can switch on them
# without importing runner-side code (the Mac client must be able to
# build without the runner package installed in dev).
_WS_CLOSE_NOT_FOUND = 4404
_WS_CLOSE_REPLAY_TRUNCATED = 4410
_WS_CLOSE_INVALID_PARAMS = 4422


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class JobClientError(RuntimeError):
    """Base for all client-side failures the caller can act on."""


class JobNotFoundError(JobClientError):
    """The runner replied 404 — the job_id is unknown.

    Surfaced from REST 404s and WS close code 4404. Lets the caller
    distinguish "the runner is reachable but doesn't know this job"
    from "the runner is unreachable" — different recovery paths.
    """


class ReplayTruncatedError(JobClientError):
    """Replay-from-offset went past the oldest event still in the buffer.

    Raised by :meth:`JobClient.subscribe_events` when the server closes
    the WS with code 4410. The caller usually wants to refetch
    :meth:`JobClient.get_status` and resume from the new offset, or
    decide that the gap is irrecoverable.
    """


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class JobClient:
    """Async client for the in-pod runner.

    Construct once per tunnel; multiple jobs over the lifetime of one
    tunnel reuse the same :class:`JobClient`. Closing the client
    (:meth:`aclose` or ``async with``) flushes the underlying
    :class:`httpx.AsyncClient` connection pool — does *not* tear down
    the SSH tunnel, that's the tunnel manager's concern.

    Args:
        base_url: HTTP URL of the local end of the SSH tunnel —
            typically ``http://127.0.0.1:18080`` after the manager
            picks a free port. The trailing slash is normalised.
        request_timeout: per-call HTTP timeout. Multipart uploads
            override this for the body (uploads can take minutes).
        http_client: optional pre-built :class:`httpx.AsyncClient` —
            tests pass one wired to ``httpx.MockTransport`` to avoid
            a real socket. Production passes ``None`` and lets the
            client build its own.
        ws_connect: optional WebSocket factory — tests pass a fake
            that yields canned frames. Production uses
            :func:`websockets.connect`.

    Test seams (``http_client`` / ``ws_connect``) are kept narrow on
    purpose: production code never has to know about them, and tests
    don't have to fork an ssh + uvicorn pair to exercise the wire.
    """

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        http_client: httpx.AsyncClient | None = None,
        ws_connect: Any | None = None,
    ) -> None:
        # Normalise — trailing slash on a base URL trips httpx's join
        # logic in subtle ways; strip it once at boot.
        self._base_url = base_url.rstrip("/")
        self._request_timeout = request_timeout
        # Either inject (tests) or build a fresh pool (production).
        # We track ``_owns_client`` so :meth:`aclose` only closes
        # what we created.
        if http_client is not None:
            self._client = http_client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=request_timeout,
            )
            self._owns_client = True
        # WebSocket factory (tests inject; prod uses real one).
        self._ws_connect = ws_connect or websockets.connect

    # --- lifecycle ---------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying HTTP client (best-effort).

        Tunnels and their sockets are managed elsewhere — this method
        only flushes the HTTP connection pool.
        """
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> JobClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # --- meta probes -------------------------------------------------------

    async def health_check(self) -> bool:
        """``GET /healthz`` — used by the tunnel manager to confirm
        the tunnel is open and the runner is alive before any
        business call goes out.

        Returns ``True`` on 200, ``False`` on any other status or
        transport error. We deliberately swallow connection errors
        here so callers can write a simple ``while not await
        client.health_check(): await asyncio.sleep(...)`` loop.
        """
        try:
            response = await self._client.get("/healthz")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def send_heartbeat(
        self, *, ttl_seconds: float | None = None,
    ) -> bool:
        """Phase 11.E — explicit "control plane is active" ping.

        Used by :class:`ControlPlaneHeartbeat` to tell the in-pod
        runner that the Mac orchestrator process is still alive and
        actively managing this run. While these pings keep landing,
        :class:`MacHeartbeat` stays fresh and :class:`PodTerminator`
        will pick the SHORT_GRACE / "alive" terminal-hook path
        regardless of WS / REST traffic.

        The ping is fire-and-forget from the orchestrator's
        standpoint — transient httpx errors return ``False`` and the
        caller logs + retries on the next interval. The pod side
        retry logic in :class:`PodTerminator` accommodates a single
        missed ping cycle.

        Args:
            ttl_seconds: Override the runner's default explicit TTL
                (120 s). Mostly useful for tests; production code
                passes ``None``.

        Returns:
            ``True`` on a 200 response, ``False`` on any non-200
            status or transport error.
        """
        body: dict[str, Any] = {}
        if ttl_seconds is not None:
            body["ttl_seconds"] = float(ttl_seconds)
        try:
            response = await self._client.post(
                "/api/v1/control/heartbeat",
                json=body,
            )
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    # --- REST endpoints ----------------------------------------------------

    async def submit_job(
        self,
        job_spec: dict[str, Any],
        *,
        plugins_payload: bytes | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Submit a job via multipart POST.

        Wire shape (per :mod:`src.runner.api.jobs`):

        - ``job_spec`` is a multipart **form field** (string), not a
          file. The runner parses ``json.loads`` on the value.
        - ``plugins_payload`` is a multipart **file part** and is
          required by the runner (``File(...)``). When the caller
          passes ``None`` we send an empty ZIP-like blob so the
          contract holds — the Phase 6 PluginUnpacker tolerates
          empty bundles for SFT-only jobs.
        - The runner replies with ``202 Accepted`` plus
          :class:`JobSubmittedResponse` (``job_id``, ``sequence``,
          ``offset``).

        Args:
            job_spec: JSON-serialisable dict matching
                :class:`ryotenkai_shared.contracts.runner_api.JobSpec` (the runner
                validates with ``extra="forbid"``, so unknown fields
                blow up at submit time, not at first event).
            plugins_payload: optional ZIP bytes — the packed
                ``community/`` payload (see :mod:`src.community.pack`).
                ``None`` sends an empty placeholder.
            timeout: per-request override. Multipart uploads can be
                tens of MB on a slow link; we let the caller raise
                this above the default 30 s when needed.

        Returns:
            The runner's response body as a dict — typically
            ``{"job_id": "...", "sequence": 0, "offset": 0}``.

        Raises:
            JobClientError: any non-2xx response. The exception
                message includes status code and body for debugging.
        """
        # ``data`` for form fields, ``files`` for file parts. Mixing
        # them in one call is httpx's documented multipart shape.
        data = {"job_spec": json.dumps(job_spec)}
        files: dict[str, tuple[str, bytes, str]] = {
            "plugins_payload": (
                "plugins.zip",
                plugins_payload if plugins_payload is not None else b"",
                "application/zip",
            ),
        }

        try:
            response = await self._client.post(
                "/api/v1/jobs",
                data=data,
                files=files,
                timeout=timeout if timeout is not None else self._request_timeout,
            )
        except httpx.HTTPError as exc:
            raise JobClientError(f"submit failed: transport error: {exc!r}") from exc

        # Runner returns 202 for accepted-and-spawning. Tolerate 200
        # too in case a future revision drops the ``status_code=202``
        # decorator — we'd rather succeed than break on a harmless
        # spec change.
        if response.status_code not in (200, 202):
            raise JobClientError(
                f"submit failed: {response.status_code} {response.text[:300]}",
            )

        return response.json()  # type: ignore[no-any-return]

    async def get_status(self, job_id: str) -> dict[str, Any]:
        """``GET /api/v1/jobs/{id}`` — current snapshot for ``job_id``.

        Raises:
            JobNotFoundError: 404 from the runner.
            JobClientError: any other non-2xx, or a transport error.
        """
        try:
            response = await self._client.get(f"/api/v1/jobs/{job_id}")
        except httpx.HTTPError as exc:
            raise JobClientError(f"get_status transport error: {exc!r}") from exc

        if response.status_code == 404:
            raise JobNotFoundError(f"unknown job_id: {job_id!r}")
        if response.status_code != 200:
            raise JobClientError(
                f"get_status failed: {response.status_code} {response.text[:300]}",
            )
        return response.json()  # type: ignore[no-any-return]

    async def request_stop(
        self,
        job_id: str,
        *,
        grace_seconds: float | None = None,
    ) -> dict[str, Any]:
        """``POST /api/v1/jobs/{id}/stop`` — request graceful stop.

        The runner returns 202 the moment SIGTERM is in flight (see
        :meth:`src.runner.supervisor.Supervisor.request_stop`); the
        FSM transitions to ``cancelled`` once the trainer reaps. Use
        :meth:`subscribe_events` or :meth:`get_status` to wait for
        the terminal state.

        Args:
            grace_seconds: optional override of the supervisor's
                default 30 s SIGTERM-to-SIGKILL window. Pass ``None``
                to use the server-side default.

        Raises:
            JobNotFoundError / JobClientError: same shape as
                :meth:`get_status`.
        """
        body: dict[str, Any] = {}
        if grace_seconds is not None:
            body["grace_seconds"] = grace_seconds

        try:
            response = await self._client.post(
                f"/api/v1/jobs/{job_id}/stop",
                json=body,
            )
        except httpx.HTTPError as exc:
            raise JobClientError(f"stop transport error: {exc!r}") from exc

        if response.status_code == 404:
            raise JobNotFoundError(f"unknown job_id: {job_id!r}")
        if response.status_code not in (200, 202):
            raise JobClientError(
                f"stop failed: {response.status_code} {response.text[:300]}",
            )
        return response.json() if response.content else {}  # type: ignore[no-any-return]

    # --- WebSocket event stream -------------------------------------------

    async def subscribe_events(
        self,
        job_id: str,
        *,
        since: int = 0,
        max_reconnect_attempts: int | None = None,
        initial_delay: float = DEFAULT_RECONNECT_INITIAL_DELAY,
        max_delay: float = DEFAULT_RECONNECT_MAX_DELAY,
        sleep: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async iterator over the WS event stream with auto-reconnect.

        Each yielded item is the JSON-decoded event payload (with at
        least ``offset``, ``timestamp``, ``kind``, ``payload`` keys
        from :class:`src.runner.event_bus.Event`). The ``offset``
        field is what the iterator passes to ``?since=`` on
        reconnect, so a Mac client that briefly loses its tunnel
        catches up automatically the moment the tunnel comes back.

        This is an async generator — call as
        ``async for event in client.subscribe_events(job_id): ...``.
        No ``await`` needed before the loop.

        Args:
            job_id: which job's events to stream.
            since: starting offset. ``0`` replays from the oldest
                event still in the buffer (or the start of stream
                if the buffer has never rolled).
            max_reconnect_attempts: cap on reconnection retries.
                ``None`` = retry forever (typical: keep the iterator
                alive while the user's terminal stays open). Set to
                a positive int for tests / scripts.
            initial_delay / max_delay: exponential backoff bounds.
            sleep: injectable sleep — tests pass a no-op so the
                backoff doesn't actually wait.

        Raises:
            JobNotFoundError: WS close code 4404.
            ReplayTruncatedError: WS close code 4410. The caller
                should refetch :meth:`get_status`, decide whether to
                restart from the latest offset, and re-call
                :meth:`subscribe_events`.
            JobClientError: invalid params (4422) or backoff
                exhausted.
        """
        sleep_fn = sleep or asyncio.sleep
        ws_url = self._http_to_ws_url(f"/api/v1/jobs/{job_id}/events")

        current_since = since
        attempt = 0
        delay = initial_delay

        while True:
            try:
                async for event in self._consume_one_session(
                    ws_url, since=current_since,
                ):
                    yield event
                    # Track latest offset for resume on next
                    # reconnect — even one event saves a replay
                    # window on the server side.
                    if isinstance(event, dict):
                        offset = event.get("offset")
                        if isinstance(offset, int):
                            current_since = offset + 1
                # The session ended cleanly (server closed code
                # 1000 or stream EOF). Treat it as a normal
                # terminal — the caller decides whether to
                # resubscribe.
                return
            except JobNotFoundError:
                raise
            except ReplayTruncatedError:
                raise
            except JobClientError:
                # Invalid params (4422) etc. — not retriable.
                raise
            except (ConnectionClosed, OSError, TimeoutError):
                # Transient — back off and retry.
                attempt += 1
                if (
                    max_reconnect_attempts is not None
                    and attempt > max_reconnect_attempts
                ):
                    raise JobClientError(
                        f"reconnect attempts exhausted "
                        f"(max={max_reconnect_attempts})",
                    ) from None
                await sleep_fn(_apply_jitter(delay))
                delay = min(delay * DEFAULT_RECONNECT_MULTIPLIER, max_delay)

    # --- internals ---------------------------------------------------------

    def _http_to_ws_url(self, path: str) -> str:
        """Translate ``http://host`` → ``ws://host`` (and tls variants)
        and append ``path``. Hand-rolled rather than via ``urllib.parse``
        because the base URL is always one of two known schemes — keeps
        the function trivially testable.
        """
        if self._base_url.startswith("https://"):
            return "wss://" + self._base_url[len("https://"):] + path
        if self._base_url.startswith("http://"):
            return "ws://" + self._base_url[len("http://"):] + path
        # Already ws://wss://, or schemeless. Trust caller.
        return self._base_url + path

    async def _consume_one_session(
        self, ws_url: str, *, since: int,
    ) -> AsyncIterator[dict[str, Any]]:
        """Open a single WS connection, yield events until the socket
        closes, translate close codes into typed exceptions.

        Split out so :meth:`subscribe_events` can wrap it in a
        retry loop without nested try/except hell.
        """
        url = f"{ws_url}?since={since}"
        try:
            async with self._ws_connect(url) as ws:
                async for raw in ws:
                    if isinstance(raw, (bytes, bytearray)):
                        raw = raw.decode("utf-8", errors="replace")
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError:
                        # Skip malformed frames — the runner only
                        # ever emits valid JSON, but we'd rather
                        # tolerate a corrupt frame than kill the
                        # whole subscription.
                        continue
        except ConnectionClosed as exc:
            # Translate runner-defined close codes into typed errors.
            # ``exc.rcvd.code`` is the modern API (websockets ≥13.0);
            # the deprecated ``exc.code`` is avoided to keep us silent
            # on warnings as the dep gets pinned forward.
            code: int | None = None
            rcvd = getattr(exc, "rcvd", None)
            if rcvd is not None:
                code = getattr(rcvd, "code", None)
            if code == _WS_CLOSE_NOT_FOUND:
                raise JobNotFoundError(
                    "unknown job (ws close 4404)",
                ) from exc
            if code == _WS_CLOSE_REPLAY_TRUNCATED:
                raise ReplayTruncatedError(
                    "replay offset rolled past oldest buffered event",
                ) from exc
            if code == _WS_CLOSE_INVALID_PARAMS:
                raise JobClientError(
                    "invalid ws params (close 4422)",
                ) from exc
            # Any other close code — let the retry loop handle it
            # as a transient.
            raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_jitter(delay: float) -> float:
    """Scale ``delay`` by a uniform random factor in
    ``[1 - DEFAULT_RECONNECT_JITTER, 1 + DEFAULT_RECONNECT_JITTER]``.

    Spreads thundering-herd reconnects when many clients drop at
    once (e.g. tunnel breaks). Floor at zero — paranoid, since
    DEFAULT_RECONNECT_JITTER is below 1, but cheap.
    """
    span = DEFAULT_RECONNECT_JITTER
    factor = 1.0 + random.uniform(-span, span)
    return max(0.0, delay * factor)
