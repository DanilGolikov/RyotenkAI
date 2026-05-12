"""Phase 5 — :class:`src.utils.clients.JobClient` contract.

The client is the Mac side's only interface to the runner. Coverage:

- TestHealthCheck       liveness probe (true / false / transport error)
- TestSubmitJob         multipart shape, with and without payload, errors
- TestGetStatus         200 → dict, 404 → JobNotFoundError, 500 → JobClientError
- TestRequestStop       grace-seconds threading, 202 acceptable, errors
- TestSubscribeEvents   WebSocket event flow, reconnect, close codes,
                        offset advancement
- TestUrlScheme         http→ws / https→wss
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ryotenkai_shared.utils.clients.job_client import (
    JobClient,
    JobClientError,
    JobNotFoundError,
    ReplayTruncatedError,
)
from ryotenkai_shared.utils.clients.problem_details import APIException

# Each async class opts in below via @pytest.mark.asyncio. The
# synchronous URL-scheme tests stay unmarked.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client_with_handler(handler) -> JobClient:
    """Build a :class:`JobClient` whose HTTP layer is wired to a
    :class:`httpx.MockTransport` running ``handler``. WebSocket
    factory is left as the real one — tests that exercise WS pass
    a custom factory inline."""
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:18080")
    return JobClient("http://127.0.0.1:18080", http_client=http)


class _FakeWebSocket:
    """Minimal duck-type for :func:`websockets.connect`'s context
    manager. Iterates over ``frames`` then optionally raises
    ``raise_on_exhaustion``.
    """

    def __init__(
        self,
        frames: list[str],
        *,
        raise_on_exhaustion: BaseException | None = None,
    ) -> None:
        self._frames = list(frames)
        self._raise = raise_on_exhaustion

    def __aiter__(self) -> _FakeWebSocket:
        return self

    async def __anext__(self) -> str:
        if not self._frames:
            if self._raise is not None:
                raise self._raise
            raise StopAsyncIteration
        return self._frames.pop(0)

    async def __aenter__(self) -> _FakeWebSocket:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


def _ws_factory_from_sessions(sessions: list[_FakeWebSocket]):
    """Return a callable that, on each invocation, hands back the
    next session in order. Lets a test simulate "first session
    drops, second session reconnects" deterministically."""
    iterator = iter(sessions)

    def _connect(_url: str) -> _FakeWebSocket:
        try:
            return next(iterator)
        except StopIteration:
            raise AssertionError(
                "ws_factory ran out of sessions — test scripted too few",
            )
    return _connect


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHealthCheck:
    async def test_returns_true_on_200(self) -> None:
        client = _client_with_handler(
            lambda _r: httpx.Response(200, json={"status": "ok"}),
        )
        try:
            assert await client.health_check() is True
        finally:
            await client.aclose()

    async def test_returns_false_on_503(self) -> None:
        client = _client_with_handler(
            lambda _r: httpx.Response(503, text="draining"),
        )
        try:
            assert await client.health_check() is False
        finally:
            await client.aclose()

    async def test_returns_false_on_transport_error(self) -> None:
        def _raise(_r: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns")

        client = _client_with_handler(_raise)
        try:
            assert await client.health_check() is False
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# submit_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSubmitJob:
    async def test_multipart_includes_job_spec_as_form_field(self) -> None:
        captured: dict[str, Any] = {}

        def _h(request: httpx.Request) -> httpx.Response:
            captured["content_type"] = request.headers.get("content-type", "")
            captured["body"] = request.content
            return httpx.Response(
                202, json={"job_id": "j-1", "sequence": 0, "offset": 0},
            )

        client = _client_with_handler(_h)
        try:
            result = await client.submit_job(
                {"job_id": "j-1", "command": ["python", "-c", "pass"]},
            )
        finally:
            await client.aclose()

        assert result == {"job_id": "j-1", "sequence": 0, "offset": 0}
        assert "multipart/form-data" in captured["content_type"]
        # ``job_spec`` is sent as a multipart form field (not a file
        # part), which is the runner's wire contract.
        assert b'name="job_spec"' in captured["body"]
        assert b'"job_id": "j-1"' in captured["body"]
        # ``plugins_payload`` is the file part — always present even
        # when the caller passes ``None`` (empty placeholder).
        assert b'name="plugins_payload"' in captured["body"]

    async def test_with_plugins_payload_includes_zip_bytes(self) -> None:
        captured: dict[str, Any] = {}

        def _h(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.content
            return httpx.Response(202, json={"job_id": "j-2", "sequence": 0, "offset": 0})

        client = _client_with_handler(_h)
        try:
            await client.submit_job(
                {"job_id": "j-2", "command": ["x"]},
                plugins_payload=b"PK\x03\x04fake-zip-bytes",
            )
        finally:
            await client.aclose()

        assert b"plugins_payload" in captured["body"]
        assert b"PK\x03\x04fake-zip-bytes" in captured["body"]

    async def test_non_2xx_raises_problem_details(self) -> None:
        # Phase 3 PR-3.3: non-2xx responses are parsed via
        # ``parse_problem_details``. Plain-text bodies surface as
        # ``TransportError`` (subclass of ``APIException``).
        client = _client_with_handler(
            lambda _r: httpx.Response(409, text="busy"),
        )
        try:
            with pytest.raises(APIException, match="409"):
                await client.submit_job({"job_id": "x", "command": ["x"]})
        finally:
            await client.aclose()

    async def test_transport_error_wrapped(self) -> None:
        def _raise(_r: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns")

        client = _client_with_handler(_raise)
        try:
            with pytest.raises(JobClientError, match="transport error"):
                await client.submit_job({"job_id": "x", "command": ["x"]})
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetStatus:
    async def test_200_returns_dict(self) -> None:
        client = _client_with_handler(
            lambda _r: httpx.Response(
                200, json={"job_id": "j-1", "state": "running"},
            ),
        )
        try:
            assert await client.get_status("j-1") == {
                "job_id": "j-1", "state": "running",
            }
        finally:
            await client.aclose()

    async def test_404_raises_not_found(self) -> None:
        client = _client_with_handler(lambda _r: httpx.Response(404))
        try:
            with pytest.raises(JobNotFoundError, match="j-missing"):
                await client.get_status("j-missing")
        finally:
            await client.aclose()

    async def test_500_raises_problem_details(self) -> None:
        # Phase 3 PR-3.3: non-2xx → parse_problem_details. Text body
        # → TransportError (APIException subclass).
        client = _client_with_handler(lambda _r: httpx.Response(500, text="boom"))
        try:
            with pytest.raises(APIException, match="500"):
                await client.get_status("j-1")
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# request_stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRequestStop:
    async def test_default_grace_omits_field(self) -> None:
        captured: dict[str, Any] = {}

        def _h(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content or b"{}")
            return httpx.Response(202)

        client = _client_with_handler(_h)
        try:
            await client.request_stop("j-1")
        finally:
            await client.aclose()
        assert captured["body"] == {}

    async def test_grace_seconds_threaded_through(self) -> None:
        captured: dict[str, Any] = {}

        def _h(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content or b"{}")
            return httpx.Response(202, json={"ok": True})

        client = _client_with_handler(_h)
        try:
            await client.request_stop("j-1", grace_seconds=15.0)
        finally:
            await client.aclose()
        assert captured["body"] == {"grace_seconds": 15.0}

    async def test_404_raises_not_found(self) -> None:
        client = _client_with_handler(lambda _r: httpx.Response(404))
        try:
            with pytest.raises(JobNotFoundError):
                await client.request_stop("j-x")
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# subscribe_events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSubscribeEvents:
    async def test_yields_decoded_events(self) -> None:
        ws = _FakeWebSocket(frames=[
            json.dumps({"offset": 0, "kind": "trainer_spawned", "payload": {}}),
            json.dumps({"offset": 1, "kind": "trainer_log", "payload": {"line": "hi"}}),
        ])
        client = JobClient(
            "http://127.0.0.1:18080",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _r: httpx.Response(404)),
            ),
            ws_connect=_ws_factory_from_sessions([ws]),
        )
        try:
            collected = []
            async for event in client.subscribe_events("j-1"):
                collected.append(event)
        finally:
            await client.aclose()
        assert [e["kind"] for e in collected] == ["trainer_spawned", "trainer_log"]
        assert [e["offset"] for e in collected] == [0, 1]

    async def test_reconnect_after_transient_drop(self) -> None:
        from websockets.exceptions import ConnectionClosed

        # Session 1 emits one event then "drops" (raises ConnectionClosed
        # without a runner-typed code). Session 2 picks up where we left
        # off.
        first = _FakeWebSocket(
            frames=[
                json.dumps({"offset": 0, "kind": "trainer_spawned", "payload": {}}),
            ],
            raise_on_exhaustion=ConnectionClosed(rcvd=None, sent=None),
        )
        second = _FakeWebSocket(
            frames=[
                json.dumps({"offset": 1, "kind": "trainer_log", "payload": {}}),
            ],
        )

        urls_seen: list[str] = []
        sessions = iter([first, second])

        def _connect(url: str) -> _FakeWebSocket:
            urls_seen.append(url)
            return next(sessions)

        async def _no_sleep(_s: float) -> None:
            return None

        client = JobClient(
            "http://127.0.0.1:18080",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _r: httpx.Response(404)),
            ),
            ws_connect=_connect,
        )
        try:
            collected = []
            async for event in client.subscribe_events("j-1", sleep=_no_sleep):
                collected.append(event)
        finally:
            await client.aclose()

        assert [e["offset"] for e in collected] == [0, 1]
        # Reconnect URL must include since=1 (offset 0 + 1).
        assert "since=0" in urls_seen[0]
        assert "since=1" in urls_seen[1]

    async def test_close_code_4404_raises_not_found(self) -> None:
        from websockets.exceptions import ConnectionClosed
        from websockets.frames import Close

        ws = _FakeWebSocket(
            frames=[],
            raise_on_exhaustion=ConnectionClosed(
                rcvd=Close(code=4404, reason="not found"),
                sent=None,
            ),
        )
        client = JobClient(
            "http://127.0.0.1:18080",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _r: httpx.Response(404)),
            ),
            ws_connect=_ws_factory_from_sessions([ws]),
        )
        try:
            with pytest.raises(JobNotFoundError):
                async for _ in client.subscribe_events("j-missing"):
                    pass
        finally:
            await client.aclose()

    async def test_close_code_4410_raises_replay_truncated(self) -> None:
        from websockets.exceptions import ConnectionClosed
        from websockets.frames import Close

        ws = _FakeWebSocket(
            frames=[],
            raise_on_exhaustion=ConnectionClosed(
                rcvd=Close(code=4410, reason="truncated"),
                sent=None,
            ),
        )
        client = JobClient(
            "http://127.0.0.1:18080",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _r: httpx.Response(404)),
            ),
            ws_connect=_ws_factory_from_sessions([ws]),
        )
        try:
            with pytest.raises(ReplayTruncatedError):
                async for _ in client.subscribe_events("j-1", since=999_999):
                    pass
        finally:
            await client.aclose()

    async def test_max_reconnect_attempts_exhausted(self) -> None:
        from websockets.exceptions import ConnectionClosed

        # Every session drops without a typed close code → triggers
        # the retry path. With max_reconnect_attempts=2 we get exactly
        # 1 initial + 2 retries = 3 sessions, then JobClientError.
        sessions = [
            _FakeWebSocket(
                frames=[],
                raise_on_exhaustion=ConnectionClosed(rcvd=None, sent=None),
            )
            for _ in range(3)
        ]

        async def _no_sleep(_s: float) -> None:
            return None

        client = JobClient(
            "http://127.0.0.1:18080",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _r: httpx.Response(404)),
            ),
            ws_connect=_ws_factory_from_sessions(sessions),
        )
        try:
            with pytest.raises(JobClientError, match="reconnect attempts exhausted"):
                async for _ in client.subscribe_events(
                    "j-1",
                    max_reconnect_attempts=2,
                    sleep=_no_sleep,
                ):
                    pass
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# URL scheme translation
# ---------------------------------------------------------------------------


class TestUrlScheme:
    def test_http_to_ws(self) -> None:
        client = JobClient(
            "http://127.0.0.1:18080",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _r: httpx.Response(200)),
            ),
        )
        assert client._http_to_ws_url("/x") == "ws://127.0.0.1:18080/x"

    def test_https_to_wss(self) -> None:
        client = JobClient(
            "https://example.com",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _r: httpx.Response(200)),
            ),
        )
        assert client._http_to_ws_url("/y") == "wss://example.com/y"

    def test_trailing_slash_stripped(self) -> None:
        client = JobClient(
            "http://127.0.0.1:18080/",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _r: httpx.Response(200)),
            ),
        )
        assert client._base_url == "http://127.0.0.1:18080"
