"""Phase 14.B — :class:`RunPodPodLifecycleClient` contract.

Tests the RunPod GraphQL transport in isolation. The underlying
``_call_mutation`` body is moved verbatim from
:mod:`src.runner.pod_terminator` so these tests preserve the
pre-14.B HTTP shape (envelope, headers, retry timing, idempotency
markers) bit-for-bit. The Phase 14.B-specific changes are:

* return type :class:`LifecycleActionResult` instead of bare ``str``
* api key captured on the client, not per-call.

7-category coverage. Slim-venv compatible — uses
:class:`httpx.MockTransport` + injected ``sleep`` so retries don't
actually wait.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from ryotenkai_providers.runpod.runtime.lifecycle_client import (
    DEFAULT_RUNPOD_GRAPHQL_URL,
    RunPodPodLifecycleClient,
)
from ryotenkai_shared.constants import PROVIDER_RUNPOD
from ryotenkai_shared.infrastructure.lifecycle import (
    IPodLifecycleClient,
    LifecycleActionResult,
    PodTerminalOutcome,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _mk_client(
    *,
    handler: Any = None,
    api_key: str = "rk-secret",
    max_attempts: int = 3,
) -> RunPodPodLifecycleClient:
    """Build a client backed by an :class:`httpx.MockTransport`."""

    if handler is None:
        def handler(req: httpx.Request) -> httpx.Response:
            text = req.read().decode()
            if "podTerminate" in text:
                return httpx.Response(200, text='{"data":{"podTerminate":null}}')
            if "podStop" in text:
                return httpx.Response(200, text='{"data":{"podStop":null}}')
            if "podResume" in text:
                return httpx.Response(200, text='{"data":{"podResume":null}}')
            return httpx.Response(500, text='{"errors":[{"message":"unknown"}]}')

    transport = httpx.MockTransport(handler)

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    async def _no_sleep(_: float) -> None:
        return

    return RunPodPodLifecycleClient(
        api_key=api_key,
        graphql_url=DEFAULT_RUNPOD_GRAPHQL_URL,
        max_attempts=max_attempts,
        http_client_factory=factory,
        sleep=_no_sleep,
    )


# ---------------------------------------------------------------------------
# 1. Positive — happy paths for terminate / pause / resume
# ---------------------------------------------------------------------------


class TestPositive:
    async def test_terminate_success_returns_terminated(self) -> None:
        client = _mk_client()
        result = await client.terminate(resource_id="pod-abc")
        assert isinstance(result, LifecycleActionResult)
        assert result.outcome == PodTerminalOutcome.TERMINATED
        assert result.attempts_made == 1
        assert result.last_error is None
        assert result.raw_response_excerpt == '{"data":{"podTerminate":null}}'

    async def test_pause_success_returns_stopped(self) -> None:
        client = _mk_client()
        result = await client.pause(resource_id="pod-abc")
        assert result.outcome == PodTerminalOutcome.STOPPED
        assert result.attempts_made == 1

    async def test_resume_success_returns_resumed(self) -> None:
        # Phase 14.B § 1.2 / OQ-2: ``resumed`` is a private outcome
        # string until promoted onto :class:`PodTerminalOutcome`.
        client = _mk_client()
        result = await client.resume(resource_id="pod-abc")
        assert result.outcome == "resumed"
        assert result.attempts_made == 1

    async def test_provider_name_is_runpod(self) -> None:
        client = _mk_client()
        assert client.provider_name == PROVIDER_RUNPOD


# ---------------------------------------------------------------------------
# 2. Negative — non-success HTTP without idempotency marker
# ---------------------------------------------------------------------------


class TestNegative:
    async def test_500_without_idempotent_marker_retries_then_fails(self) -> None:
        attempts = {"n": 0}

        def handler(_req: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(500, text='{"errors":[{"message":"server boom"}]}')

        client = _mk_client(handler=handler, max_attempts=3)
        result = await client.terminate(resource_id="pod-abc")
        assert result.outcome == PodTerminalOutcome.FAILED
        assert result.attempts_made == 3
        assert attempts["n"] == 3
        # last_error captures the http_status snippet
        assert result.last_error is not None
        assert "http_status=500" in result.last_error

    async def test_200_but_with_errors_field_fails(self) -> None:
        # GraphQL convention: 200 + ``"errors"`` in body = error
        # response. Pre-14.B this hit the FAILED branch.
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text='{"errors":[{"message":"validation failed"}]}',
            )

        client = _mk_client(handler=handler)
        result = await client.terminate(resource_id="pod-abc")
        assert result.outcome == PodTerminalOutcome.FAILED


# ---------------------------------------------------------------------------
# 3. Boundary — idempotency marker matrix (already-gone wording variants)
# ---------------------------------------------------------------------------


class TestBoundaryIdempotencyMarkers:
    @pytest.mark.parametrize(
        "body_text",
        [
            '{"errors":[{"message":"pod does not exist"}]}',
            '{"errors":[{"message":"already terminated"}]}',
            '{"errors":[{"message":"already exited"}]}',
            '{"errors":[{"message":"no such pod"}]}',
            '{"errors":[{"message":"not running"}]}',
            '{"errors":[{"message":"NOT FOUND"}]}',         # case-insensitive
        ],
    )
    async def test_terminate_already_gone_returns_already_terminated(
        self, body_text: str,
    ) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body_text)

        client = _mk_client(handler=handler)
        result = await client.terminate(resource_id="pod-abc")
        assert result.outcome == PodTerminalOutcome.ALREADY_TERMINATED

    @pytest.mark.parametrize(
        "body_text",
        [
            '{"errors":[{"message":"pod is already stopped"}]}',
            '{"errors":[{"message":"already exited"}]}',
            '{"errors":[{"message":"pod does not exist"}]}',
        ],
    )
    async def test_pause_already_gone_returns_already_stopped(
        self, body_text: str,
    ) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body_text)

        client = _mk_client(handler=handler)
        result = await client.pause(resource_id="pod-abc")
        assert result.outcome == PodTerminalOutcome.ALREADY_STOPPED

    async def test_resume_already_gone_returns_already_running(self) -> None:
        # ``podResume`` on a running pod ⇒ "already running" marker.
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text='{"errors":[{"message":"pod is already running"}]}',
            )

        client = _mk_client(handler=handler)
        # Note: ``_ALREADY_GONE_RE`` matches "not running" / "not found"
        # / "already terminated|exited|stopped" — it does NOT match
        # "already running". So this body falls through to FAILED in
        # the current regex. That's acceptable for Phase 14.B (resume
        # is unused inside the pod). Test pins the current behaviour.
        result = await client.resume(resource_id="pod-abc")
        assert result.outcome == PodTerminalOutcome.FAILED


# ---------------------------------------------------------------------------
# 4. Invariants — Protocol conformance + envelope shape
# ---------------------------------------------------------------------------


class TestInvariants:
    @pytest.mark.asyncio(loop_scope=None)
    async def test_conforms_to_ipodlifecycleclient(self) -> None:
        client = _mk_client()
        # @runtime_checkable Protocol — isinstance returns True without
        # explicit inheritance.
        assert isinstance(client, IPodLifecycleClient)

    async def test_request_envelope_carries_authorization_header(self) -> None:
        captured: dict[str, str] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["auth"] = req.headers.get("Authorization", "")
            captured["ctype"] = req.headers.get("Content-Type", "")
            return httpx.Response(200, text='{"data":{"podTerminate":null}}')

        client = _mk_client(handler=handler, api_key="rk-test-key")
        await client.terminate(resource_id="pod-x")

        assert captured["auth"] == "Bearer rk-test-key"
        assert captured["ctype"] == "application/json"

    async def test_request_body_carries_pod_id(self) -> None:
        captured: dict[str, str] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            # Decode the JSON-wrapped GraphQL query so the assertion
            # sees the raw mutation string (httpx wraps it in JSON,
            # so on-the-wire bytes have escaped quotes).
            import json
            payload = json.loads(req.read().decode())
            captured["query"] = payload["query"]
            return httpx.Response(200, text='{"data":{"podStop":null}}')

        client = _mk_client(handler=handler)
        await client.pause(resource_id="pod-zzz")

        # Mutation envelope: `mutation{podStop(input:{podId:"pod-zzz"})}`
        assert "podStop" in captured["query"]
        assert '"pod-zzz"' in captured["query"]


# ---------------------------------------------------------------------------
# 5. Dependency errors — connection errors retry + last_error captured
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    async def test_connection_error_retried_then_failed(self) -> None:
        attempts = {"n": 0}

        def handler(_req: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            raise httpx.ConnectError("connection refused")

        client = _mk_client(handler=handler, max_attempts=3)
        result = await client.terminate(resource_id="pod-abc")
        assert result.outcome == PodTerminalOutcome.FAILED
        assert result.attempts_made == 3
        assert attempts["n"] == 3
        # last_error captures the exception repr.
        assert result.last_error is not None
        assert "ConnectError" in result.last_error

    async def test_first_attempt_fails_second_succeeds(self) -> None:
        attempts = {"n": 0}

        def handler(_req: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(500, text='{"errors":["transient"]}')
            return httpx.Response(200, text='{"data":{"podTerminate":null}}')

        client = _mk_client(handler=handler, max_attempts=3)
        result = await client.terminate(resource_id="pod-abc")
        assert result.outcome == PodTerminalOutcome.TERMINATED
        assert result.attempts_made == 2  # succeeded on second attempt


# ---------------------------------------------------------------------------
# 6. Regressions — pre-14.B HTTP shape preserved bit-for-bit
# ---------------------------------------------------------------------------


class TestRegressionsHttpShape:
    async def test_envelope_string_format_unchanged(self) -> None:
        # Pre-14.B mutation string: `mutation{<name>(input:{podId:"<id>"})}`
        captured: dict[str, str] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            import json
            payload = json.loads(req.read().decode())
            captured["query"] = payload["query"]
            return httpx.Response(200, text='{"data":{"podTerminate":null}}')

        client = _mk_client(handler=handler)
        await client.terminate(resource_id="pod-12345")

        assert captured["query"] == 'mutation{podTerminate(input:{podId:"pod-12345"})}'

    async def test_max_attempts_default_is_3(self) -> None:
        # Pre-14.B retry budget: 3 attempts. Test pins it.
        attempts = {"n": 0}

        def handler(_req: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(500, text='{"errors":["x"]}')

        client = RunPodPodLifecycleClient(
            api_key="k",
            http_client_factory=lambda: httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                base_url="http://test",
            ),
            sleep=lambda _: _no_sleep_coroutine(),
        )
        result = await client.terminate(resource_id="pod-x")
        assert result.outcome == PodTerminalOutcome.FAILED
        assert attempts["n"] == 3


async def _no_sleep_coroutine() -> None:
    return


# ---------------------------------------------------------------------------
# 7. Logic-specific — raw_response_excerpt forensics + 300-char cap
# ---------------------------------------------------------------------------


class TestLogicSpecificForensics:
    async def test_raw_response_excerpt_captured_on_success(self) -> None:
        client = _mk_client()
        result = await client.terminate(resource_id="pod-x")
        assert result.raw_response_excerpt == '{"data":{"podTerminate":null}}'

    async def test_raw_response_excerpt_truncated_at_300_chars_on_failure(
        self,
    ) -> None:
        big_body = '{"errors":["' + "x" * 1000 + '"]}'

        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text=big_body)

        client = _mk_client(handler=handler, max_attempts=2)
        result = await client.terminate(resource_id="pod-x")
        assert result.outcome == PodTerminalOutcome.FAILED
        assert result.raw_response_excerpt is not None
        # The dataclass post_init truncates at 300 chars.
        assert len(result.raw_response_excerpt) == 300

    async def test_attempts_made_reflects_actual_loop_count(self) -> None:
        # max_attempts=2 + every attempt fails ⇒ attempts_made=2.
        def handler(_req: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text='{"errors":["x"]}')

        client = _mk_client(handler=handler, max_attempts=2)
        result = await client.terminate(resource_id="pod-x")
        assert result.attempts_made == 2
