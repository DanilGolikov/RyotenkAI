"""Phase 11.E — ``POST /api/v1/control/heartbeat`` contract.

Pin the wire shape of the new control-plane heartbeat endpoint:
* Empty body → applies the runner default explicit TTL (120 s).
* Custom ``ttl_seconds`` → applied as-is.
* Negative / zero TTL → 422 (Pydantic ``gt=0`` constraint).
* Result echoes the applied TTL for client-side sanity checking.
* Heartbeat ledger actually refreshed.
"""

from __future__ import annotations

from typing import Any


def _post_heartbeat(client, body: dict[str, Any] | None = None):
    """Send POST /api/v1/control/heartbeat with optional body.

    Empty / no-body posts use ``json={}`` so FastAPI's body-required
    behaviour is satisfied without depending on transport-level
    'no Content-Type → optional body' subtleties. The Mac client
    (:class:`ControlPlaneHeartbeat`) always sends ``json={}`` when
    no TTL override is provided.
    """
    return client.post(
        "/api/v1/control/heartbeat",
        json=body if body is not None else {},
    )


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_no_body_applies_default_ttl(self, runner_client) -> None:
        resp = _post_heartbeat(runner_client)
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        # Default explicit TTL = 120s (Phase 11.E).
        assert body["ttl_seconds_applied"] == 120.0

    def test_explicit_ttl_applied(self, runner_client) -> None:
        resp = _post_heartbeat(runner_client, {"ttl_seconds": 90.0})
        assert resp.status_code == 200
        assert resp.json()["ttl_seconds_applied"] == 90.0

    def test_heartbeat_ledger_refreshed(self, runner_client) -> None:
        # Before the ping: heartbeat is NOT alive (fresh app).
        heartbeat = runner_client.app.state.heartbeat
        assert heartbeat.is_alive() is False
        # Ping with 120s TTL.
        resp = _post_heartbeat(runner_client, {"ttl_seconds": 120.0})
        assert resp.status_code == 200
        # After ping: alive.
        assert heartbeat.is_alive() is True


# ---------------------------------------------------------------------------
# 2. Negative — invalid TTL rejected
# ---------------------------------------------------------------------------


class TestNegative:
    def test_zero_ttl_rejected(self, runner_client) -> None:
        resp = _post_heartbeat(runner_client, {"ttl_seconds": 0.0})
        assert resp.status_code == 422

    def test_negative_ttl_rejected(self, runner_client) -> None:
        resp = _post_heartbeat(runner_client, {"ttl_seconds": -1.0})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_explicit_ttl_null_uses_default(self, runner_client) -> None:
        # Phase 11.E — pinging with explicit `null` should use default.
        resp = _post_heartbeat(runner_client, {"ttl_seconds": None})
        assert resp.status_code == 200
        assert resp.json()["ttl_seconds_applied"] == 120.0

    def test_idempotent_repeated_pings(self, runner_client) -> None:
        # Two pings in a row both succeed; ledger stays alive.
        for _ in range(3):
            resp = _post_heartbeat(runner_client)
            assert resp.status_code == 200
        assert runner_client.app.state.heartbeat.is_alive() is True


# ---------------------------------------------------------------------------
# 4. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_response_shape_pinned(self, runner_client) -> None:
        # Pin the exact response keys so client-side parsers don't
        # accidentally rely on undocumented fields.
        resp = _post_heartbeat(runner_client)
        body = resp.json()
        assert set(body.keys()) == {"ok", "ttl_seconds_applied"}
