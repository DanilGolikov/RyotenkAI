"""Tests for ``pod_session._wait_for_ssh`` — now a thin wrapper over
:class:`PodSshWaiter`.

Phase A2 Batch 11 (2026-05-15): the waiter now raises typed exceptions;
``_wait_for_ssh`` returns ``(host, port)`` directly and re-raises.
"""

from __future__ import annotations

from types import SimpleNamespace

from typing import Any
from unittest.mock import MagicMock

import pytest

from ryotenkai_providers.runpod.inference.pods import pod_session as ps
from ryotenkai_providers.runpod.lifecycle.policy import INFERENCE_PROFILE
from ryotenkai_providers.runpod.models import PodSnapshot, SshEndpoint
from ryotenkai_shared.errors import ProviderUnavailableError

pytestmark = pytest.mark.unit


_SSH_OK = SshEndpoint(host="1.2.3.4", port=2222)


def _ready_snapshot() -> PodSnapshot:
    return PodSnapshot(
        pod_id="pod-x", status="RUNNING", uptime_seconds=10,
        ssh_endpoint=_SSH_OK, port_count=1,
    )


# ---------------------------------------------------------------------------
# Wrapper contract
# ---------------------------------------------------------------------------


def test_wait_for_ssh_returns_host_port_tuple_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class StaticWaiter:
        def __init__(self, *, query: object, policy: object, **_: object) -> None:
            captured["policy"] = policy
            captured["query"] = query

        def wait(self, pod_id: str) -> PodSnapshot:
            captured["pod_id"] = pod_id
            return _ready_snapshot()

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.lifecycle.PodSshWaiter", StaticWaiter,
    )
    api = SimpleNamespace()
    out = ps._wait_for_ssh(api=api, pod_id="pod-x", timeout_sec=600)

    assert out == ("1.2.3.4", 2222)
    assert captured["pod_id"] == "pod-x"


def test_wait_for_ssh_uses_inference_profile_when_timeout_matches_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No transformation when caller passes the profile's own default —
    avoids creating a redundant ``replace`` copy."""
    captured: dict[str, object] = {}

    class StaticWaiter:
        def __init__(self, *, query: object, policy: object, **_: object) -> None:
            captured["policy"] = policy

        def wait(self, pod_id: str) -> PodSnapshot:
            return _ready_snapshot()

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.lifecycle.PodSshWaiter", StaticWaiter,
    )
    ps._wait_for_ssh(
        api=MagicMock(),
        pod_id="pod-x",
        timeout_sec=INFERENCE_PROFILE.total_timeout_s,
    )
    assert captured["policy"] is INFERENCE_PROFILE


def test_wait_for_ssh_overrides_only_total_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom timeout → only ``total_timeout_s`` differs from the profile."""
    captured: dict[str, object] = {}

    class StaticWaiter:
        def __init__(self, *, query: object, policy: object, **_: object) -> None:
            captured["policy"] = policy

        def wait(self, pod_id: str) -> PodSnapshot:
            return _ready_snapshot()

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.lifecycle.PodSshWaiter", StaticWaiter,
    )
    ps._wait_for_ssh(api=MagicMock(), pod_id="pod-x", timeout_sec=42)
    pol = captured["policy"]
    assert pol.total_timeout_s == 42  # type: ignore[attr-defined]
    assert pol.no_exposed_tcp_grace_s == INFERENCE_PROFILE.no_exposed_tcp_grace_s  # type: ignore[attr-defined]
    assert pol.poll_interval_s == INFERENCE_PROFILE.poll_interval_s  # type: ignore[attr-defined]


def test_wait_for_ssh_propagates_waiter_error_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper does not transform exception shape — caller-side branches
    on the canonical waiter error codes (``RUNPOD_*``) via ``context['code']``."""

    class StaticWaiter:
        def __init__(self, **_: object) -> None:
            pass

        def wait(self, pod_id: str) -> PodSnapshot:
            raise ProviderUnavailableError(
                detail="stuck",
                context={"code": "RUNPOD_NO_EXPOSED_TCP", "pod_id": "pod-x"},
            )

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.lifecycle.PodSshWaiter", StaticWaiter,
    )
    with pytest.raises(ProviderUnavailableError) as ei:
        ps._wait_for_ssh(api=MagicMock(), pod_id="pod-x", timeout_sec=600)
    assert ei.value.context["code"] == "RUNPOD_NO_EXPOSED_TCP"
