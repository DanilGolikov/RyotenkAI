"""Tests for ``pod_session._wait_for_ssh`` — including the early-bailout
path for the RunPod "RUNNING-without-SSH-ports" failure mode (Layer-1
mitigation, plan §2 D5).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from src.providers.runpod.inference.pods import pod_session as ps
from src.utils.result import Failure, Success


def _pod_payload(*, status: str, with_ports: bool) -> dict[str, Any]:
    """Build a GraphQL-shaped payload that ``PodSnapshot.from_graphql`` parses."""
    ports: list[dict[str, Any]] = []
    if with_ports:
        ports.append({
            "isIpPublic": True,
            "privatePort": 22,
            "ip": "1.2.3.4",
            "publicPort": 2222,
            "type": "tcp",
        })
    return {
        "id": "pod-x",
        "desiredStatus": status,
        "runtime": {"uptimeInSeconds": 60, "ports": ports},
    }


def _api_returning(payloads: list[dict[str, Any]]) -> MagicMock:
    """Fake ``api.get_pod`` that returns ``payloads`` cyclically (last value
    sticks once exhausted)."""
    api = MagicMock()
    iterator = iter(payloads)
    last_holder: list[dict[str, Any]] = [payloads[-1]] if payloads else [{}]

    def _get_pod(*, pod_id: str):
        try:
            payload = next(iterator)
            last_holder[0] = payload
        except StopIteration:
            payload = last_holder[0]
        return Success(payload)

    api.get_pod = MagicMock(side_effect=_get_pod)
    return api


def _stable_time_seq(values: list[float]) -> Any:
    """Return a side_effect that yields ``values`` then sticks on the last."""
    state = {"i": 0}

    def _next():
        i = state["i"]
        if i < len(values):
            state["i"] = i + 1
            return values[i]
        return values[-1]

    return _next


class TestHappyPath:
    def test_running_with_ssh_endpoint_and_tcp_open_returns_ok(self) -> None:
        api = _api_returning([_pod_payload(status="RUNNING", with_ports=True)])
        with patch.object(ps, "socket") as sock_mod, \
             patch.object(ps.time, "sleep"):
            sock_mod.create_connection.return_value.__enter__ = MagicMock(return_value=None)
            sock_mod.create_connection.return_value.__exit__ = MagicMock(return_value=False)
            res = ps._wait_for_ssh(api=api, pod_id="pod-x", timeout_sec=10)
        assert res.is_success()
        host, port = res.unwrap()
        assert host == "1.2.3.4"
        assert port == 2222


class TestEarlyBailoutNoPorts:
    def test_running_without_ports_for_threshold_returns_specific_err(self) -> None:
        # All polls return RUNNING with no ports. Drive time forward
        # past the 180-s bailout window deterministically.
        api = _api_returning([_pod_payload(status="RUNNING", with_ports=False)])
        # Iter 1: t=0 (deadline=600), t=10 (while), t=10 (set since), t=20 (elapsed log)
        # Iter 2: t=200 (while), t=200 (delta=190 > 180 → bailout)
        with patch.object(ps.time, "time",
                          side_effect=_stable_time_seq([
                              0.0, 10.0, 10.0, 20.0, 200.0, 200.0,
                          ])), \
             patch.object(ps.time, "sleep"), \
             patch.object(ps, "socket"):
            res = ps._wait_for_ssh(api=api, pod_id="pod-x", timeout_sec=600)

        assert res.is_failure()
        assert res.unwrap_err().code == "POD_SSH_PORTS_NOT_ALLOCATED"

    def test_status_change_resets_no_ports_counter(self) -> None:
        # RUNNING-no-ports → STARTING (transient) → RUNNING-no-ports.
        # The counter must reset on the transient state change so we
        # don't trip the bailout prematurely.
        payloads = [
            _pod_payload(status="RUNNING", with_ports=False),
            _pod_payload(status="STARTING", with_ports=False),
            _pod_payload(status="RUNNING", with_ports=True),  # finally OK
        ]
        api = _api_returning(payloads)

        # Time ticks normally; bailout never triggers because RUNNING
        # window resets in the middle.
        with patch.object(ps.time, "sleep"), \
             patch.object(ps, "socket") as sock_mod:
            sock_mod.create_connection.return_value.__enter__ = MagicMock(return_value=None)
            sock_mod.create_connection.return_value.__exit__ = MagicMock(return_value=False)
            res = ps._wait_for_ssh(api=api, pod_id="pod-x", timeout_sec=600)

        assert res.is_success()


class TestProvisioningKeepsFullTimeout:
    def test_provisioning_status_does_not_trigger_early_bailout(self) -> None:
        # Pod is in PROVISIONING (capacity slow) — early bailout must NOT fire,
        # we need the full 600-s window. Cap the test by exhausting the deadline.
        api = _api_returning([_pod_payload(status="PROVISIONING", with_ports=False)])
        # Tight deadline (1s); time jumps past it on the second iteration.
        with patch.object(ps.time, "time", side_effect=_stable_time_seq([0.0, 0.5, 0.5, 1.5])), \
             patch.object(ps.time, "sleep"):
            res = ps._wait_for_ssh(api=api, pod_id="pod-x", timeout_sec=1)
        assert res.is_failure()
        # Generic timeout, NOT the specific PORTS_NOT_ALLOCATED code.
        assert res.unwrap_err().code == "POD_SSH_READY_TIMEOUT"


class TestApiFailureDoesNotInterruptPolling:
    def test_get_pod_failure_continues_polling(self) -> None:
        api = MagicMock()
        # First call fails, second returns RUNNING+ports.
        api.get_pod = MagicMock(side_effect=[
            Failure(MagicMock(message="api boom")),
            Success(_pod_payload(status="RUNNING", with_ports=True)),
        ])
        with patch.object(ps.time, "sleep"), \
             patch.object(ps, "socket") as sock_mod:
            sock_mod.create_connection.return_value.__enter__ = MagicMock(return_value=None)
            sock_mod.create_connection.return_value.__exit__ = MagicMock(return_value=False)
            res = ps._wait_for_ssh(api=api, pod_id="pod-x", timeout_sec=600)
        assert res.is_success()


class TestTcpProbeFailureDoesNotTriggerBailout:
    def test_ssh_port_present_but_tcp_refused_does_not_count_as_no_ports(self) -> None:
        # Ports are allocated but TCP probe fails — this is a different
        # failure mode (sshd not yet listening). The bailout is for the
        # specific RunPod-stuck-no-ports symptom only; here we should
        # keep waiting for the full timeout.
        api = _api_returning([_pod_payload(status="RUNNING", with_ports=True)])
        with patch.object(ps.time, "time",
                          side_effect=_stable_time_seq([0.0, 0.5, 0.5, 1.5])), \
             patch.object(ps.time, "sleep"), \
             patch.object(ps, "socket") as sock_mod:
            sock_mod.create_connection.side_effect = OSError("refused")
            res = ps._wait_for_ssh(api=api, pod_id="pod-x", timeout_sec=1)
        # Generic timeout — neither bailout nor success.
        assert res.is_failure()
        assert res.unwrap_err().code == "POD_SSH_READY_TIMEOUT"
