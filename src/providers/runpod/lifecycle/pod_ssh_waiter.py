"""Single canonical pod-readiness primitive.

Replaces three near-duplicate wait loops:

* ``PodLifecycleManager.wait_for_ready`` — training-side connect
* ``pod_session._wait_for_ssh`` — inference eval-session activation
* ``artifacts._wait_for_pod_ssh_ready`` — chat-script (out-of-process,
  separate manual mirror — see ``artifacts.py`` docstring)

The waiter takes a :class:`PodQuery` (Protocol-typed dependency
satisfied by both ``RunPodTrainingPodControl`` and
``RunPodInferencePodControl``) and three injectable seams: the cancel-
aware sleep (``sleep_cancellable`` by default), the clock
(``time.monotonic`` for NTP-step safety), and the TCP probe.

Cancellation contract: the waiter does NOT catch
:class:`PipelineCancelled` — it lets the exception propagate so the
provider's ``except PipelineCancelled:`` cleanup hook can synchronously
terminate the in-flight pod. This is intentional layer-separation; the
waiter's job is "is the pod ready or stuck?", not "what to do on
cancel?".

Error codes (frozen):

* ``RUNPOD_POD_FAILED`` — snapshot reports a terminal status.
* ``RUNPOD_NO_EXPOSED_TCP`` — RUNNING with ports allocated but no SSH
  endpoint after ``no_exposed_tcp_grace_s``. (Pre-existing — community
  cloud machines that don't support exposed TCP. Specific symptom that
  doesn't recover on the same pod, so we recreate immediately.)
* ``RUNPOD_POD_TIMEOUT`` — total deadline exceeded. This is the catch-
  all for "pod didn't come up in time" — including the legacy "RUNNING
  with port_count == 0 for the whole window" case. The retry policy
  on the provider side recreates the pod when this fires.
* Pass-through query errors propagate unchanged (e.g.
  ``RUNPOD_POD_DATA_MISSING``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from src.utils.cancellation import sleep_cancellable
from src.providers.runpod.lifecycle.policy import (
    TRAINING_PROFILE,
    WaitPolicy,
)
from src.providers.runpod.lifecycle.tcp_probe import default_tcp_probe
from src.providers.runpod.models import PodSnapshot
from src.utils.logger import logger
from src.utils.result import Err, Ok, ProviderError, Result

#: Codes from the underlying ``query_pod_snapshot`` that the waiter
#: treats as terminal (no point retrying — abort fast). Anything else
#: is treated as transient and the loop continues.
_TERMINAL_QUERY_CODES: frozenset[str] = frozenset(
    {
        "RUNPOD_POD_DATA_MISSING",
    }
)


class PodQuery(Protocol):
    """The single read-only method the waiter needs.

    Both ``RunPodTrainingPodControl`` and ``RunPodInferencePodControl``
    already expose this method (the inference one was added in the
    previous commit for symmetry).
    """

    def query_pod_snapshot(self, pod_id: str) -> Result[PodSnapshot, ProviderError]: ...


#: Log callback signature: ``(level, message)``.
LogFn = Callable[[str, str], None]

#: TCP-probe signature: ``(host, port, timeout) -> reachable``.
TcpProbeFn = Callable[[str, int, float], bool]


def _stdlib_log_adapter(level: str, message: str) -> None:
    """Default log callback: forwards to the project logger."""
    fn = getattr(logger, level, None) or logger.info
    fn(message)


@dataclass
class PodSshWaiter:
    """Poll ``query_pod_snapshot`` until the pod is SSH-ready, terminal,
    or stuck.

    The waiter is single-use per ``wait()`` call — internal state is
    local to the method, no carryover between invocations.
    """

    query: PodQuery
    policy: WaitPolicy = field(default_factory=lambda: TRAINING_PROFILE)
    log: LogFn = _stdlib_log_adapter

    #: Monotonic clock — picked over ``time.time()`` so an NTP step
    #: doesn't shorten or extend the deadline.
    clock: Callable[[], float] = time.monotonic

    #: Cancel-aware sleep. Raises :class:`PipelineCancelled` if the
    #: cancel event is set during the wait — the exception propagates
    #: out of ``wait()`` for the provider's cleanup hook to catch.
    sleep: Callable[[float], None] = sleep_cancellable

    #: Injectable TCP probe. Production uses ``default_tcp_probe``;
    #: tests inject deterministic fakes.
    tcp_probe: TcpProbeFn = default_tcp_probe

    def wait(self, pod_id: str) -> Result[PodSnapshot, ProviderError]:
        """Block until the pod is SSH-ready or fails / times out.

        Returns ``Ok(snapshot)`` when the pod reaches RUNNING with an
        SSH endpoint AND (when enabled) the SSH port accepts a TCP
        connection. Returns ``Err(...)`` for terminal pod state, RunPod
        platform stuck-states, or total-timeout.

        Re-raises :class:`PipelineCancelled` from the cancel-aware
        sleep on Ctrl+C — the provider catches it on its own boundary.
        """
        deadline = self.clock() + self.policy.total_timeout_s
        ports_without_ssh_since: float | None = None
        last_preview = ""
        last_log_at = 0.0

        while True:
            now = self.clock()
            if now >= deadline:
                return self._timeout_err(pod_id)

            query_result = self.query.query_pod_snapshot(pod_id)
            if query_result.is_failure():
                err = query_result.unwrap_err()
                if err.code in _TERMINAL_QUERY_CODES:
                    return Err(err)
                self.log("warning", f"[POD_WAIT] query failed (will retry): {err.message}")
                self.sleep(self.policy.poll_interval_s)
                continue

            snapshot = query_result.unwrap()

            if snapshot.is_terminal:
                return self._terminal_err(pod_id, snapshot)

            tcp_ok = None
            if snapshot.is_ready:
                if not self.policy.tcp_probe_enabled:
                    return Ok(snapshot)
                ssh = snapshot.ssh_endpoint
                # Defensive: ``is_ready`` already guarantees ssh is set.
                assert ssh is not None
                if self.tcp_probe(ssh.host, ssh.port, self.policy.tcp_probe_timeout_s):
                    return Ok(snapshot)
                tcp_ok = False

            # Early-bailout for the one state we know doesn't self-heal:
            # ports allocated but SSH endpoint never shows up. Empirically
            # caused by community-cloud nodes that don't support exposed
            # TCP — keeping this matches pre-refactor behaviour.
            #
            # Note: ``RUNNING with port_count == 0`` does NOT have an
            # early bailout. The platform sometimes takes the full timeout
            # window to allocate ports; cutting short here forced a retry
            # half-way through what would otherwise have been a successful
            # boot. Let ``RUNPOD_POD_TIMEOUT`` handle the genuinely-stuck
            # case, and let the provider retry on that.
            if snapshot.status == "RUNNING" and snapshot.port_count > 0 and snapshot.ssh_endpoint is None:
                if ports_without_ssh_since is None:
                    ports_without_ssh_since = now
                elif now - ports_without_ssh_since >= self.policy.no_exposed_tcp_grace_s:
                    return self._no_exposed_tcp_err(pod_id, snapshot)
            else:
                ports_without_ssh_since = None

            # Status logging ----------------------------------------------
            elapsed = int(now - (deadline - self.policy.total_timeout_s))
            preview = self._preview(snapshot, tcp_ok)
            if preview != last_preview or now - last_log_at >= self.policy.repeat_log_interval_s:
                self.log(
                    "info",
                    f"[POD_WAIT] {preview} (elapsed {elapsed}s/{self.policy.total_timeout_s}s)",
                )
                last_preview = preview
                last_log_at = now

            self.sleep(self.policy.poll_interval_s)

    # ------------------------------------------------------------------
    # Error builders — kept as small private helpers so the loop body
    # stays readable.
    # ------------------------------------------------------------------

    def _timeout_err(self, pod_id: str) -> Result[PodSnapshot, ProviderError]:
        return Err(
            ProviderError(
                message=f"Timeout waiting for pod to be ready ({self.policy.total_timeout_s}s)",
                code="RUNPOD_POD_TIMEOUT",
                details={"pod_id": pod_id, "timeout": self.policy.total_timeout_s},
            )
        )

    @staticmethod
    def _terminal_err(pod_id: str, snapshot: PodSnapshot) -> Result[PodSnapshot, ProviderError]:
        return Err(
            ProviderError(
                message=f"Pod entered failed state: {snapshot.status}",
                code="RUNPOD_POD_FAILED",
                details={"pod_id": pod_id, "status": snapshot.status},
            )
        )

    def _no_exposed_tcp_err(self, pod_id: str, snapshot: PodSnapshot) -> Result[PodSnapshot, ProviderError]:
        return Err(
            ProviderError(
                message=(
                    f"Pod has {snapshot.port_count} port(s) but no SSH over exposed TCP "
                    f"after {self.policy.no_exposed_tcp_grace_s}s. This machine likely "
                    "doesn't support exposed TCP ports (community cloud limitation). "
                    "Pod will be recreated."
                ),
                code="RUNPOD_NO_EXPOSED_TCP",
                details={"pod_id": pod_id, "port_count": snapshot.port_count},
            )
        )

    @staticmethod
    def _preview(snapshot: PodSnapshot, tcp_ok: bool | None) -> str:
        ssh = snapshot.ssh_endpoint
        host = ssh.host if ssh else "∅"
        port = ssh.port if ssh else "∅"
        tcp_label = "OK" if tcp_ok is True else ("NO" if tcp_ok is False else "∅")
        return (
            f"status={snapshot.status or '∅'} ip={host} ssh_port={port} " f"tcp={tcp_label} ports={snapshot.port_count}"
        )


__all__ = [
    "PodQuery",
    "PodSshWaiter",
]
