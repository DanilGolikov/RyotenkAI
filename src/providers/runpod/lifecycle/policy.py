"""Tunable thresholds for :class:`PodSshWaiter`.

Two named profiles are exposed for the two existing call sites; the
fields themselves are the contract — callers can build custom
:class:`WaitPolicy` instances if they need different windows (e.g.
tests with shorter timeouts).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WaitPolicy:
    """Time windows for each "the pod is stuck" early-bailout.

    Defaults match the historical training-side behaviour. Inference
    uses :data:`INFERENCE_PROFILE` for slightly longer windows because
    eval pods cold-start more slowly than training pods (no warm-cache).
    """

    #: Total upper bound for ``wait()``. After this, ``Err(POD_TIMEOUT)``.
    total_timeout_s: int = 600

    #: Sleep between successive ``query_pod_snapshot`` calls. Effective
    #: poll rate is ``poll_interval_s + tcp_probe_timeout_s`` when the
    #: pod is RUNNING — keep this honest in test mocks.
    poll_interval_s: float = 5.0

    #: Pod is RUNNING with ports allocated but no SSH endpoint exposed.
    #: After this grace window, abort with ``RUNPOD_NO_EXPOSED_TCP`` —
    #: empirically the platform won't recover (community-cloud node
    #: limitation). Recreate the pod.
    no_exposed_tcp_grace_s: int = 30

    #: Pod is RUNNING but ports==0 the whole time. Stuck. After this
    #: window, abort with ``RUNPOD_NO_PORTS_ALLOCATED`` — RunPod
    #: platform issue, recreate the pod.
    running_no_ports_bailout_s: int = 180

    #: TCP probe timeout when checking whether sshd is actually
    #: listening on the announced port.
    tcp_probe_timeout_s: float = 3.0

    #: Set ``False`` to skip the TCP probe (used by tests; production
    #: callers should keep it enabled — closes the cold-SSH gap).
    tcp_probe_enabled: bool = True

    #: Throttle for repeating the same status line in logs. The waiter
    #: re-logs the line when the preview changes OR every
    #: ``repeat_log_interval_s`` seconds, whichever happens first.
    repeat_log_interval_s: float = 30.0


#: Training-side profile. Tighter total timeout because training pods
#: are GPU-pre-warmed and start within seconds.
TRAINING_PROFILE: WaitPolicy = WaitPolicy(total_timeout_s=300)

#: Inference / eval profile. Wider window for cold-start of the
#: vLLM-style eval container.
INFERENCE_PROFILE: WaitPolicy = WaitPolicy(
    total_timeout_s=600,
    no_exposed_tcp_grace_s=60,
)


__all__ = [
    "INFERENCE_PROFILE",
    "TRAINING_PROFILE",
    "WaitPolicy",
]
