"""Canonical pod-readiness primitives for RunPod-backed providers.

Public surface:

* :class:`PodSshWaiter` — the single ``wait(pod_id) -> Result`` primitive.
  Replaces three near-duplicate wait loops (training lifecycle manager,
  inference eval session, chat-script artifacts).
* :class:`PodQuery` — the Protocol both ``RunPodTrainingPodControl`` and
  ``RunPodInferencePodControl`` already satisfy (``query_pod_snapshot``).
* :class:`WaitPolicy` — tunable thresholds. ``TRAINING_PROFILE`` and
  ``INFERENCE_PROFILE`` for the two existing call sites.
* :func:`default_tcp_probe` — small ``socket`` helper that confirms the
  SSH port is actually accepting connections (closes the cold-SSH gap
  where RunPod reports RUNNING before sshd is up).

Module placement: ``providers/runpod/lifecycle/`` rather than a
top-level ``lifecycle/`` because everything in here is RunPod-specific
(SDK shape, error-code vocabulary, platform-issue early-bailouts). When
a second cloud provider arrives, the abstractions cross-cut at the
``ResolvedProject``/launcher layer, not here.
"""

from __future__ import annotations

from src.providers.runpod.lifecycle.pod_ssh_waiter import (
    PodQuery,
    PodSshWaiter,
)
from src.providers.runpod.lifecycle.policy import (
    INFERENCE_PROFILE,
    TRAINING_PROFILE,
    WaitPolicy,
)
from src.providers.runpod.lifecycle.tcp_probe import default_tcp_probe

__all__ = [
    "INFERENCE_PROFILE",
    "TRAINING_PROFILE",
    "PodQuery",
    "PodSshWaiter",
    "WaitPolicy",
    "default_tcp_probe",
]
