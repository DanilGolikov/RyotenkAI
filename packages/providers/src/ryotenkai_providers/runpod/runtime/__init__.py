"""Phase 14.B — RunPod runner-side runtime package.

In-pod control-plane code that runs **inside** the GPU pod's
container, distinct from
:mod:`src.providers.runpod.training` (Mac-side launcher) and
:mod:`src.providers.runpod.inference` (Mac-side inference flow).

Currently only houses
:mod:`src.providers.runpod.runtime.lifecycle_client` — the RunPod
GraphQL impl of
:class:`~src.runner.runtime.lifecycle_client.IPodLifecycleClient`.
Phase 14.E or beyond may add other in-pod-only RunPod modules
here.
"""
