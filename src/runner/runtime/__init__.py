"""Phase 14.B — runner-side runtime bootstrap package.

Houses the abstractions the FastAPI lifespan needs to translate
``RYOTENKAI_RUNTIME_PROVIDER`` (Phase 14.A) into a working
in-pod control plane:

* :mod:`src.runner.runtime.lifecycle_client` — the Protocol the
  :class:`~src.runner.pod_terminator.PodTerminator` dispatches against.
* :mod:`src.runner.runtime.provider_registry` — env-driven resolver
  that picks the right :class:`IPodLifecycleClient` impl + extracts
  the lifespan-static config (volume kind, keep-on-error, resource id).

Provider impls live next to the rest of each provider's code:

* ``src/providers/runpod/runtime/lifecycle_client.py``
* ``src/providers/single_node/runtime/lifecycle_client.py``

This package contains NO provider-specific logic — it only knows
the Protocol shape and how to look it up by name.
"""
