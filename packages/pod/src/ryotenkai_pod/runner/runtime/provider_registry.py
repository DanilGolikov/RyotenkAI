"""Phase 14.B — env-driven :class:`IPodLifecycleClient` resolver.

Single bootstrap seam. The lifespan reads env, asks this module
which client to construct, hands it to
:class:`~src.runner.pod_terminator.PodTerminator`. **No other module
in :mod:`src.runner` reads** :data:`~src.constants.RUNTIME_PROVIDER_ENV_VAR`
**or** ``RUNPOD_*``.

Environment contract (the lifespan receives this from the Mac
launcher via Phase 14.A's
:meth:`~src.providers.training.interfaces.IGPUProvider.required_runtime_env_vars`):

* ``RYOTENKAI_RUNTIME_PROVIDER`` — required. One of
  :data:`~src.constants.PROVIDER_RUNPOD` /
  :data:`~src.constants.PROVIDER_SINGLE_NODE`.
* ``RUNPOD_API_KEY`` — required when provider=``runpod``.
* ``RUNPOD_POD_ID`` — required when provider=``runpod``.
* ``RUNPOD_VOLUME_KIND`` — optional. ``persistent`` (default) or
  ``network``. Anything else falls back to ``persistent`` (warning
  logged by caller).
* ``RUNPOD_KEEP_ON_ERROR`` — optional. Literal ``"true"`` enables
  it; everything else is treated as ``false``.

Bootstrap failure modes (raise :class:`BootstrapConfigError`,
caught by the lifespan, re-raised so uvicorn exits non-zero):

#. ``RYOTENKAI_RUNTIME_PROVIDER`` unset or empty.
#. Value not registered.
#. Value=``runpod`` but ``RUNPOD_API_KEY`` or ``RUNPOD_POD_ID``
   missing.

NO graceful degradation — Phase 14.B § 1.7 documents the rationale:
silent ``SKIPPED`` outcomes 4 hours later when the run terminates
are a known operator-dashboard pain point. We make boot loud.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

from ryotenkai_shared.constants import (
    PROVIDER_RUNPOD,
    PROVIDER_SINGLE_NODE,
    RUNTIME_PROVIDER_ENV_VAR,
)
from ryotenkai_shared.infrastructure.lifecycle import IPodLifecycleClient

__all__ = [
    "BootstrapConfigError",
    "registered_providers",
    "resolve_keep_on_error_from_env",
    "resolve_lifecycle_client_from_env",
    "resolve_resource_id_from_env",
    "resolve_volume_kind_from_env",
]


class BootstrapConfigError(RuntimeError):
    """Raised at lifespan when env config can't satisfy the runner bootstrap.

    Caller (the FastAPI lifespan) MUST re-raise so uvicorn exits
    non-zero — silent fall-through to ``NoOpPodLifecycleClient``
    would defeat the fail-fast contract that Phase 14.B § 1.7
    locked.
    """


def _build_runpod_client(env: Mapping[str, str]) -> IPodLifecycleClient:
    api_key = env.get("RUNPOD_API_KEY")
    pod_id = env.get("RUNPOD_POD_ID")
    if not api_key:
        raise BootstrapConfigError(
            f"{RUNTIME_PROVIDER_ENV_VAR}=runpod requires RUNPOD_API_KEY",
        )
    if not pod_id:
        raise BootstrapConfigError(
            f"{RUNTIME_PROVIDER_ENV_VAR}=runpod requires RUNPOD_POD_ID",
        )
    # Local import: keeps single-node-only deployments from importing
    # httpx at module-load time.
    from ryotenkai_providers.runpod.runtime.lifecycle_client import (
        RunPodPodLifecycleClient,
    )
    return RunPodPodLifecycleClient(api_key=api_key)


def _build_single_node_client(env: Mapping[str, str]) -> IPodLifecycleClient:
    # No env reads — single-node has no creds to validate.
    from ryotenkai_providers.single_node.runtime.lifecycle_client import (
        NoOpPodLifecycleClient,
    )
    return NoOpPodLifecycleClient()


_REGISTRY: Final[dict[str, Callable[[Mapping[str, str]], IPodLifecycleClient]]] = {
    PROVIDER_RUNPOD: _build_runpod_client,
    PROVIDER_SINGLE_NODE: _build_single_node_client,
}


def registered_providers() -> tuple[str, ...]:
    """Sorted tuple of registered provider names.

    Used by tests + the cross-Protocol invariant test (Phase 14.B
    § 8.2 § 12 — ensures every Mac-side provider with
    ``supports_lifecycle_actions=True`` has a runner-side entry
    here).
    """
    return tuple(sorted(_REGISTRY))


def resolve_lifecycle_client_from_env(
    env: Mapping[str, str],
) -> IPodLifecycleClient:
    """Build the right :class:`IPodLifecycleClient` for the current pod.

    Args:
        env: Mapping of env vars (typically ``os.environ`` or a test
            fixture). NOT consulted globally — caller passes in
            exactly the env the runner should bootstrap from.

    Raises:
        BootstrapConfigError: env doesn't satisfy any registered
            provider's preconditions. Message names the failing var
            and lists the known providers — operator-friendly.
    """
    name = env.get(RUNTIME_PROVIDER_ENV_VAR)
    if not name:
        raise BootstrapConfigError(
            f"{RUNTIME_PROVIDER_ENV_VAR} is not set; runner cannot "
            f"select a lifecycle client. Known providers: "
            f"{list(registered_providers())}",
        )
    builder = _REGISTRY.get(name)
    if builder is None:
        raise BootstrapConfigError(
            f"{RUNTIME_PROVIDER_ENV_VAR}={name!r} is not registered. "
            f"Known providers: {list(registered_providers())}",
        )
    return builder(env)


def resolve_volume_kind_from_env(env: Mapping[str, str]) -> str:
    """Read ``RUNPOD_VOLUME_KIND`` from env, clamp invalid values.

    Returns:
        ``"persistent"`` (default) or ``"network"``. Anything else
        falls back to ``"persistent"`` — preserves pre-14.B
        :meth:`~src.runner.pod_terminator.PodTerminator.decide_and_act`
        behaviour bit-for-bit.
    """
    raw = (env.get("RUNPOD_VOLUME_KIND") or "persistent").lower()
    return raw if raw in ("persistent", "network") else "persistent"


def resolve_keep_on_error_from_env(env: Mapping[str, str]) -> bool:
    """Read ``RUNPOD_KEEP_ON_ERROR`` from env.

    Only literal ``"true"`` enables it. Anything else (``"yes"``,
    ``"1"``, ``""``, unset) is :data:`False` — preserves pre-14.B
    behaviour.
    """
    return (env.get("RUNPOD_KEEP_ON_ERROR") or "").lower() == "true"


def resolve_resource_id_from_env(env: Mapping[str, str]) -> str:
    """Provider-static resource id read at lifespan boot.

    For RunPod returns :envvar:`RUNPOD_POD_ID`. For single-node
    returns the empty string (single-node has no resource concept).

    The lifespan calls
    :func:`resolve_lifecycle_client_from_env` BEFORE this helper, so
    by the time we read the pod id we're guaranteed the env passes
    the validation gate (
    :func:`resolve_lifecycle_client_from_env` would have raised
    otherwise).
    """
    name = env.get(RUNTIME_PROVIDER_ENV_VAR, "")
    if name == PROVIDER_RUNPOD:
        return env.get("RUNPOD_POD_ID", "")
    return ""
