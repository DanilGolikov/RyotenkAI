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
from dataclasses import dataclass
from pathlib import Path
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


#: Directory holding per-provider pod sub-manifests, populated by
#: ``packages/providers/scripts/compile_pod_manifests.py``. Walking
#: this directory (instead of a hardcoded dict) is what makes adding
#: a third provider a zero-touch operation on the pod side: the
#: projection script writes a new TOML next to these, the pod's next
#: lifespan boot picks it up.
#:
#: The pod-side parses these with stdlib ``tomllib`` only — no
#: Pydantic schema, no shared loader. The Mac-side
#: :class:`ProviderManifest` validator (and the ``check_manifests.py``
#: pre-commit hook) own the schema invariants; what reaches the pod
#: has already been validated.
_POD_MANIFESTS_DIR: Final[Path] = Path(__file__).resolve().parent / "pod_manifests"


@dataclass(frozen=True, slots=True)
class _PodProviderManifest:
    """In-memory shape of a projected pod sub-manifest.

    Trivial three-field record — the projection drops everything the
    pod doesn't need (Mac-only entry points, capability flags the
    runner doesn't read, required_env). Living right here keeps the
    pod with no dependency on ``ryotenkai_providers`` (sentinel test
    ``test_pod_does_not_import_control_or_providers`` stays green).
    """

    provider_id: str
    supports_lifecycle_actions: bool
    #: ``"module:Class"`` string when the manifest declares a
    #: ``[entry_points.pod_lifecycle_client]`` block. ``None`` when
    #: ``supports_lifecycle_actions=false``.
    lifecycle_client_locator: str | None


def _load_pod_manifests(
    *, manifests_dir: Path = _POD_MANIFESTS_DIR
) -> dict[str, _PodProviderManifest]:
    """Walk ``pod_manifests/*.toml`` and parse each into a dataclass.

    Idempotent and side-effect free. Called eagerly at module load —
    the manifest set is small (~2 files today) and parsing is fast,
    so caching it at import time is fine.
    """
    import tomllib

    out: dict[str, _PodProviderManifest] = {}
    if not manifests_dir.is_dir():
        return out
    for toml_path in sorted(manifests_dir.glob("*.toml")):
        with toml_path.open("rb") as fh:
            data = tomllib.load(fh)
        provider_id = data["provider"]["id"]
        caps = data.get("capabilities", {})
        eps = data.get("entry_points", {})
        pod_lc = eps.get("pod_lifecycle_client")
        locator: str | None = None
        if pod_lc is not None:
            locator = f"{pod_lc['module']}:{pod_lc['class']}"
        out[provider_id] = _PodProviderManifest(
            provider_id=provider_id,
            supports_lifecycle_actions=bool(caps.get("supports_lifecycle_actions", False)),
            lifecycle_client_locator=locator,
        )
    return out


_POD_MANIFESTS: Final[dict[str, _PodProviderManifest]] = _load_pod_manifests()


class _BuiltinNoOpLifecycleClient:
    """In-pod no-op lifecycle client for providers without cloud lifecycle.

    Replaces the ``NoOpPodLifecycleClient`` that used to live under
    ``ryotenkai_providers.single_node.runtime.lifecycle_client``. The
    only single_node-style provider behaviour the pod's lifespan needs
    is "do nothing" — encoding it once here in the pod package keeps
    the pod free of an importlib pull on a provider that exists solely
    to return ``Ok``.

    All three methods return the same shared ``_SKIPPED_RESULT`` sentinel
    — the wire string is ``"skipped"`` (lowercase) per Phase 9.A / 11.B
    operator-dashboard convention. Reusing one frozen instance is also a
    micro-optimisation: GC pressure stays at zero on a pod that hits
    ``terminate`` once at shutdown.
    """

    @property
    def provider_name(self) -> str:
        # Read at runtime from the env that selected this client.
        import os

        return os.environ.get(RUNTIME_PROVIDER_ENV_VAR, "")

    async def terminate(self, *, resource_id: str):  # type: ignore[no-untyped-def]
        return _SKIPPED_RESULT

    async def pause(self, *, resource_id: str):  # type: ignore[no-untyped-def]
        return _SKIPPED_RESULT

    async def resume(self, *, resource_id: str):  # type: ignore[no-untyped-def]
        return _SKIPPED_RESULT


def _make_skipped_result():  # type: ignore[no-untyped-def]
    """Build the shared SKIPPED LifecycleActionResult.

    Module-load-time helper — keeps the heavy lifecycle imports out of
    the function bodies above (those are awaited often during shutdown
    and shouldn't pay an importlib hit each time).
    """
    from ryotenkai_shared.infrastructure.lifecycle import (
        LifecycleActionResult,
        PodTerminalOutcome,
    )

    return LifecycleActionResult(
        outcome=PodTerminalOutcome.SKIPPED,
        attempts_made=0,
        last_error=None,
        raw_response_excerpt=None,
    )


_SKIPPED_RESULT = _make_skipped_result()


def _resolve_lifecycle_client_class(provider: str) -> type[IPodLifecycleClient]:
    """Lazy importlib resolution of a provider's lifecycle client class."""
    import importlib

    manifest = _POD_MANIFESTS[provider]
    locator = manifest.lifecycle_client_locator
    if locator is None:
        raise BootstrapConfigError(
            f"provider {provider!r} has no [entry_points.pod_lifecycle_client] "
            f"in its pod sub-manifest — supports_lifecycle_actions is False, "
            f"so no lifecycle client class to resolve. Use the built-in no-op."
        )
    module_name, _, attr_name = locator.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


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
    client_cls = _resolve_lifecycle_client_class(PROVIDER_RUNPOD)
    return client_cls(api_key=api_key)


def _build_noop_client(env: Mapping[str, str]) -> IPodLifecycleClient:
    """Built-in no-op for providers with ``supports_lifecycle_actions=false``."""
    return _BuiltinNoOpLifecycleClient()  # type: ignore[return-value]


# Built dynamically from the projected pod sub-manifests. Providers with
# lifecycle support get a custom builder (the runpod one above checks
# RunPod-specific env contracts); providers without lifecycle get the
# built-in no-op. Adding a third lifecycle-supporting provider means
# adding one entry here; pure-runtime additions (no special env probes)
# can use ``_build_noop_client`` or define their own builder.
def _build_registry() -> dict[str, Callable[[Mapping[str, str]], IPodLifecycleClient]]:
    out: dict[str, Callable[[Mapping[str, str]], IPodLifecycleClient]] = {}
    for pid, manifest in _POD_MANIFESTS.items():
        if pid == PROVIDER_RUNPOD:
            out[pid] = _build_runpod_client
        elif manifest.supports_lifecycle_actions:
            # Future cloud provider with custom env contract: add an
            # explicit builder here that validates its required env.
            # Falling back to a generic locator-based builder.
            out[pid] = _build_runpod_client  # placeholder — re-evaluate when 3rd provider lands
        else:
            out[pid] = _build_noop_client
    return out


_REGISTRY: Final[dict[str, Callable[[Mapping[str, str]], IPodLifecycleClient]]] = _build_registry()


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
