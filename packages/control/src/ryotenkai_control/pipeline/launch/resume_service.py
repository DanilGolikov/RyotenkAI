"""Phase 14.C — provider-agnostic resume orchestrator.

Single source of truth for the wake-pod flow that previously
duplicated between :func:`src.api.services.launch_service.resume_pod_for_run`
(REST surface, ~99 LOC) and
:func:`src.cli.commands.run._resume_pod_if_needed` (CLI surface,
~108 LOC). The body of the resume logic was byte-identical; only
the output shape differed (typed `ResumePodResponse` vs streamed
`typer.echo` lines).

After 14.C both surfaces are thin adapters:

* REST: 5-line wrapper translating :class:`ResumeOutcome` → existing
  ``ResumePodResponse`` shape (wire-shape preserved for Web UI).
* CLI: ~30-line adapter feeding ``typer.echo`` from a progress
  callback and converting non-OK outcomes into ``die()`` calls
  with operator-friendly hints.

Design notes (per plan § 14.C):

* **Sync facade with internal `asyncio.run`** — both consumers are
  sync. The async island is contained inside
  :func:`resume_pod_with_retry`.
* **Provider gating via** :class:`ITerminalActionProvider` —
  no string-comparison; type-system enforces that single-node (which
  doesn't conform) is automatically skipped.
* **Optional progress callback** — REST passes ``None`` (silent);
  CLI passes a typer.echo bridge.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ryotenkai_shared.constants import PROVIDER_RUNPOD, PROVIDER_SINGLE_NODE
from ryotenkai_control.pipeline.launch.pod_availability import (
    PodAvailability,
    PodAvailabilityProbe,
    load_pod_metadata_for_run,
    resume_pod_with_retry,
)
from ryotenkai_providers.training.interfaces import ITerminalActionProvider

if TYPE_CHECKING:
    from ryotenkai_control.pipeline.state.models import PodMetadata

__all__ = [
    "LaunchResumeService",
    "ProgressCallback",
    "ResumeOutcome",
    "ResumeProgress",
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResumeProgress:
    """Single progress event emitted by :meth:`LaunchResumeService.resume`.

    ``kind`` discriminator — JSON-friendly strings for future
    consumers (e.g. Web UI live-progress streaming):

      * ``"probing"`` — about to call probe.
      * ``"verdict"`` — probe returned; ``message`` describes status.
      * ``"resuming"`` — about to call ``resume_pod_with_retry``.
      * ``"resumed"`` — pod is reachable; ``message`` carries timing.
      * ``"skipped"`` — provider doesn't support resume, missing
        creds, or no metadata to act on.

    ``detail`` — kind-specific structured payload (e.g.
    ``{"pod_id": "...", "provider": "runpod"}``). CLI ignores it,
    only displays ``message``.
    """

    kind: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResumeOutcome:
    """Terminal state of a resume attempt.

    ``availability`` is a :class:`PodAvailability` ``.value`` string
    (``running`` / ``sleeping_resumable`` / ``sleeping_resume_failed``
    / ``gone`` / ``probe_failed``) or the special string
    ``"skipped"`` for cases where no resume action was attempted
    (legacy run, missing creds, non-lifecycle provider).

    ``ok=True`` means the pipeline can continue with the run.
    ``ok=False`` means the caller MUST surface ``message`` to the
    user (CLI dies, REST returns non-ok response).

    ``elapsed_seconds`` and ``attempts_made`` are populated only
    when a resume call was actually attempted; otherwise ``None``
    and ``0`` respectively.
    """

    availability: str
    ok: bool
    message: str
    elapsed_seconds: float | None = None
    attempts_made: int = 0
    capacity_exhausted: bool = False


ProgressCallback = Callable[[ResumeProgress], None]


# ---------------------------------------------------------------------------
# Provider resolver
# ---------------------------------------------------------------------------


def _default_resolve_lifecycle_provider(
    provider_name: str,
) -> ITerminalActionProvider | None:
    """Phase 14.C — resolve :class:`PodMetadata.provider` to a live
    :class:`ITerminalActionProvider` instance, or ``None`` when:

    * Provider name is unknown to the registry.
    * Provider doesn't conform to :class:`ITerminalActionProvider`
      (e.g. single-node — has no in-pod resume mechanism).
    * Required env (e.g. ``RUNPOD_API_KEY``) is missing —
      ``LaunchResumeService.resume`` surfaces this as a "skipped:
      missing creds" outcome with explicit ``message``.

    Returns ``None`` for all skip cases; the service distinguishes
    them by re-checking env / provider_name to compose the right
    user-facing message.
    """
    # Manifest-driven dispatch: provider declares ``entry_points.resume_factory``
    # in its ``provider.toml``; registry resolves it lazily. Replaces the
    # ``if PROVIDER_RUNPOD: from ... import RunPodProvider`` string-dispatch.
    # ``RUNPOD_API_KEY`` is still pulled from the operator env here — the
    # registry passes it through to ``from_resume_metadata`` as the
    # ``api_key`` kwarg; providers that don't need a cred just ignore it.
    from ryotenkai_providers.registry import get_registry

    api_key = os.environ.get("RUNPOD_API_KEY")
    try:
        return get_registry().create_resume_provider(
            provider_name, api_key=api_key,
        )
    except Exception:
        # Unknown provider / unavailable resume / missing creds — service
        # surfaces the right "skipped" message based on outcome elsewhere.
        return None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class LaunchResumeService:
    """Provider-agnostic resume orchestrator.

    Test seam: ``provider_resolver`` lets tests inject a fake that
    returns a pre-built :class:`ITerminalActionProvider` stub. Default
    uses :func:`_default_resolve_lifecycle_provider`.

    Usage (production):

        svc = LaunchResumeService()
        outcome = svc.resume(run_dir, on_progress=cb)
        if not outcome.ok:
            # render outcome.message to user

    Usage (tests):

        fake = _FakeRunPodProvider()
        svc = LaunchResumeService(
            provider_resolver=lambda name: fake if name == "runpod" else None,
        )
    """

    def __init__(
        self,
        *,
        provider_resolver: Callable[[str], ITerminalActionProvider | None]
        | None = None,
    ) -> None:
        self._resolve_provider = (
            provider_resolver or _default_resolve_lifecycle_provider
        )

    def resume(
        self,
        run_dir: Path,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> ResumeOutcome:
        """Phase 14.C — orchestrate the resume flow.

        Steps:

        1. Load :class:`PodMetadata` from ``run_dir``.
        2. If absent → outcome ``running`` (legacy run, ok=True).
        3. Emit ``probing`` progress.
        4. Resolve provider via ``provider_resolver``.
        5. If provider is None or doesn't conform → outcome
           ``skipped`` (ok=True, pipeline continues).
        6. Call probe → emit ``verdict`` progress.
        7. If RUNNING → outcome ``running`` (ok=True).
        8. If GONE / PROBE_FAILED → outcome ``ok=False`` with
           appropriate availability + message.
        9. If SLEEPING_RESUMABLE → emit ``resuming`` progress, call
           ``resume_pod_with_retry``, emit ``resumed`` progress on
           success or ``ok=False`` outcome on capacity exhaustion.

        ``on_progress`` is ``None``-safe — REST callers omit it; CLI
        callers pass a typer.echo bridge.
        """
        emit = on_progress if on_progress is not None else _noop_progress

        # Step 1-2: Load metadata.
        metadata = load_pod_metadata_for_run(run_dir)
        if metadata is None:
            return ResumeOutcome(
                availability=PodAvailability.RUNNING.value,
                ok=True,
                message=(
                    "No pod metadata recorded for this run (legacy "
                    "attempt). Continue with normal resume flow."
                ),
            )

        # Step 3: Probe announcement.
        emit(
            ResumeProgress(
                kind="probing",
                message=(
                    f"Probing pod {metadata.pod_id} ({metadata.provider})..."
                ),
                detail={
                    "pod_id": metadata.pod_id,
                    "provider": metadata.provider,
                },
            ),
        )

        # Step 4-5: Resolve provider; bail with skipped on failure.
        provider = self._resolve_provider(metadata.provider)
        if provider is None or not isinstance(provider, ITerminalActionProvider):
            return self._build_skipped_outcome(metadata.provider)

        # Step 6: Probe.
        verdict = self._probe(provider, metadata)
        emit(
            ResumeProgress(
                kind="verdict",
                message=(
                    f"Pod status: {verdict.availability.value}"
                    + (f" — {verdict.message}" if verdict.message else "")
                ),
                detail={
                    "availability": verdict.availability.value,
                    "raw_status": verdict.runpod_status,
                },
            ),
        )

        # Step 7: Already running — nothing to do.
        if verdict.availability == PodAvailability.RUNNING:
            return ResumeOutcome(
                availability=PodAvailability.RUNNING.value,
                ok=True,
                message=verdict.message or "Pod is already running",
            )

        # Step 8a: Pod gone.
        if verdict.availability == PodAvailability.GONE:
            return ResumeOutcome(
                availability=verdict.availability.value,
                ok=False,
                message=verdict.message or "Pod has been terminated",
            )

        # Step 8b: Probe failed.
        if verdict.availability == PodAvailability.PROBE_FAILED:
            return ResumeOutcome(
                availability=verdict.availability.value,
                ok=False,
                message=verdict.message or "Pod probe failed",
            )

        # Step 8c: Defensive — unexpected verdict that isn't
        # SLEEPING_RESUMABLE (e.g. SLEEPING_RESUME_FAILED never came
        # back from a probe; that's a resume-attempt outcome).
        if verdict.availability != PodAvailability.SLEEPING_RESUMABLE:
            return ResumeOutcome(
                availability=verdict.availability.value,
                ok=False,
                message=(
                    f"Unexpected pod state: {verdict.availability.value}"
                ),
            )

        # Step 9: Wake.
        return self._do_resume(provider, metadata, emit)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _probe(
        self, provider: ITerminalActionProvider, metadata: "PodMetadata",
    ) -> "Any":
        """Build a probe wired to the provider's transport + run it."""
        # Resolve the underlying API client. For RunPod the provider
        # exposes an ``_api_client.query_pod(pod_id) -> Result``; we
        # adapt to the (pod_id) -> dict shape the probe expects.
        api_client = getattr(provider, "_api_client", None)
        if api_client is None or not hasattr(api_client, "query_pod"):
            # Defensive — production flows always have this; tests
            # using a fake provider stub need to expose
            # ``_api_client.query_pod`` too.
            from ryotenkai_control.pipeline.launch.pod_availability import ProbeResult
            return ProbeResult(
                availability=PodAvailability.PROBE_FAILED,
                pod_id=metadata.pod_id,
                message=(
                    "Provider does not expose a query_pod transport "
                    "for resume probe (defensive — should not occur "
                    "in production)"
                ),
            )

        def _query_pod(pod_id: str) -> dict[str, Any]:
            # Phase A2 Batch 11+12: ``api_client.query_pod`` raises a
            # typed exception on transport failure and returns the raw
            # dict on success — no more Result shape to unwrap.
            return api_client.query_pod(pod_id)

        probe = PodAvailabilityProbe(query_pod=_query_pod)
        return probe.probe(metadata)

    def _do_resume(
        self,
        provider: ITerminalActionProvider,
        metadata: "PodMetadata",
        emit: ProgressCallback,
    ) -> ResumeOutcome:
        """Call ``resume_pod_with_retry`` with the provider's resume
        transport."""
        emit(
            ResumeProgress(
                kind="resuming",
                message="Resuming pod (capacity-aware retry, ≤5min budget)...",
                detail={"pod_id": metadata.pod_id},
            ),
        )

        # Provider's resume() method (Phase A2 Batch 12) raises typed
        # exceptions directly. resume_pod_with_retry expects a
        # callable returning bool/Awaitable[bool] that raises on
        # transport failure — propagate exceptions as RuntimeError so
        # the existing retry helper's capacity-classifier logic still
        # sees a RuntimeError shape.
        async def _resume_call(pod_id: str) -> bool:
            try:
                provider.resume(resource_id=pod_id)
            except Exception as exc:
                raise RuntimeError(str(exc)) from exc
            return True

        # Capacity-error classification via capability Protocol — replaces
        # the ``if metadata.provider == PROVIDER_RUNPOD: import
        # is_capacity_error_message`` string-dispatch. Providers that
        # implement :class:`ICapacityErrorClassifier` expose
        # ``is_capacity_error(message)``; others get ``None`` (no retry
        # classifier) for clean fall-through.
        from ryotenkai_providers.training.interfaces import ICapacityErrorClassifier

        is_capacity_error: Callable[[str], bool] | None = (
            provider.is_capacity_error  # type: ignore[attr-defined]
            if isinstance(provider, ICapacityErrorClassifier)
            else None
        )

        outcome = asyncio.run(
            resume_pod_with_retry(
                metadata.pod_id,
                resume_call=_resume_call,
                is_capacity_error=is_capacity_error,
            ),
        )

        if outcome.ok:
            emit(
                ResumeProgress(
                    kind="resumed",
                    message=(
                        f"Pod resumed in {outcome.elapsed_seconds:.1f}s "
                        f"({outcome.attempts} attempt(s))"
                    ),
                    detail={
                        "elapsed_seconds": outcome.elapsed_seconds,
                        "attempts": outcome.attempts,
                    },
                ),
            )
            return ResumeOutcome(
                availability=PodAvailability.RUNNING.value,
                ok=True,
                message=(
                    f"Pod resumed in {outcome.elapsed_seconds:.1f}s "
                    f"({outcome.attempts} attempt(s))"
                ),
                elapsed_seconds=outcome.elapsed_seconds,
                attempts_made=outcome.attempts,
            )

        return ResumeOutcome(
            availability=(
                PodAvailability.SLEEPING_RESUME_FAILED.value
                if outcome.capacity_exhausted
                else PodAvailability.PROBE_FAILED.value
            ),
            ok=False,
            message=outcome.error_message,
            elapsed_seconds=outcome.elapsed_seconds,
            attempts_made=outcome.attempts,
            capacity_exhausted=outcome.capacity_exhausted,
        )

    @staticmethod
    def _build_skipped_outcome(provider_name: str) -> ResumeOutcome:
        """Compose the right ``message`` for the skipped path based on
        what the registry says about the provider's capabilities.

        Decision tree (capability-flag driven; replaces the legacy
        string-check on ``PROVIDER_RUNPOD`` / ``PROVIDER_SINGLE_NODE``):

        1. Provider unknown ⇒ "no in-pod resume mechanism".
        2. Provider has lifecycle support but resume_factory missing
           required env (e.g. ``RUNPOD_API_KEY`` unset) ⇒ name the env.
        3. Provider has no lifecycle support (single_node) ⇒ explicit
           "no in-pod resume mechanism" message.
        """
        from ryotenkai_providers.registry import get_registry

        registry = get_registry()
        try:
            manifest = registry.get_manifest(provider_name)
        except KeyError:
            return ResumeOutcome(
                availability="skipped",
                ok=True,
                message=(
                    f"Provider {provider_name!r} is not registered; "
                    "continue with normal flow."
                ),
            )
        if manifest.capabilities.supports_lifecycle_actions:
            # Provider supports lifecycle but resolver still returned
            # None — most likely missing required env. Surface the
            # first missing required secret name.
            missing = [
                spec.name
                for spec in manifest.required_env
                if spec.secret and not spec.optional
                and not os.environ.get(spec.name)
            ]
            if missing:
                return ResumeOutcome(
                    availability="skipped",
                    ok=True,
                    message=f"{missing[0]} not in environment",
                )
        # No lifecycle — single_node and equivalents.
        return ResumeOutcome(
            availability="skipped",
            ok=True,
            message=(
                f"Provider {provider_name!r} has no in-pod resume "
                "mechanism; continue with normal flow."
            ),
        )


def _noop_progress(_: ResumeProgress) -> None:
    """Default progress callback — drops events on the floor."""
