from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query

from ryotenkai_control.api.dependencies import get_runs_dir
from ryotenkai_control.api.schemas.health import HealthStatus
from ryotenkai_control.events import (
    EventEmitterRegistry,
    collect_metrics_for_emitter,
)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus)
def health(runs_dir: Path = Depends(get_runs_dir)) -> HealthStatus:
    readable = runs_dir.exists() and runs_dir.is_dir()
    return HealthStatus(
        status="ok" if readable else "degraded",
        runs_dir=str(runs_dir),
        runs_dir_readable=readable,
    )


# ---------------------------------------------------------------------------
# Phase 8 — Event-subsystem health
# ---------------------------------------------------------------------------
#
# Reads :class:`EventEmitterRegistry.instance()` and snapshots every
# registered emitter via :func:`collect_metrics_for_emitter`. The
# response is a plain dict (no Pydantic model) — the field set is
# wide and expected to evolve as new counters are added; locking the
# wire format into a Pydantic model would either bloat the schema
# every release or force `extra="allow"`. The current shape is
# documented in ``docs/plans/ethereal-tumbling-patterson.md`` (Phase 8).
#
# ``status`` semantics:
#
# * ``"no_active_runs"`` — registry is empty (no orchestrator-side
#   emitters are currently held). Distinct from ``"healthy"`` so an
#   operator can tell "nothing wrong" apart from "nothing happening".
# * ``"degraded"``       — at least one per-run snapshot has a
#   non-zero failure counter (emit failures, drops, fsync failures,
#   offset collisions). Conservative — single failure flips the bit
#   so the operator decides whether to dig in.
# * ``"healthy"``        — registry has at least one emitter and all
#   per-run snapshots are clean.


@router.get("/health/events")
def events_health(
    run_id: str | None = Query(
        default=None,
        description=(
            "Restrict the snapshot to a single run. When omitted "
            "the response aggregates every emitter in the "
            "EventEmitterRegistry."
        ),
    ),
) -> dict[str, Any]:
    """Observability snapshot of the event subsystem.

    Returns a JSON dict with per-run counters + an aggregate
    ``health_indicators`` summary. See module docstring for ``status``
    semantics.
    """
    registry = EventEmitterRegistry.instance()

    if run_id is not None:
        emitter = registry.get(run_id)
        emitters: dict[str, Any] = {run_id: emitter} if emitter is not None else {}
    else:
        # Snapshot the registry then resolve emitters — keeps the
        # response self-consistent even if a run deregisters between
        # the listing and the per-emitter read.
        emitters = {rid: registry.get(rid) for rid in registry.active_run_ids()}
        emitters = {rid: em for rid, em in emitters.items() if em is not None}

    per_run: dict[str, dict[str, Any]] = {}
    any_emit_failures = False
    any_drops = False
    any_fsync_failures = False
    any_offset_collisions = False
    any_write_failures = False

    for rid, emitter in emitters.items():
        snapshot = collect_metrics_for_emitter(emitter)
        per_run[rid] = snapshot.to_dict()
        # Aggregate health indicators across all runs — a single bad
        # run flips the global ``status`` to ``degraded``.
        if snapshot.emitter_events_emit_failed_total:
            any_emit_failures = True
        if (
            snapshot.bus_dropped_total > 0
            or any(v > 0 for v in snapshot.bus_dropped_per_consumer.values())
            or snapshot.emitter_events_remote_dropped_total
        ):
            any_drops = True
        if snapshot.journal_fsync_failed_total > 0:
            any_fsync_failures = True
        if snapshot.journal_write_failed_total > 0:
            any_write_failures = True
        if snapshot.emitter_offset_collisions_detected_total > 0:
            any_offset_collisions = True

    if not emitters:
        status = "no_active_runs"
    elif (
        any_emit_failures
        or any_drops
        or any_fsync_failures
        or any_write_failures
        or any_offset_collisions
    ):
        status = "degraded"
    else:
        status = "healthy"

    return {
        "status": status,
        "active_runs": list(emitters.keys()),
        "per_run": per_run,
        "health_indicators": {
            "any_emit_failures": any_emit_failures,
            "any_drops": any_drops,
            "any_fsync_failures": any_fsync_failures,
            "any_write_failures": any_write_failures,
            "any_offset_collisions": any_offset_collisions,
        },
    }
