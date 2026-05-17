"""Control-side event transport: journal + in-memory bus + dedup + emitter.

This package is the control-plane counterpart to
``ryotenkai_pod.runner.event_bus`` / ``event_journal`` / ``event_emitter``.
It satisfies the :class:`ryotenkai_shared.events.IEventEmitter` Protocol
and provides three orthogonal collaborators that the orchestrator composes
into a single :class:`ControlEventEmitter`:

* :class:`JournalWriter` — append-only JSONL writer with length-prefixed
  framing and batched fsync. SSOT for emitted events.
* :class:`JournalReader` — tail-able reader with torn-write truncation,
  per-source offset reconstruction, and offset-range replay.
* :class:`InMemoryBus` — bounded ring buffer + per-consumer cursors
  (Ray-style ``MultiConsumerEventBuffer``) for live SSE/WS fan-out.
* :class:`EventDedup` — ``(run_id, source, offset)`` dedup set with
  journal-tail reconstruction and TTL eviction; implements the R-13
  contract for :meth:`IEventEmitter.emit_remote`.
* :class:`ControlEventEmitter` — composes journal/bus/dedup into the
  Protocol surface.
* :func:`slice_journal` — pure offset-range slice helper used by the
  Phase 6 HTTP replay endpoint.
"""

from __future__ import annotations

from ryotenkai_control.events.dedup import EventDedup
from ryotenkai_control.events.dedup_sweeper import DedupTTLSweeper
from ryotenkai_control.events.emitter import ControlEventEmitter
from ryotenkai_control.events.in_memory_bus import InMemoryBus
from ryotenkai_control.events.journal_reader import JournalReader
from ryotenkai_control.events.journal_writer import JournalWriter
from ryotenkai_control.events.metrics import (
    EventSubsystemMetrics,
    collect_metrics,
    collect_metrics_for_emitter,
)
from ryotenkai_control.events.mlflow_finalizer import MlflowFinalizer
from ryotenkai_control.events.registry import EventEmitterRegistry
from ryotenkai_control.events.replay import slice_journal

__all__ = [
    "ControlEventEmitter",
    "DedupTTLSweeper",
    "EventDedup",
    "EventEmitterRegistry",
    "EventSubsystemMetrics",
    "InMemoryBus",
    "JournalReader",
    "JournalWriter",
    "MlflowFinalizer",
    "collect_metrics",
    "collect_metrics_for_emitter",
    "slice_journal",
]
