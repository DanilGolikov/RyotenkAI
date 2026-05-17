"""Pod-event forwarding extracted from :mod:`training_monitor`.

After Group 1 (Pod→Control event forwarding, 2026-05-16) the
:class:`TrainingMonitor` grew several methods whose responsibility is
**purely** to interpret WS-received pod events and forward them onto
the control-side typed event journal. This module collects that logic
behind a small, focused class so the monitor can shrink back under the
1500-line architectural guardrail.

Scope of this module (pure relocation — no behavioural changes):

- :meth:`dispatch_event` (was ``TrainingMonitor._dispatch_event``):
  classify each WS-received event dict, fire the monitor's milestone
  log lines (first-event, trainer_spawned), forward typed envelopes
  through :meth:`forward_pod_envelope`, route ``health_snapshot`` to
  :meth:`maybe_log_status`, and translate ``trainer_exited`` /
  terminal FSM state events into a terminal outcome dict via callbacks
  back into the monitor.
- :meth:`is_new_envelope_shape`: heuristic for new vs legacy wire shape.
- :meth:`forward_pod_envelope`: sanitise, validate, ``emit_remote``.
- :meth:`maybe_log_status`: rate-limited ALIVE-line logging.
- :meth:`replay_then_resume_or_fallback`: HTTP ``/events/replay``
  paginated fallback when WS subscribe raises
  :class:`ReplayTruncatedError`.

The forwarder reads + mutates a small slice of monitor state
(``_last_event_at``, ``_last_status_log_time``, ``_first_event_logged``,
``_trainer_started_logged``, ``_last_offset``, ``_training_start_time``)
via a ``MonitorState`` Protocol. Trainer-exit / reconciliation logic
remains on the monitor — the forwarder calls back via the
``handle_trainer_exited`` / ``terminal_from_state`` /
``fallback_to_status`` callables passed at construction.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import ValidationError

from ryotenkai_shared.events import EVENT_ADAPTER, BaseEvent
from ryotenkai_shared.utils.clients.job_client import (
    JobClientError,
    JobNotFoundError,
)
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ryotenkai_shared.events import IEventEmitter
    from ryotenkai_shared.utils.clients.job_client import JobClient


# Rate-limit window for the ``[MONITOR] running ...`` status line.
# Mirrors :data:`training_monitor.TRAINING_MONITOR_LOG_STATUS_INTERVAL`
# so the forwarder is self-contained for unit testing without an import
# cycle.
_LOG_STATUS_INTERVAL_DEFAULT = 15


class MonitorState(Protocol):
    """Mutable slice of :class:`TrainingMonitor` state the forwarder owns.

    The Protocol exists for typing-only — at runtime the forwarder is
    handed the :class:`TrainingMonitor` instance directly. Restricting
    the visible attributes through this Protocol keeps the boundary
    explicit so it's obvious *which* monitor state the forwarder touches.
    """

    _first_event_logged: bool
    _trainer_started_logged: bool
    _last_event_at: datetime
    _last_status_log_time: float
    _last_offset: int
    _training_start_time: float


class PodEventForwarder:
    """Forwards pod-side WS events into control's typed event journal.

    Responsibilities (extracted from :class:`TrainingMonitor`):

    * ``dispatch_event`` — classify each WS event dict, route to inline
      milestone / status logging (legacy back-compat) **and** emit_remote
      (typed forwarding) for typed envelopes.
    * ``is_new_envelope_shape`` — heuristic for new vs legacy wire shape.
    * ``forward_pod_envelope`` — sanitise, validate, ``emit_remote``.
    * ``maybe_log_status`` — rate-limited operator status line.
    * ``replay_then_resume_or_fallback`` — HTTP ``/events/replay``
      paginated fallback when the WS replay buffer rolls past us.

    Behaviour parity: this is a **pure relocation** from
    :class:`TrainingMonitor`. Any change in observed behaviour is a
    bug, not a feature.
    """

    def __init__(
        self,
        *,
        state: MonitorState,
        emitter: IEventEmitter | None,
        handle_trainer_exited: Callable[[dict[str, Any]], dict[str, Any]],
        terminal_from_state: Callable[[str, dict[str, Any]], dict[str, Any]],
        fallback_to_status: Callable[[JobClient, str], Awaitable[dict[str, Any]]],
        terminal_states: frozenset[str],
        log_status_interval: int = _LOG_STATUS_INTERVAL_DEFAULT,
    ) -> None:
        self._state = state
        self._emitter = emitter
        self._handle_trainer_exited = handle_trainer_exited
        self._terminal_from_state = terminal_from_state
        self._fallback_to_status = fallback_to_status
        self._terminal_states = terminal_states
        self._log_status_interval = log_status_interval

    # --- emitter wiring -------------------------------------------------

    def set_emitter(self, emitter: IEventEmitter | None) -> None:
        """Inject an emitter after construction (lazy wiring).

        Mirrors :meth:`TrainingMonitor.set_emitter` so callers that wire
        the emitter post-construction can keep the forwarder in sync.
        """
        self._emitter = emitter

    # --- dispatch -------------------------------------------------------

    def dispatch_event(
        self,
        event: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Fire callbacks; return a terminal outcome dict or ``None``
        to keep listening.

        Recognised event kinds (everything else is logged at debug
        and ignored):

        - first event (any kind) → "[MONITOR] WS stream open"
        - ``trainer_spawned`` → "[MONITOR] Trainer process started"
        - ``health_snapshot`` → rate-limited ALIVE status line
        - ``trainer_exited`` → delegate to monitor's
          ``_handle_trainer_exited`` (terminal outcome / raise)
        - bare FSM-state terminal → delegate to monitor's
          ``_terminal_from_state``
        - other kinds → log only at debug

        Note: ``trainer_log`` events were removed in the data-plane
        refactor — trainer stdout/stderr now lands in
        ``trainer.stdio.log`` on the pod (written by the Supervisor's
        pump) and is pulled to Mac via LogManager scp. The Web UI's
        LogDock reads ``trainer.stdio.log`` directly. The bus / WS
        stream carries only control + telemetry events.
        """
        kind = event.get("kind") or ""
        payload = event.get("payload") or {}

        if not self._state._first_event_logged:
            self._state._first_event_logged = True
            logger.info("[MONITOR] WS event stream open — runner is reachable")

        if kind == "trainer_spawned" and not self._state._trainer_started_logged:
            self._state._trainer_started_logged = True
            pid = payload.get("pid")
            logger.info(
                "[MONITOR] Trainer process started%s",
                f" (pid={pid})" if pid else "",
            )

        # Phase 10 follow-up: forward typed pod envelopes to the
        # control-side emitter so they land in ``events.jsonl`` (and the
        # in-memory bus → SSE). Done BEFORE the kind-specific dispatch
        # below so the journal sees every pod event the monitor saw,
        # regardless of whether the inline callback above produced a
        # terminal outcome.
        self.forward_pod_envelope(event)

        if kind == "health_snapshot":
            self.maybe_log_status(payload)
            # Refresh the "last event from pod" anchor so the timeout
            # envelope (if we end up firing one) carries the most recent
            # observation. Per-event resource metrics will be folded
            # back into typed events in Phase 5; the rate-limited
            # ``[MONITOR] running ...`` line above remains the
            # operator-facing surface for now.
            self._state._last_event_at = datetime.now(UTC)
            return None

        if kind == "trainer_exited":
            return self._handle_trainer_exited(payload)

        # Catch-all for FSM-state transitions emitted alongside
        # trainer_exited. The runner publishes a structured event
        # for the transition itself; we use it as a backstop so a
        # missed ``trainer_exited`` (e.g. supervisor crash) still
        # surfaces a terminal state to the orchestrator.
        state = payload.get("state") if isinstance(payload, dict) else None
        if isinstance(state, str) and state in self._terminal_states:
            return self._terminal_from_state(state, payload)

        logger.debug(f"[MONITOR] event kind={kind!r} (no callback)")
        return None

    # --- pod-event forwarding (Phase 10 follow-up) ----------------------

    def is_new_envelope_shape(self, event: dict[str, Any]) -> bool:
        """Heuristic: does ``event`` carry the new typed-envelope shape?

        Wire shape (Phase 2 ``envelope_to_wire``) always includes
        ``kind_dotted`` for known typed events PLUS the canonical
        envelope fields ``event_id`` / ``schema_version`` / ``time``.
        The legacy pre-Phase-2 ``Event.to_dict()`` shape only carries
        ``{offset, timestamp, kind, payload}`` — no ``kind_dotted``,
        no ``event_id``.

        Returning ``False`` here causes :meth:`forward_pod_envelope` to
        skip the ``emit_remote`` call — the legacy shape is intentionally
        NOT round-tripped through the typed adapter, both to avoid
        materializing thousands of synthetic ``UnknownEvent`` envelopes
        and to keep the wire-format migration explicit (anything pod
        wants journaled on control MUST be a typed event by the time
        Phase 4 lands).
        """
        return any(field in event for field in ("kind_dotted", "event_id", "schema_version"))

    def forward_pod_envelope(self, event: dict[str, Any]) -> None:
        """Validate ``event`` as a typed envelope and emit_remote it.

        Never raises — validation failures, missing emitter, missing
        envelope fields all turn into a warning + continue. The pod
        WS wire shape mirrors the canonical envelope; we just need
        to normalise the back-compat ``timestamp`` alias and undo the
        ``kind`` ↔ ``kind_dotted`` swap before handing the dict to
        :data:`EVENT_ADAPTER`.
        """
        if self._emitter is None:
            return
        if not self.is_new_envelope_shape(event):
            # Legacy shape — emitter would have to wrap as UnknownEvent
            # and that's intentionally not the goal here. Inline callbacks
            # above still ran, so the operator surfaces (logs / terminal)
            # are unaffected.
            return

        # Build a sanitised dict (do not mutate ``event`` — the caller
        # also reads from it for the offset bookkeeping below).
        envelope_dict = dict(event)

        # Restore the canonical ``kind`` field. Wire shape stores the
        # dotted kind under ``kind_dotted`` and overwrites ``kind`` with
        # the legacy alias (for pre-Phase-4 consumers). For validation
        # we need the dotted form back on ``kind``.
        kind_dotted = envelope_dict.pop("kind_dotted", None)
        if isinstance(kind_dotted, str) and kind_dotted:
            envelope_dict["kind"] = kind_dotted

        # ``timestamp`` is a back-compat mirror of canonical ``time``;
        # the envelope schema only accepts ``time``. If both are
        # present we keep ``time``; if only ``timestamp`` arrived
        # (defensive — should not happen with Phase 2 ``envelope_to_wire``)
        # we promote it.
        if "time" not in envelope_dict and "timestamp" in envelope_dict:
            envelope_dict["time"] = envelope_dict["timestamp"]
        envelope_dict.pop("timestamp", None)

        try:
            typed_event = EVENT_ADAPTER.validate_python(envelope_dict)
        except ValidationError as exc:
            logger.warning(
                "[MONITOR] dropping pod event (envelope validation failed): "
                "kind=%s err=%s",
                envelope_dict.get("kind"), exc,
            )
            return
        except Exception as exc:
            # Defensive — any unexpected validation error must not crash
            # the WS consumer loop. Logged and dropped.
            logger.warning(
                "[MONITOR] dropping pod event (unexpected validation error): "
                "kind=%s err=%s: %s",
                envelope_dict.get("kind"), type(exc).__name__, exc,
            )
            return

        # ``emit_remote`` is itself never-raises, but wrap defensively
        # so a future implementation that re-raises cannot kill the
        # WS consumer loop.
        try:
            assert isinstance(typed_event, BaseEvent)  # discriminator union narrows
            self._emitter.emit_remote(typed_event)
        except Exception as exc:
            # Defensive boundary — emit_remote's contract is "never raises"
            # but we still log + drop if something slips through.
            logger.warning(
                "[MONITOR] emit_remote failed (event dropped): "
                "kind=%s err=%s: %s",
                typed_event.kind, type(exc).__name__, exc,
            )

    def maybe_log_status(self, payload: dict[str, Any]) -> None:
        """Emit a rate-limited ``[MONITOR] ALIVE`` line for the operator.

        The legacy SSH-polling monitor printed a one-line status every
        15 s so the user could tell at a glance that training was
        progressing without tailing trainer stdout. We restore the
        same surface against the WS event stream — driven by the
        runner's ``health_snapshot`` events instead of an SSH probe.

        Missing fields render as ``—`` rather than ``0`` so the user
        can distinguish "GPU is genuinely idle" from "psutil/nvidia-smi
        couldn't read the value".
        """
        now = time.time()
        if now - self._state._last_status_log_time < self._log_status_interval:
            return
        self._state._last_status_log_time = now

        elapsed = max(0.0, now - self._state._training_start_time)
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        gpu_util = payload.get("gpu_util_percent")
        vram_pct = payload.get("gpu_memory_percent")
        vram_used = payload.get("vram_used_gb")
        vram_total = payload.get("vram_total_gb")
        gpu_temp = payload.get("gpu_temp_c")
        cpu = payload.get("cpu_percent")
        ram_used = payload.get("ram_used_gb")
        ram_total = payload.get("ram_total_gb")

        # VRAM: prefer absolute GB if both fields present (richer signal
        # for operator capacity planning), fall back to percent, else "—".
        if isinstance(vram_used, (int, float)) and isinstance(vram_total, (int, float)):
            vram_str = (
                f"{vram_used:.1f}/{vram_total:.0f} GB"
                + (f" ({vram_pct:.0f}%)" if isinstance(vram_pct, (int, float)) else "")
            )
        elif isinstance(vram_pct, (int, float)):
            vram_str = f"{vram_pct:.0f}%"
        else:
            vram_str = "—"

        ram_str = (
            f"{ram_used:.1f}/{ram_total:.0f} GB"
            if isinstance(ram_used, (int, float)) and isinstance(ram_total, (int, float))
            else "—"
        )

        # ``running`` matches the develop-branch convention; the legacy
        # ``ALIVE`` token confused operators ("alive vs what?") — here
        # we always log when the trainer is actively running so the
        # state name matches the FSM JobState.value.
        logger.info(
            "[MONITOR] running | %s | GPU: %s | VRAM: %s | Temp: %s | CPU: %s | RAM: %s",
            elapsed_str,
            f"{gpu_util:.0f}%" if isinstance(gpu_util, (int, float)) else "—",
            vram_str,
            f"{gpu_temp:.0f}C" if isinstance(gpu_temp, (int, float)) else "—",
            f"{cpu:.0f}%" if isinstance(cpu, (int, float)) else "—",
            ram_str,
        )

    # --- replay fallback (Group 1 fix) ---------------------------------

    async def replay_then_resume_or_fallback(
        self,
        client: JobClient,
        job_id: str,
    ) -> dict[str, Any]:
        """Drain the pod's HTTP replay endpoint, then re-subscribe at the
        fresh tail. Falls back to ``fallback_to_status`` on any
        transport error so the pipeline still terminates cleanly.

        Closes the post-Phase-6.a gap where the HTTP replay endpoint
        existed but was never invoked — the monitor jumped straight
        from :class:`ReplayTruncatedError` to a status snapshot,
        dropping the training timeline on every Mac sleep beyond the
        ring's tail.

        Pagination contract: the runner returns ``X-Next-Offset`` as
        either the last yielded offset, or the request's
        ``after_offset`` when no rows matched. We stop when the
        cursor stops advancing.
        """
        # Local imports to avoid an import cycle between this module
        # and ``training_monitor`` when the latter imports us at top.
        from ryotenkai_shared.errors import TrainingFailedError
        from ryotenkai_shared.utils.clients.job_client import ReplayTruncatedError

        # Resume cursor — the monitor tracks the last offset it saw
        # on the WS stream. After replay we re-subscribe from one past
        # the highest offset we observed via HTTP.
        last_offset = self._state._last_offset - 1 if self._state._last_offset > 0 else -1
        replayed_count = 0

        try:
            while True:
                events, next_offset = await client.replay_events(
                    job_id, after_offset=last_offset,
                )
                for event in events:
                    offset = event.get("offset")
                    if isinstance(offset, int):
                        self._state._last_offset = offset + 1
                    terminal = self.dispatch_event(event)
                    if terminal is not None:
                        return terminal
                    replayed_count += 1
                if next_offset <= last_offset or not events:
                    break
                last_offset = next_offset
        except JobNotFoundError:
            raise
        except JobClientError as exc:
            logger.warning(
                "[MONITOR] HTTP replay failed after ReplayTruncatedError "
                "(falling back to status snapshot): %s",
                exc,
            )
            return await self._fallback_to_status(client, job_id)
        except Exception as exc:
            # Defensive boundary — any unexpected replay failure should
            # not break the cleanup path; fall back to status snapshot.
            logger.warning(
                "[MONITOR] unexpected HTTP replay failure (falling back "
                "to status snapshot): %s: %s",
                type(exc).__name__, exc,
            )
            return await self._fallback_to_status(client, job_id)

        logger.info(
            "[MONITOR] HTTP replay drained %d events; re-subscribing WS "
            "from offset %d",
            replayed_count, self._state._last_offset,
        )

        # Re-subscribe with the post-replay offset so we pick up live
        # events going forward. If the re-subscribe also raises a
        # truncation error (extremely unlikely — the journal would have
        # to roll between the replay drain and the WS reconnect), we
        # fall back to the snapshot rather than recurse indefinitely.
        try:
            async for event in client.subscribe_events(
                job_id, since=self._state._last_offset,
            ):
                offset = event.get("offset")
                if isinstance(offset, int):
                    self._state._last_offset = offset + 1
                terminal = self.dispatch_event(event)
                if terminal is not None:
                    return terminal
            raise TrainingFailedError(
                detail=(
                    "runner closed the event stream before reaching "
                    "terminal state (post HTTP replay re-subscribe)"
                ),
                context={"legacy_code": "MONITOR_STREAM_EOF"},
            )
        except JobNotFoundError as exc:
            raise TrainingFailedError(
                detail=(
                    f"runner reports unknown job {job_id!r} "
                    "(post HTTP replay re-subscribe)"
                ),
                context={"legacy_code": "MONITOR_JOB_NOT_FOUND"},
                cause=exc,
            ) from exc
        except ReplayTruncatedError:
            return await self._fallback_to_status(client, job_id)
        except JobClientError as exc:
            raise TrainingFailedError(
                detail=f"runner client error (post HTTP replay): {exc}",
                context={"legacy_code": "MONITOR_CLIENT_ERROR"},
                cause=exc,
            ) from exc
