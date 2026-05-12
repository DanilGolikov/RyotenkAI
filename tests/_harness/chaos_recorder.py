"""DebugRecorder — append-only timeline of chaos-scenario events.

Each :class:`ChaosScenario` step (``precondition``, ``inject``,
``steady_state``, ``cleanup``) appends entries via the
:class:`DebugRecorder` carried on the :class:`ScenarioContext`. On
failure the :class:`ScenarioRunner` flushes a JSON dump alongside
``tests/.debug_bundles`` so the postmortem includes both the timeline
of attempted interventions AND the sidecar-state snapshots captured
before / after each step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tests._harness.clock import Clock, RealClock


@dataclass
class RecorderEntry:
    """One row of the chaos timeline."""

    timestamp: float
    step: str
    event: str
    payload: dict[str, Any] = field(default_factory=dict)


class DebugRecorder:
    """Append-only, JSON-serialisable timeline of scenario events.

    The recorder is intentionally tiny: tests / scenarios push named
    events with a free-form payload, the runner serialises them on
    failure. There is no automatic rotation — the in-memory buffer
    lives only for the duration of a single scenario.
    """

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock: Clock = clock if clock is not None else RealClock()
        self._entries: list[RecorderEntry] = []

    def record(self, step: str, event: str, **payload: Any) -> None:
        """Append one entry with the current clock timestamp.

        ``step`` is one of ``precondition`` / ``inject`` /
        ``steady_state`` / ``cleanup`` (the four lifecycle methods on
        :class:`ChaosScenario`). ``event`` is a free-form short name
        like ``"inject_429"`` or ``"assert_succeeded"``. ``payload`` is
        forwarded verbatim — keep it JSON-serialisable.
        """
        self._entries.append(
            RecorderEntry(
                timestamp=self._clock.now(),
                step=step,
                event=event,
                payload=dict(payload),
            ),
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [
            {
                "timestamp": entry.timestamp,
                "step": entry.step,
                "event": entry.event,
                "payload": dict(entry.payload),
            }
            for entry in self._entries
        ]

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_list(), indent=indent, default=str)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._entries)


__all__ = ["DebugRecorder", "RecorderEntry"]
