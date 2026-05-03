"""
MLflowEventLog — in-memory event store with OpenTelemetry-compatible schema.

Responsibilities (Single Responsibility):
  - Store events in memory during pipeline run
  - Provide filtered read access to events
  - Export events as JSON artifact via injected log_dict callable

No MLflow SDK calls — purely in-memory with a callback for persistence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.training.constants import (
    CATEGORY_TRAINING,
    MLFLOW_EVENTS_DISPLAY_LIMIT,
    MLFLOW_KEY_EVENT_TYPE,
    MLFLOW_MD_SEPARATOR,
    MLFLOW_OTEL_INFO,
    MLFLOW_SEVERITY_ERROR,
    MLFLOW_SEVERITY_INFO,
    MLFLOW_SEVERITY_START,
    MLFLOW_SEVERITY_WARNING,
)
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class MLflowEventLog:
    """
    In-memory event log with OpenTelemetry-compatible schema.

    Owns all event state: _events list, _event_counter, _has_errors.
    Does not depend on any MLflow SDK — persistence is delegated to
    a log_dict callable injected at export time.

    Usage:
        event_log = MLflowEventLog()
        event_log.log_event("start", "Training started", category="training")
        events = event_log.get_events(category="training")
        event_log.log_events_artifact("training_events.json", log_dict_fn=manager.log_dict)
    """

    # OpenTelemetry-compatible severity mapping
    _SEVERITY_MAP: ClassVar[dict[str, tuple[str, int]]] = {
        MLFLOW_SEVERITY_START: (MLFLOW_OTEL_INFO, 9),
        "complete": (MLFLOW_OTEL_INFO, 9),
        MLFLOW_SEVERITY_INFO: (MLFLOW_OTEL_INFO, 9),
        "checkpoint": (MLFLOW_OTEL_INFO, 9),
        MLFLOW_SEVERITY_WARNING: ("WARN", 13),
        MLFLOW_SEVERITY_ERROR: ("ERROR", 17),
    }

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._event_counter: int = 0
        self._has_errors: bool = False

    @property
    def has_errors(self) -> bool:
        """True if any error event was recorded."""
        return self._has_errors

    @property
    def event_count(self) -> int:
        """Total number of recorded events."""
        return self._event_counter

    def log_event(
        self,
        event_type: str,
        message: str,
        *,
        category: str = CATEGORY_TRAINING,
        source: str = "",
        **metadata: Any,
    ) -> dict[str, Any]:
        """
        Log a pipeline event in normalized OpenTelemetry-compatible format.

        Schema:
            - timestamp: ISO-8601 timestamp
            - event_type: start|complete|info|checkpoint|warning|error
            - severity: OTEL severity text (INFO/WARN/ERROR)
            - severity_number: OTEL severity number (9/13/17)
            - message: Human-readable description
            - category: training|memory|system|pipeline
            - source: Component that generated event
            - attributes: Additional key-value data (normalized)

        Args:
            event_type: Type of event
            message: Human-readable description
            category: Event category
            source: Component that generated the event
            **metadata: Additional key-value data → goes to `attributes`

        Returns:
            Event dict
        """
        from datetime import datetime

        severity_text, severity_number = self._SEVERITY_MAP.get(event_type, (MLFLOW_OTEL_INFO, 9))

        event: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            MLFLOW_KEY_EVENT_TYPE: event_type,
            "severity": severity_text,
            "severity_number": severity_number,
            "message": message,
            "category": category,
            "source": source,
        }

        if metadata:
            event["attributes"] = metadata

        self._events.append(event)
        self._event_counter += 1

        if event_type == MLFLOW_SEVERITY_ERROR:
            self._has_errors = True

        logger.debug(f"[MLFLOW:EVENT] [{event_type.upper()}] {message}")
        return event

    def log_event_start(self, message: str, **kwargs: Any) -> dict[str, Any]:
        """Log start event."""
        return self.log_event(MLFLOW_SEVERITY_START, message, **kwargs)

    def log_event_complete(self, message: str, **kwargs: Any) -> dict[str, Any]:
        """Log complete event."""
        return self.log_event("complete", message, **kwargs)

    def log_event_error(self, message: str, **kwargs: Any) -> dict[str, Any]:
        """Log error event."""
        return self.log_event(MLFLOW_SEVERITY_ERROR, message, **kwargs)

    def log_event_warning(self, message: str, **kwargs: Any) -> dict[str, Any]:
        """Log warning event."""
        return self.log_event(MLFLOW_SEVERITY_WARNING, message, **kwargs)

    def log_event_info(self, message: str, **kwargs: Any) -> dict[str, Any]:
        """Log info event."""
        return self.log_event(MLFLOW_SEVERITY_INFO, message, **kwargs)

    def log_event_checkpoint(self, message: str, **kwargs: Any) -> dict[str, Any]:
        """Log checkpoint event."""
        return self.log_event("checkpoint", message, **kwargs)

    def get_events(self, category: str | None = None) -> list[dict[str, Any]]:
        """
        Get collected events.

        Args:
            category: Filter by category (optional)

        Returns:
            List of event dicts (copy)
        """
        if category is None:
            return self._events.copy()
        return [e for e in self._events if e.get("category") == category]

    def log_events_artifact(
        self,
        artifact_name: str = "training_events.json",
        *,
        log_dict_fn: Callable[..., bool],
        run_id: str | None = None,
    ) -> bool:
        """
        Export events as JSON artifact via injected callable.

        Args:
            artifact_name: Artifact filename
            log_dict_fn: Callable matching signature log_dict(dict, str, run_id) -> bool
            run_id: Optional run_id to log to

        Returns:
            True if logged successfully
        """
        if not self._events:
            logger.debug("[MLFLOW:EVENTS] No events to log")
            return False

        try:
            log_dict_fn({"events": self._events, "total": len(self._events)}, artifact_name, run_id)
            logger.info(f"[MLFLOW:EVENTS] Logged {len(self._events)} events as artifact")
            return True
        except Exception as e:
            logger.warning(f"[MLFLOW:EVENTS] Failed to log events artifact: {e}")
            return False

    def generate_summary_section(self) -> list[str]:
        """
        Generate the Events Timeline section for a Markdown summary.

        Returns:
            List of Markdown lines
        """
        lines: list[str] = []
        lines.append("## Events Timeline")
        lines.append("")

        if self._events:
            lines.append("| Time | Category | Event |")
            lines.append("|------|----------|-------|")

            display_events = (
                self._events[-MLFLOW_EVENTS_DISPLAY_LIMIT:]
                if len(self._events) > MLFLOW_EVENTS_DISPLAY_LIMIT
                else self._events
            )
            if len(self._events) > MLFLOW_EVENTS_DISPLAY_LIMIT:
                lines.append(f"| ... | | *({len(self._events) - MLFLOW_EVENTS_DISPLAY_LIMIT} earlier events hidden)* |")

            import contextlib

            for event in display_events:
                time_str = event.get("timestamp", "")
                if time_str:
                    with contextlib.suppress(IndexError, AttributeError):
                        time_str = time_str.split("T")[1].split(".")[0]
                category = event.get("category", "")
                event_type = event.get(MLFLOW_KEY_EVENT_TYPE, "").upper()
                message = event.get("message", "")
                lines.append(f"| {time_str} | {category} | [{event_type}] {message} |")
        else:
            lines.append("*(No events recorded)*")

        lines.append("")
        lines.append(MLFLOW_MD_SEPARATOR)
        lines.append("")

        # Event statistics
        if self._events:
            error_count = len([e for e in self._events if e.get(MLFLOW_KEY_EVENT_TYPE) == "error"])
            warning_count = len([e for e in self._events if e.get(MLFLOW_KEY_EVENT_TYPE) == "warning"])

            lines.append("### Event Statistics")
            lines.append(f"- Total events: {len(self._events)}")
            if error_count > 0:
                lines.append(f"- Errors: {error_count}")
            if warning_count > 0:
                lines.append(f"- Warnings: {warning_count}")
            lines.append("")

        return lines

    def clear(self) -> None:
        """Reset all event state (called on cleanup)."""
        self._events.clear()
        self._event_counter = 0
        self._has_errors = False


__all__ = ["MLflowEventLog"]
