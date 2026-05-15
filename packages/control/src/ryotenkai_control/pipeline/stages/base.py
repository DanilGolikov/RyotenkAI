"""
Base class for all pipeline stages.
Implements the Strategy Pattern for flexible stage execution.

Lifecycle:
    1. setup()   - Initialize resources (optional, override if needed)
    2. execute() - Main stage logic (required)
    3. teardown() - Cleanup resources (always called after execute, even on error)
    4. cleanup() - Emergency cleanup (called on pipeline failure)

Phase A2 Batch 6 — raise-based interface
----------------------------------------
The legacy ``Result[T, AppError]`` return shape was migrated to raise-based:

* ``setup(ctx) -> None`` — raises ``RyotenkAIError`` on failure.
* ``execute(ctx) -> dict[str, Any]`` — returns output dict, raises on failure.
* ``teardown(ctx) -> None`` — raises on failure (best-effort; swallowed by run()).
* ``run(ctx) -> dict[str, Any]`` — orchestrates setup → execute → teardown.

Phase A2 Batch 10 (2026-05-15) — every control-side pipeline stage now
returns ``dict[str, Any]`` or raises :class:`RyotenkAIError` directly.
The :func:`_adapt_legacy_to_typed` shim is kept as a **defensive guard**
for test mocks that still wire ``stage.run.return_value = Ok(...)`` (used
heavily in ``tests/unit/control/test_pipeline_orchestrator.py``). The
shim is a pure pass-through for plain dicts and never affects production
flow now that no stage actually returns ``Result``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ryotenkai_shared.errors import InternalError, RyotenkAIError
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from ryotenkai_shared.config import PipelineConfig


def _adapt_legacy_to_typed(value: Any) -> dict[str, Any]:
    """Backward-compat shim — pass-through for dicts; unwrap legacy
    ``Result[T, AppError]`` returns from test mocks; normalise everything
    else to ``{}``.

    No production stage returns ``Result`` after Phase A2 Batch 10, so
    the legacy-unwrap branches only fire for unit tests that mock
    ``stage.run.return_value = Ok(...)`` for orchestrator coverage.
    Failure mocks raise :class:`InternalError` carrying the legacy
    ``code``/``message`` on ``context``.
    """
    if hasattr(value, "is_failure") and callable(value.is_failure):
        if value.is_failure():
            err = value.unwrap_err()
            legacy_code = getattr(err, "code", "UNKNOWN")
            legacy_message = getattr(err, "message", None) or str(err)
            raise InternalError(
                detail=legacy_message,
                context={
                    "legacy_code": legacy_code,
                    "legacy_details": getattr(err, "details", None),
                },
            )
        if hasattr(value, "unwrap") and callable(value.unwrap):
            unwrapped = value.unwrap()
            return unwrapped if isinstance(unwrapped, dict) else {}
        return {}
    return value if isinstance(value, dict) else {}


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.

    Each stage must implement the execute method.
    Optionally override setup(), teardown(), and cleanup() for resource management.

    Lifecycle:
        1. setup(context) - Called before execute. Initialize resources here.
        2. execute(context) - Main stage logic. Required.
        3. teardown() - Called after execute (always, even on error). Cleanup resources here.
        4. cleanup() - Called by orchestrator on pipeline failure. Emergency cleanup.

    Example:
        class MyStage(PipelineStage):
            def setup(self, context):
                self._connection = create_connection()

            def execute(self, context):
                # use self._connection
                return {"output": ...}

            def teardown(self):
                if self._connection:
                    self._connection.close()

            def cleanup(self):
                # Emergency cleanup - force close resources
                self.teardown()
    """

    def __init__(self, config: PipelineConfig, stage_name: str):
        self.config = config
        self.stage_name = stage_name
        self.metadata: dict[str, Any] = {}

    def setup(self, _context: dict[str, Any]) -> None:
        """
        Initialize stage resources before execute.

        Override this method if your stage needs to:
        - Open connections
        - Create temporary files
        - Allocate resources

        Args:
            _context: Pipeline context (unused in base implementation)

        Raises:
            RyotenkAIError: on setup failure (caller routes via _handle_stage_failure).
        """
        return None

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the pipeline stage.

        Args:
            context: Dictionary containing data from previous stages

        Returns:
            Output dict (merged into the pipeline context by the loop).

        Raises:
            RyotenkAIError: on stage failure.
        """

    def teardown(self) -> None:
        """
        Cleanup stage resources after execute.

        ALWAYS called after execute, even if execute fails.
        Override this method if your stage needs to:
        - Close connections
        - Delete temporary files
        - Release resources

        Should not raise exceptions - log warnings instead.
        """
        return None

    def cleanup(self) -> None:
        """
        Pipeline-level cleanup (called by orchestrator in a finally block).

        Use this for resources that must live beyond a single stage execution,
        e.g. cloud GPUs / remote inference endpoints created in an earlier stage
        and torn down at the very end of the pipeline.

        Default: no-op.
        Should not raise exceptions - log warnings instead.
        """
        return None

    def log_start(self) -> None:
        """Log the start of the stage."""
        logger.info(f"🚀 Starting: {self.stage_name}")

    def log_end(self, success: bool) -> None:
        """Log the end of the stage."""
        status = "✅" if success else "❌"
        logger.info(f"{status} Stage completed: {self.stage_name}\n")

    def update_context(self, context: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        """
        Update the context with new data from this stage.

        Args:
            context: Current context
            updates: New data to add

        Returns:
            Updated context dictionary
        """
        context[self.stage_name] = updates
        return context

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Wrapper method that adds lifecycle management around execute.

        Lifecycle:
            1. log_start()
            2. setup(context) - if raises, skip execute; teardown not run.
            3. execute(context) - any raised RyotenkAIError propagates.
            4. teardown() - always called if setup completed (even on execute failure).
            5. log_end()

        Args:
            context: Dictionary containing data from previous stages

        Returns:
            Output dict from ``execute``.

        Raises:
            RyotenkAIError: on setup/execute failure.
            KeyboardInterrupt / SystemExit: re-raised verbatim — the loop
                owns the interrupt boundary.
        """
        self.log_start()

        # Step 1: Setup
        setup_completed = False
        try:
            self.setup(context)
            setup_completed = True
        except RyotenkAIError:
            # Setup failed — surface to loop unchanged. log_end first so
            # observability reflects the failure path.
            logger.error(f"Setup failed in {self.stage_name}")
            self.log_end(False)
            raise
        except KeyboardInterrupt:
            self.log_end(False)
            raise
        except Exception as exc:
            # Unexpected (non-typed) setup error → wrap as InternalError so
            # the loop only ever catches RyotenkAIError.
            logger.exception(f"Setup error in {self.stage_name}: {exc}")
            self.log_end(False)
            raise InternalError(
                detail=f"Setup error in {self.stage_name}: {exc!s}",
                context={"stage": self.stage_name, "exception_type": type(exc).__name__},
                cause=exc,
            ) from exc

        # Step 2: Execute (+ teardown finally)
        try:
            raw = self.execute(context)
            # ``_adapt_legacy_to_typed`` normalises return value to dict.
            # Production stages already return dict; the shim only matters
            # for test mocks that still wire ``Ok(...)`` /``Err(...)``.
            output = _adapt_legacy_to_typed(raw)
            self.log_end(True)
            return output
        except RyotenkAIError:
            self.log_end(False)
            raise
        except KeyboardInterrupt:
            # Re-raise so orchestrator can handle it (direct KBI path, e.g. unit tests
            # without a registered signal handler).  In production the signal handler
            # calls sys.exit() → SystemExit, which bypasses this block entirely and
            # propagates through the finally clause automatically.
            self.log_end(False)
            raise
        except Exception as exc:
            logger.exception(f"Unexpected error in {self.stage_name}: {exc}")
            self.log_end(False)
            raise InternalError(
                detail=f"Unexpected error in {self.stage_name}: {exc!s}",
                context={"stage": self.stage_name, "exception_type": type(exc).__name__},
                cause=exc,
            ) from exc
        finally:
            # Step 3: Teardown (always if setup succeeded)
            if setup_completed:
                try:
                    self.teardown()
                except Exception as exc:  # noqa: BLE001 — defensive
                    logger.warning(f"Teardown error in {self.stage_name}: {exc}")


__all__ = ["PipelineStage", "_adapt_legacy_to_typed"]
