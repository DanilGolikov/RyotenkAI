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

Stages still returning legacy ``Result[T, AppError]`` are bridged by the
``_adapt_legacy_to_typed`` shim until their own batch (7-10) rewrites them.
The shim itself is scheduled for deletion in Batch 10.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ryotenkai_shared.errors import InternalError, RyotenkAIError
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from ryotenkai_shared.config import PipelineConfig


# TODO Phase A2 Batch 10: delete _adapt_legacy_to_typed once all stages
# return dict directly. While this shim lives, legacy Result-returning
# stages keep working without breaking the new raise-based loop.
def _adapt_legacy_to_typed(value: Any) -> dict[str, Any]:
    """Bridge a legacy ``Result[T, AppError]`` return into the typed shape.

    Stages migrated by Batches 7-10 will return ``dict[str, Any]`` directly
    and bypass this shim entirely. Until then, legacy ``Result`` objects
    coming back from ``execute()`` are unwrapped here:

    * ``Failure(err)`` raises :class:`InternalError` carrying the legacy
      ``code``/``message`` on ``context`` so observability stays intact.
    * ``Success(value)`` yields the inner ``value`` (or ``{}`` if absent).
    * Non-``Result`` values pass through (must be ``dict`` for the loop).
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
            RyotenkAIError: on stage failure. Legacy stages may still return
                ``Result[T, AppError]``; the shim ``_adapt_legacy_to_typed``
                in :meth:`run` converts them until Batches 7-10 land.
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
            Output dict from ``execute`` (post-shim for legacy stages).

        Raises:
            RyotenkAIError: on setup/execute failure. The shim
                :func:`_adapt_legacy_to_typed` converts legacy
                ``Result[T, AppError]`` returns into raises until Batches
                7-10 rewrite all stages.
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
            # Shim: bridge legacy Result-returning stages until Batches 7-10
            # rewrite them. Direct dict returns pass through unchanged.
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
