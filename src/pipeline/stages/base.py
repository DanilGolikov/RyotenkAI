"""
Base class for all pipeline stages.
Implements the Strategy Pattern for flexible stage execution.

Lifecycle:
    1. setup()   - Initialize resources (optional, override if needed)
    2. execute() - Main stage logic (required)
    3. teardown() - Cleanup resources (always called after execute, even on error)
    4. cleanup() - Emergency cleanup (called on pipeline failure)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from src.utils.logger import logger
from src.utils.result import AppError, Err, Ok, Result

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig


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
                return Ok(None)

            def execute(self, context):
                # use self._connection
                return Ok({...})

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

    def setup(self, _context: dict[str, Any]) -> Result[None, AppError]:
        """
        Initialize stage resources before execute.

        Override this method if your stage needs to:
        - Open connections
        - Create temporary files
        - Allocate resources

        Args:
            _context: Pipeline context (unused in base implementation)

        Returns:
            Result[None, AppError]: Ok if setup successful, Err with AppError otherwise
        """
        return Ok(None)

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> Result[dict[str, Any], AppError]:
        """
        Execute the pipeline stage.

        Args:
            context: Dictionary containing data from previous stages

        Returns:
            Result[Dict[str, Any], AppError]: Success with updated context or Err with AppError
        """
        pass

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

    def run(self, context: dict[str, Any]) -> Result[dict[str, Any], AppError]:
        """
        Wrapper method that adds lifecycle management around execute.

        Lifecycle:
            1. log_start()
            2. setup(context) - if fails, skip execute and teardown
            3. execute(context)
            4. teardown() - always called if setup succeeded
            5. log_end()

        Args:
            context: Dictionary containing data from previous stages

        Returns:
            Result[Dict[str, Any], AppError]: Success with updated context or Err with AppError
        """
        self.log_start()

        # Step 1: Setup
        setup_completed = False
        try:
            setup_result = self.setup(context)
            if setup_result.is_failure():
                err_val = setup_result.unwrap_err()
                logger.error(f"Setup failed in {self.stage_name}: {err_val}")
                self.log_end(False)
                return Err(err_val)
            setup_completed = True
        except Exception as e:
            logger.exception(f"Setup error in {self.stage_name}: {e}")
            self.log_end(False)
            return Err(AppError(message=f"Setup error in {self.stage_name}: {e!s}", code="SETUP_ERROR"))

        # Step 2: Execute
        try:
            result = self.execute(context)
            self.log_end(result.is_success())
            return result
        except KeyboardInterrupt:
            # Re-raise so orchestrator can handle it (direct KBI path, e.g. unit tests
            # without a registered signal handler).  In production the signal handler
            # calls sys.exit() → SystemExit, which bypasses this block entirely and
            # propagates through the finally clause automatically.
            self.log_end(False)
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in {self.stage_name}: {e}")
            self.log_end(False)
            return Err(
                AppError(
                    message=f"Unexpected error in {self.stage_name}: {e!s}",
                    code="UNEXPECTED_ERROR",
                    details={"stage": self.stage_name, "exception_type": type(e).__name__},
                )
            )
        finally:
            # Step 3: Teardown (always if setup succeeded)
            if setup_completed:
                try:
                    self.teardown()
                except Exception as e:
                    logger.warning(f"Teardown error in {self.stage_name}: {e}")
