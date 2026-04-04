"""
MLflowRunLifecycleMixin — MLflow run lifecycle management.

Responsibilities:
  - start_run()       — context manager for parent runs
  - start_nested_run() — context manager for nested (child) runs
  - end_run()         — explicit run termination
  - _get_active_run_id() — helper for resolving current run id
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Generator

logger = get_logger(__name__)


class MLflowRunLifecycleMixin:
    """
    Mixin: MLflow run lifecycle — start, end, and nested run management.

    Assumes the following attributes exist on self (set by MLflowManager.__init__):
      _mlflow, _run, _run_id, _parent_run_id, _nested_run_stack
    """

    def _get_active_run_id(self, run_id: str | None = None) -> str | None:
        return run_id or self._run_id  # type: ignore[attr-defined]

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        description: str | None = None,
    ) -> Generator[Any, None, None]:
        """Start a parent MLflow run (context manager)."""
        if self._mlflow is None:  # type: ignore[attr-defined]
            yield None
            return

        if description is None:
            description = self._load_description_file()  # type: ignore[attr-defined]

        try:
            with self._mlflow.start_run(  # type: ignore[attr-defined]
                run_name=run_name,
                description=description,
                log_system_metrics=True,
            ) as run:
                self._run = run  # type: ignore[attr-defined]
                self._run_id = run.info.run_id  # type: ignore[attr-defined]
                self._parent_run_id = run.info.run_id  # type: ignore[attr-defined]
                logger.info(f"[MLFLOW] Run started: {run.info.run_id}")
                yield run
                logger.info(f"[MLFLOW] Run completed: {run.info.run_id}")
        except Exception as e:
            logger.warning(f"[MLFLOW] Run error: {e}")
            yield None
        finally:
            self._run = None  # type: ignore[attr-defined]
            self._parent_run_id = None  # type: ignore[attr-defined]
            self._nested_run_stack.clear()  # type: ignore[attr-defined]

    def end_run(self, status: str = "FINISHED") -> None:
        """Explicitly end current run with status."""
        if self._mlflow is None:  # type: ignore[attr-defined]
            return
        try:
            self._mlflow.end_run(status=status)  # type: ignore[attr-defined]
            logger.info(f"[MLFLOW] Run ended: {status}")
        except Exception as e:
            logger.warning(f"[MLFLOW] Failed to end run: {e}")

    @contextmanager
    def start_nested_run(
        self,
        run_name: str,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> Generator[Any, None, None]:
        """Start a nested (child) run within current parent run."""
        if self._mlflow is None:  # type: ignore[attr-defined]
            yield None
            return

        if self._parent_run_id is None:  # type: ignore[attr-defined]
            logger.warning("[MLFLOW] start_nested_run called without parent run, starting as regular run")
            with self.start_run(run_name=run_name, description=description) as run:  # type: ignore[attr-defined]
                if tags:
                    self.set_tags(tags)  # type: ignore[attr-defined]
                yield run
            return

        try:
            with self._mlflow.start_run(  # type: ignore[attr-defined]
                run_name=run_name,
                nested=True,
                description=description,
                log_system_metrics=True,
            ) as nested_run:
                self._run = nested_run  # type: ignore[attr-defined]
                self._run_id = nested_run.info.run_id  # type: ignore[attr-defined]
                self._nested_run_stack.append(nested_run.info.run_id)  # type: ignore[attr-defined]

                if tags:
                    self._mlflow.set_tags(tags)  # type: ignore[attr-defined]

                self._mlflow.set_tags(  # type: ignore[attr-defined]
                    {
                        "mlflow.parentRunId": self._parent_run_id,  # type: ignore[attr-defined]
                        "nested_run_depth": str(len(self._nested_run_stack)),  # type: ignore[attr-defined]
                    }
                )

                logger.info(
                    f"[MLFLOW] Nested run started: {run_name} "
                    f"(parent: {self._parent_run_id[:8]}...)"  # type: ignore[attr-defined]
                )
                yield nested_run
                logger.info(f"[MLFLOW] Nested run completed: {run_name}")

        except Exception as e:
            logger.warning(f"[MLFLOW] Nested run error: {e}")
            yield None
        finally:
            if self._nested_run_stack:  # type: ignore[attr-defined]
                self._nested_run_stack.pop()  # type: ignore[attr-defined]
            self._run_id = (  # type: ignore[attr-defined]
                self._nested_run_stack[-1]  # type: ignore[attr-defined]
                if self._nested_run_stack  # type: ignore[attr-defined]
                else self._parent_run_id  # type: ignore[attr-defined]
            )


__all__ = ["MLflowRunLifecycleMixin"]
