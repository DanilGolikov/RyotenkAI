from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from textual.worker import Worker, WorkerState

from src.tui.adapters.delete_backend import DeleteMode, DeleteResult, TuiDeleteBackend
from src.tui.logging import suppress_project_console_logging

if TYPE_CHECKING:
    from textual.dom import DOMNode


_TUI_LOG = logging.getLogger("ryotenkai.tui.delete_flow")


class DeleteAction(StrEnum):
    DELETE_ALL = "delete_all"
    DELETE_LOCAL_ONLY = "delete_local_only"
    CANCEL = "cancel"

    def to_mode(self) -> DeleteMode | None:
        if self == DeleteAction.DELETE_ALL:
            return DeleteMode.LOCAL_AND_MLFLOW
        if self == DeleteAction.DELETE_LOCAL_ONLY:
            return DeleteMode.LOCAL_ONLY
        return None


@dataclass(frozen=True, slots=True)
class DeleteRequest:
    targets: tuple[Path, ...]
    mode: DeleteMode


class _DeleteWorkerHost(Protocol):
    def run_worker(
        self,
        work,
        *,
        name: str | None = "",
        group: str = "default",
        description: str = "",
        exit_on_error: bool = True,
        start: bool = True,
        exclusive: bool = False,
        thread: bool = False,
    ) -> Worker[list[DeleteResult]]: ...


class TuiDeleteController:
    def __init__(
        self,
        host: _DeleteWorkerHost,
        *,
        service_factory: Callable[[], TuiDeleteBackend] = TuiDeleteBackend,
        on_pending: Callable[[DeleteRequest], None] | None = None,
        on_success: Callable[[DeleteRequest, list[DeleteResult]], None] | None = None,
        on_error: Callable[[DeleteRequest, Exception], None] | None = None,
    ) -> None:
        self._host = host
        self._service_factory = service_factory
        self._on_pending = on_pending
        self._on_success = on_success
        self._on_error = on_error
        self._workers: dict[Worker[list[DeleteResult]], DeleteRequest] = {}

    @property
    def is_busy(self) -> bool:
        return bool(self._workers)

    def start_delete(self, targets: Sequence[Path], *, mode: DeleteMode) -> bool:
        normalized_targets = tuple(target.expanduser().resolve() for target in targets)
        if not normalized_targets or self.is_busy:
            return False

        request = DeleteRequest(targets=normalized_targets, mode=mode)
        if self._on_pending is not None:
            self._on_pending(request)

        worker = self._host.run_worker(
            partial(self._run_delete_request, request),
            thread=True,
            exclusive=False,
            description="Delete run targets",
        )
        self._workers[worker] = request
        return True

    def handle_worker_state_changed(self, event: Worker.StateChanged) -> bool:
        request = self._workers.get(event.worker)
        if request is None:
            return False

        if event.state == WorkerState.SUCCESS:
            self._workers.pop(event.worker, None)
            if self._on_success is not None:
                self._on_success(request, event.worker.result or [])
            return True

        if event.state == WorkerState.ERROR:
            self._workers.pop(event.worker, None)
            error = event.worker.error
            if not isinstance(error, Exception):
                error = RuntimeError(str(error))
            if self._on_error is not None:
                self._on_error(request, error)
            return True

        return True

    def _run_delete_request(self, request: DeleteRequest) -> list[DeleteResult]:
        with suppress_project_console_logging():
            service = self._service_factory()
            return [service.delete_target(target, mode=request.mode) for target in request.targets]


def format_delete_completion_message(request: DeleteRequest, results: Sequence[DeleteResult]) -> str:
    deleted_targets = sum(1 for result in results if result.local_deleted)
    deleted_mlflow_runs = sum(len(result.deleted_mlflow_run_ids) for result in results)

    if request.mode == DeleteMode.LOCAL_ONLY:
        return f"Deleted {deleted_targets} folder(s). MLflow runs kept."

    if deleted_mlflow_runs:
        return f"Deleted {deleted_targets} folder(s) and {deleted_mlflow_runs} MLflow run(s)."

    return f"Deleted {deleted_targets} folder(s). No linked MLflow runs found."


__all__ = [
    "DeleteAction",
    "DeleteRequest",
    "TuiDeleteController",
    "format_delete_completion_message",
    "suppress_project_console_logging",
]
