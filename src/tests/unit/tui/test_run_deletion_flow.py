from __future__ import annotations

import logging
from pathlib import Path

from textual.worker import WorkerState

from src.tui.adapters.delete_backend import DeleteMode, DeleteResult
from src.tui.run_deletion_flow import (
    DeleteAction,
    DeleteRequest,
    TuiDeleteController,
    format_delete_completion_message,
    suppress_project_console_logging,
)


def test_delete_action_maps_to_run_deletion_mode() -> None:
    assert DeleteAction.DELETE_ALL.to_mode() == DeleteMode.LOCAL_AND_MLFLOW
    assert DeleteAction.DELETE_LOCAL_ONLY.to_mode() == DeleteMode.LOCAL_ONLY
    assert DeleteAction.CANCEL.to_mode() is None


def test_format_delete_completion_message_for_full_delete() -> None:
    request = DeleteRequest(
        targets=(Path("/tmp/run_1"),),
        mode=DeleteMode.LOCAL_AND_MLFLOW,
    )
    results = [
        DeleteResult(
            target=Path("/tmp/run_1"),
            run_dirs=(Path("/tmp/run_1"),),
            deleted_mlflow_run_ids=("root_1", "child_1"),
            local_deleted=True,
            issues=(),
        )
    ]

    message = format_delete_completion_message(request, results)

    assert message == "Deleted 1 folder(s) and 2 MLflow run(s)."


def test_format_delete_completion_message_for_local_only_delete() -> None:
    request = DeleteRequest(
        targets=(Path("/tmp/run_1"),),
        mode=DeleteMode.LOCAL_ONLY,
    )
    results = [
        DeleteResult(
            target=Path("/tmp/run_1"),
            run_dirs=(Path("/tmp/run_1"),),
            deleted_mlflow_run_ids=(),
            local_deleted=True,
            issues=(),
        )
    ]

    message = format_delete_completion_message(request, results)

    assert message == "Deleted 1 folder(s). MLflow runs kept."


def test_controller_starts_worker_and_dispatches_success_callback(tmp_path: Path) -> None:
    target = tmp_path / "run_1"
    observed_pending: list[DeleteRequest] = []
    observed_success: list[tuple[DeleteRequest, list[DeleteResult]]] = []
    worker_calls: list[dict[str, object]] = []

    class DummyWorker:
        def __init__(self) -> None:
            self.result = None
            self.error = None

    class DummyHost:
        def __init__(self) -> None:
            self.worker = DummyWorker()

        def run_worker(self, work, **kwargs):
            worker_calls.append({"work": work, "kwargs": kwargs})
            return self.worker

    class DummyService:
        def delete_target(self, path, mode):
            return DeleteResult(
                target=path,
                run_dirs=(path,),
                deleted_mlflow_run_ids=("root_1",) if mode == DeleteMode.LOCAL_AND_MLFLOW else (),
                local_deleted=True,
                issues=(),
            )

    class DummyEvent:
        def __init__(self, worker, state) -> None:
            self.worker = worker
            self.state = state

    host = DummyHost()
    controller = TuiDeleteController(
        host,
        service_factory=DummyService,
        on_pending=observed_pending.append,
        on_success=lambda request, results: observed_success.append((request, results)),
    )

    assert controller.start_delete([target], mode=DeleteMode.LOCAL_AND_MLFLOW) is True
    assert controller.is_busy is True
    assert len(observed_pending) == 1
    assert worker_calls[0]["kwargs"]["thread"] is True

    host.worker.result = worker_calls[0]["work"]()
    event = DummyEvent(host.worker, WorkerState.SUCCESS)

    assert controller.handle_worker_state_changed(event) is True
    assert controller.is_busy is False
    assert len(observed_success) == 1
    assert observed_success[0][0].targets == (target.resolve(),)
    assert observed_success[0][1][0].deleted_mlflow_run_ids == ("root_1",)


def test_suppress_project_console_logging_restores_stream_handler() -> None:
    logger = logging.getLogger("ryotenkai")
    original_handlers = list(logger.handlers)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    try:
        with suppress_project_console_logging():
            assert stream_handler not in logger.handlers
        assert stream_handler in logger.handlers
    finally:
        logger.handlers.clear()
        logger.handlers.extend(original_handlers)
