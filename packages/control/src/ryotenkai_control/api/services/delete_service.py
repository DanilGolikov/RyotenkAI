from __future__ import annotations

from pathlib import Path

from ryotenkai_control.api.schemas.delete import DeleteIssueSchema, DeleteResultSchema
from ryotenkai_control.pipeline.deletion import DeleteMode, DeleteResult, RunDeleter
from ryotenkai_control.pipeline.launch import is_process_alive, read_lock_pid
from ryotenkai_shared.errors import RunIsActiveError as _SharedRunIsActiveError


def _result_to_schema(result: DeleteResult) -> DeleteResultSchema:
    return DeleteResultSchema(
        target=str(result.target),
        run_dirs=[str(p) for p in result.run_dirs],
        deleted_mlflow_run_ids=list(result.deleted_mlflow_run_ids),
        local_deleted=result.local_deleted,
        issues=[
            DeleteIssueSchema(run_dir=str(issue.run_dir), phase=issue.phase, message=issue.message)
            for issue in result.issues
        ],
        is_success=result.is_success,
    )


def ensure_not_running(run_dir: Path) -> None:
    pid = read_lock_pid(run_dir)
    if pid is not None and is_process_alive(pid):
        raise RunIsActiveError(pid)


def delete_run(run_dir: Path, mode: str, *, deleter: RunDeleter | None = None) -> DeleteResultSchema:
    ensure_not_running(run_dir)
    parsed_mode = DeleteMode(mode)
    result = (deleter or RunDeleter()).delete_target(run_dir, mode=parsed_mode)
    return _result_to_schema(result)


class RunIsActiveError(_SharedRunIsActiveError):
    """Run is active and cannot be deleted.

    Phase C: inherits from the shared typed error so the control API
    error handlers (RFC 9457 problem+json) can convert it without a
    bespoke ``HTTPException`` adapter. ``.pid`` is preserved for back-
    compat with router-side handling code.
    """

    def __init__(self, pid: int) -> None:
        super().__init__(
            f"run is active (pid={pid}), interrupt before deleting",
            context={"pid": pid},
        )
        self.pid = pid


__all__ = ["RunIsActiveError", "delete_run", "ensure_not_running"]
