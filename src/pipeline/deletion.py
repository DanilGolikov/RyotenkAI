"""Run deletion service.

Deletes a local run directory (and any nested runs) and, when configured,
cascades deletion to the MLflow experiment tree rooted at the run's
root_mlflow_run_id. Shared domain service used by both the CLI and the web API.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from src.infrastructure.mlflow.environment import MLflowEnvironment
from src.pipeline.state.queries import discover_run_dirs, load_pipeline_state

_LOG = logging.getLogger("ryotenkai.pipeline.deletion")


class DeleteMode(StrEnum):
    LOCAL_AND_MLFLOW = "local_and_mlflow"
    LOCAL_ONLY = "local_only"


@dataclass(frozen=True, slots=True)
class DeleteIssue:
    run_dir: Path
    phase: str
    message: str


@dataclass(frozen=True, slots=True)
class DeleteResult:
    target: Path
    run_dirs: tuple[Path, ...]
    deleted_mlflow_run_ids: tuple[str, ...]
    local_deleted: bool
    issues: tuple[DeleteIssue, ...]

    @property
    def is_success(self) -> bool:
        return not self.issues


class RunDeleter:
    """Delete local runs and optionally their linked MLflow run tree."""

    def __init__(self, *, rmtree=shutil.rmtree) -> None:
        self._rmtree = rmtree

    def delete_target(self, target: Path, *, mode: DeleteMode = DeleteMode.LOCAL_AND_MLFLOW) -> DeleteResult:
        resolved_target = target.expanduser().resolve()
        run_dirs = discover_run_dirs(resolved_target)
        deleted_mlflow_run_ids: list[str] = []
        issues: list[DeleteIssue] = []

        if mode == DeleteMode.LOCAL_AND_MLFLOW:
            for run_dir in run_dirs:
                issues.extend(self._delete_mlflow_for_run(run_dir, deleted_mlflow_run_ids))

        blocking_issues = [issue for issue in issues if self._is_blocking_issue(issue)]
        if blocking_issues:
            return DeleteResult(
                target=resolved_target,
                run_dirs=run_dirs,
                deleted_mlflow_run_ids=tuple(deleted_mlflow_run_ids),
                local_deleted=False,
                issues=tuple(issues),
            )

        try:
            self._rmtree(resolved_target)
        except Exception as exc:
            issues.append(DeleteIssue(run_dir=resolved_target, phase="filesystem_delete", message=str(exc)))
            _LOG.exception("Filesystem delete failed for %s", resolved_target)

        return DeleteResult(
            target=resolved_target,
            run_dirs=run_dirs,
            deleted_mlflow_run_ids=tuple(deleted_mlflow_run_ids),
            local_deleted=not any(issue.phase == "filesystem_delete" for issue in issues),
            issues=tuple(issues),
        )

    def _delete_mlflow_for_run(self, run_dir: Path, deleted_ids: list[str]) -> list[DeleteIssue]:
        try:
            state = load_pipeline_state(run_dir)
        except Exception as exc:
            _LOG.exception("Failed to load pipeline state for %s", run_dir)
            return [DeleteIssue(run_dir=run_dir, phase="state_load", message=str(exc))]

        if not state.root_mlflow_run_id:
            return []
        if not state.mlflow_runtime_tracking_uri:
            return [
                DeleteIssue(
                    run_dir=run_dir,
                    phase="mlflow_runtime_contract",
                    message="pipeline_state.json has no mlflow_runtime_tracking_uri",
                )
            ]

        environment = MLflowEnvironment(
            state.mlflow_runtime_tracking_uri,
            ca_bundle_path=state.mlflow_ca_bundle_path,
        )
        try:
            environment.activate()
            deleted_ids.extend(self._delete_run_tree(state.root_mlflow_run_id))
            return []
        except Exception as exc:
            _LOG.exception("MLflow delete failed for %s", run_dir)
            return [DeleteIssue(run_dir=run_dir, phase="mlflow_delete", message=str(exc))]
        finally:
            environment.deactivate()

    def _delete_run_tree(self, root_run_id: str) -> list[str]:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        root_run = self._safe_get_run(client, root_run_id)
        if root_run is None:
            return []

        experiment_id = root_run.info.experiment_id
        queue: list[tuple[str, int]] = [(root_run_id, 0)]
        visited: set[str] = set()
        run_ids_with_depth: list[tuple[str, int]] = []

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            run_ids_with_depth.append((current_id, depth))
            for child_id in self._search_child_run_ids(client, experiment_id, current_id):
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

        ordered_run_ids = [
            run_id
            for run_id, _depth in sorted(run_ids_with_depth, key=lambda item: (item[1], item[0]), reverse=True)
        ]
        for run_id in ordered_run_ids:
            try:
                client.delete_run(run_id)
            except Exception as exc:
                if self._is_missing_run_error(exc):
                    continue
                raise
        return ordered_run_ids

    @staticmethod
    def _safe_get_run(client, run_id: str):
        try:
            return client.get_run(run_id)
        except Exception as exc:
            if RunDeleter._is_missing_run_error(exc):
                return None
            raise

    @staticmethod
    def _search_child_run_ids(client, experiment_id: str, parent_run_id: str) -> list[str]:
        try:
            children = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=f"tags.`mlflow.parentRunId` = '{parent_run_id}'",
            )
        except Exception as exc:
            if RunDeleter._is_missing_run_error(exc):
                return []
            raise
        return [child.info.run_id for child in children]

    @staticmethod
    def _is_missing_run_error(error: Exception) -> bool:
        message = str(error).lower()
        return "resource does not exist" in message or "not found" in message or "deleted" in message

    @staticmethod
    def _is_blocking_issue(issue: DeleteIssue) -> bool:
        return issue.phase != "mlflow_runtime_contract"


__all__ = ["DeleteIssue", "DeleteMode", "DeleteResult", "RunDeleter"]
