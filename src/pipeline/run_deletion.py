from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.infrastructure.mlflow.gc import IMLflowGcAdapter, MLflowCliGcAdapter, MLflowGcConfig
from src.pipeline.state import PipelineStateStore
from src.training.managers.mlflow_manager import MLflowManager
from src.utils.config import PipelineConfig, load_config


@dataclass(frozen=True, slots=True)
class RunDeletionIssue:
    run_dir: Path
    phase: str
    message: str


@dataclass(frozen=True, slots=True)
class RunDeletionResult:
    target: Path
    run_dirs: tuple[Path, ...]
    deleted_mlflow_run_ids: tuple[str, ...]
    local_deleted: bool
    issues: tuple[RunDeletionIssue, ...]

    @property
    def is_success(self) -> bool:
        return not self.issues


class RunDeletionService:
    """Delete local run directories together with their linked MLflow runs."""

    def __init__(
        self,
        *,
        config_loader: Callable[[Path], PipelineConfig] = load_config,
        mlflow_manager_factory: Callable[[PipelineConfig], MLflowManager] | None = None,
        gc_adapter: IMLflowGcAdapter | None = None,
        rmtree: Callable[[Path], None] = shutil.rmtree,
    ) -> None:
        self._config_loader = config_loader
        self._mlflow_manager_factory = mlflow_manager_factory or (
            lambda config: MLflowManager(config, runtime_role="control_plane")
        )
        self._gc_adapter = gc_adapter or MLflowCliGcAdapter()
        self._rmtree = rmtree

    def delete_target(self, target: Path) -> RunDeletionResult:
        resolved_target = target.expanduser().resolve()
        run_dirs = tuple(self._discover_run_dirs(resolved_target))
        deleted_mlflow_run_ids: list[str] = []
        issues: list[RunDeletionIssue] = []

        for run_dir in run_dirs:
            issues.extend(self._delete_remote_state_for_run(run_dir, deleted_mlflow_run_ids))

        if issues:
            return RunDeletionResult(
                target=resolved_target,
                run_dirs=run_dirs,
                deleted_mlflow_run_ids=tuple(deleted_mlflow_run_ids),
                local_deleted=False,
                issues=tuple(issues),
            )

        try:
            self._rmtree(resolved_target)
        except Exception as exc:
            issues.append(RunDeletionIssue(run_dir=resolved_target, phase="filesystem_delete", message=str(exc)))

        return RunDeletionResult(
            target=resolved_target,
            run_dirs=run_dirs,
            deleted_mlflow_run_ids=tuple(deleted_mlflow_run_ids),
            local_deleted=not issues,
            issues=tuple(issues),
        )

    def _delete_remote_state_for_run(self, run_dir: Path, deleted_mlflow_run_ids: list[str]) -> list[RunDeletionIssue]:
        try:
            state = PipelineStateStore(run_dir).load()
        except Exception as exc:
            return [RunDeletionIssue(run_dir=run_dir, phase="state_load", message=str(exc))]

        root_run_id = state.root_mlflow_run_id
        if not root_run_id:
            return []

        config_path = Path(state.config_path).expanduser().resolve()
        try:
            config = self._config_loader(config_path)
        except Exception as exc:
            return [RunDeletionIssue(run_dir=run_dir, phase="config_load", message=str(exc))]

        manager = self._mlflow_manager_factory(config)
        try:
            if not manager.setup(timeout=5.0, max_retries=1, disable_system_metrics=True):
                return [RunDeletionIssue(run_dir=run_dir, phase="mlflow_setup", message="MLflow setup failed")]
            try:
                run_tree_ids = manager.delete_run_tree(root_run_id)
            except Exception as exc:
                return [RunDeletionIssue(run_dir=run_dir, phase="mlflow_delete", message=str(exc))]

            try:
                gc_config = MLflowGcConfig.from_runtime(
                    config=config.experiment_tracking.mlflow,
                    tracking_uri=manager.get_runtime_tracking_uri(),
                )
            except Exception as exc:
                return [RunDeletionIssue(run_dir=run_dir, phase="mlflow_gc_config", message=str(exc))]

            try:
                self._gc_adapter.hard_delete_runs(run_tree_ids, config=gc_config)
            except Exception as exc:
                return [RunDeletionIssue(run_dir=run_dir, phase="mlflow_gc", message=str(exc))]

            deleted_mlflow_run_ids.extend(run_tree_ids)
            return []
        finally:
            manager.cleanup()

    def _discover_run_dirs(self, target: Path) -> list[Path]:
        if not target.exists():
            return []
        if (target / "pipeline_state.json").exists():
            return [target]

        run_dirs: list[Path] = []
        for current in sorted(target.rglob("*"), key=lambda path: (len(path.relative_to(target).parts), path.as_posix())):
            if current.is_dir() and (current / "pipeline_state.json").exists():
                run_dirs.append(current)
        return run_dirs


__all__ = ["RunDeletionIssue", "RunDeletionResult", "RunDeletionService"]
