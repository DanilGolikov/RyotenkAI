from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.pipeline.run_deletion import RunDeletionMode, RunDeletionService
from src.pipeline.state import PipelineStateStore


def _write_state(run_dir: Path, *, config_path: Path, root_mlflow_run_id: str | None) -> None:
    PipelineStateStore(run_dir).init_state(
        logical_run_id=run_dir.name,
        config_path=str(config_path),
        training_critical_config_hash="train_hash",
        late_stage_config_hash="late_hash",
        root_mlflow_run_id=root_mlflow_run_id,
    )


def _fake_config():
    mlflow_cfg = SimpleNamespace()
    return SimpleNamespace(experiment_tracking=SimpleNamespace(mlflow=mlflow_cfg))


def test_delete_target_removes_remote_then_local(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("stub", encoding="utf-8")
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    (run_dir / "payload.txt").write_text("hello", encoding="utf-8")
    _write_state(run_dir, config_path=config_path, root_mlflow_run_id="root_1")
    calls: list[tuple[str, object]] = []

    class _Manager:
        def setup(self, **kwargs) -> bool:
            calls.append(("setup", kwargs))
            return True

        def delete_run_tree(self, root_run_id: str) -> list[str]:
            calls.append(("delete_run_tree", root_run_id))
            return ["child_1", "root_1"]

        def get_runtime_tracking_uri(self) -> str:
            return "http://localhost:5002"

        def cleanup(self) -> None:
            calls.append(("cleanup", None))

    service = RunDeletionService(
        config_loader=lambda _path: _fake_config(),
        mlflow_manager_factory=lambda _config: _Manager(),
    )

    result = service.delete_target(run_dir)

    assert result.is_success is True
    assert result.local_deleted is True
    assert not run_dir.exists()
    assert calls == [
        ("setup", {"timeout": 5.0, "max_retries": 1, "disable_system_metrics": True}),
        ("delete_run_tree", "root_1"),
        ("cleanup", None),
    ]


def test_delete_target_local_only_skips_mlflow_delete(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("stub", encoding="utf-8")
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    _write_state(run_dir, config_path=config_path, root_mlflow_run_id="root_1")

    class _Manager:
        def setup(self, **kwargs) -> bool:
            raise AssertionError("setup should not be used for local-only delete")

        def delete_run_tree(self, root_run_id: str) -> list[str]:
            raise AssertionError("delete_run_tree should not be used for local-only delete")

        def cleanup(self) -> None:
            raise AssertionError("cleanup should not be used for local-only delete")

    service = RunDeletionService(
        config_loader=lambda _path: _fake_config(),
        mlflow_manager_factory=lambda _config: _Manager(),
    )

    result = service.delete_target(run_dir, mode=RunDeletionMode.LOCAL_ONLY)

    assert result.is_success is True
    assert result.local_deleted is True
    assert not run_dir.exists()
    assert result.deleted_mlflow_run_ids == ()
    assert result.issues == ()


def test_delete_group_directory_handles_multiple_run_dirs(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("stub", encoding="utf-8")
    group_dir = tmp_path / "smoke_group"
    run_a = group_dir / "run_a"
    run_b = group_dir / "nested" / "run_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    _write_state(run_a, config_path=config_path, root_mlflow_run_id=None)
    _write_state(run_b, config_path=config_path, root_mlflow_run_id=None)

    service = RunDeletionService(
        config_loader=lambda _path: _fake_config(),
        mlflow_manager_factory=lambda _config: (_ for _ in ()).throw(AssertionError("manager should not be used")),
    )

    result = service.delete_target(group_dir)

    assert result.is_success is True
    assert result.run_dirs == (run_a, run_b)
    assert not group_dir.exists()
