from __future__ import annotations

from pathlib import Path

from src.pipeline.state import PipelineStateStore
from src.tui.adapters.delete_backend import DeleteMode, TuiDeleteBackend


def _write_state(
    run_dir: Path,
    *,
    config_path: Path,
    root_mlflow_run_id: str | None,
    mlflow_runtime_tracking_uri: str | None = "http://localhost:5002",
) -> None:
    PipelineStateStore(run_dir).init_state(
        logical_run_id=run_dir.name,
        config_path=str(config_path),
        training_critical_config_hash="train_hash",
        late_stage_config_hash="late_hash",
        root_mlflow_run_id=root_mlflow_run_id,
        mlflow_runtime_tracking_uri=mlflow_runtime_tracking_uri,
    )


def test_delete_target_removes_remote_then_local(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("stub", encoding="utf-8")
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    (run_dir / "payload.txt").write_text("hello", encoding="utf-8")
    _write_state(run_dir, config_path=config_path, root_mlflow_run_id="root_1")
    calls: list[str] = []
    backend = TuiDeleteBackend()
    monkeypatch.setattr(
        backend,
        "_delete_run_tree",
        lambda root_run_id: calls.append(root_run_id) or ["child_1", "root_1"],
    )

    result = backend.delete_target(run_dir)

    assert result.is_success is True
    assert result.local_deleted is True
    assert not run_dir.exists()
    assert calls == ["root_1"]


def test_delete_target_local_only_skips_mlflow_delete(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("stub", encoding="utf-8")
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    _write_state(run_dir, config_path=config_path, root_mlflow_run_id="root_1")

    backend = TuiDeleteBackend()

    result = backend.delete_target(run_dir, mode=DeleteMode.LOCAL_ONLY)

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

    backend = TuiDeleteBackend()

    result = backend.delete_target(group_dir)

    assert result.is_success is True
    assert result.run_dirs == (run_a, run_b)
    assert not group_dir.exists()


def test_delete_target_reports_missing_runtime_contract(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("stub", encoding="utf-8")
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    _write_state(
        run_dir,
        config_path=config_path,
        root_mlflow_run_id="root_1",
        mlflow_runtime_tracking_uri=None,
    )

    result = TuiDeleteBackend().delete_target(run_dir)

    assert result.local_deleted is True
    assert not run_dir.exists()
    assert result.issues[0].phase == "mlflow_runtime_contract"
