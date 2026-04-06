from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.tui.launch import (
    LaunchRequest,
    RestartPointOption,
    build_train_command,
    execute_launch_subprocess,
    interrupt_launch_process,
    load_restart_point_options,
    resolve_config_path_for_run,
    validate_resume_run,
)


def test_build_train_command_for_new_run(tmp_path: Path) -> None:
    request = LaunchRequest(
        mode="new_run",
        run_dir=tmp_path / "runs" / "new_run",
        config_path=tmp_path / "config.yaml",
    )

    command = build_train_command(request, python_executable="python3")

    assert command == [
        "python3",
        "-m",
        "src.main",
        "train",
        "--run-dir",
        str((tmp_path / "runs" / "new_run").resolve()),
        "--config",
        str((tmp_path / "config.yaml").resolve()),
    ]


def test_build_train_command_for_fresh(tmp_path: Path) -> None:
    request = LaunchRequest(
        mode="fresh",
        run_dir=tmp_path / "runs" / "fresh_run",
        config_path=tmp_path / "config.yaml",
    )

    command = build_train_command(request, python_executable="python3")

    assert command == [
        "python3",
        "-m",
        "src.main",
        "train",
        "--run-dir",
        str((tmp_path / "runs" / "fresh_run").resolve()),
        "--config",
        str((tmp_path / "config.yaml").resolve()),
    ]


def test_build_train_command_for_resume(tmp_path: Path) -> None:
    request = LaunchRequest(mode="resume", run_dir=tmp_path / "runs" / "resume_run")

    command = build_train_command(request, python_executable="python3")

    assert command == [
        "python3",
        "-m",
        "src.main",
        "train",
        "--run-dir",
        str((tmp_path / "runs" / "resume_run").resolve()),
        "--resume",
    ]


def test_build_train_command_for_restart(tmp_path: Path) -> None:
    request = LaunchRequest(
        mode="restart",
        run_dir=tmp_path / "runs" / "restart_run",
        restart_from_stage="Inference Deployer",
    )

    command = build_train_command(request, python_executable="python3")

    assert command == [
        "python3",
        "-m",
        "src.main",
        "train",
        "--run-dir",
        str((tmp_path / "runs" / "restart_run").resolve()),
        "--restart-from-stage",
        "Inference Deployer",
    ]


def test_resolve_config_path_for_run_reads_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.tui.launch._resolve_config_path_for_run",
        lambda run_dir, config_path=None: (tmp_path / "configs" / "pipeline.yaml").resolve(),
    )

    result = resolve_config_path_for_run(tmp_path / "runs" / "existing")

    assert result == (tmp_path / "configs" / "pipeline.yaml").resolve()


def test_load_restart_point_options_uses_config_and_restart_api(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.tui.launch._load_restart_point_options",
        lambda run_dir, config_path=None: (
            tmp_path / "cfg.yaml",
            [
                RestartPointOption(
                    stage="Inference Deployer",
                    available=True,
                    mode="fresh_or_resume",
                    reason="restart_allowed",
                ),
                RestartPointOption(
                    stage="Model Evaluator",
                    available=False,
                    mode="live_runtime_only",
                    reason="missing_inference_outputs",
                ),
            ],
        ),
    )

    config_path, points = load_restart_point_options(tmp_path / "runs" / "existing")

    assert config_path == tmp_path / "cfg.yaml"
    assert points == [
        RestartPointOption(
            stage="Inference Deployer",
            available=True,
            mode="fresh_or_resume",
            reason="restart_allowed",
        ),
        RestartPointOption(
            stage="Model Evaluator",
            available=False,
            mode="live_runtime_only",
            reason="missing_inference_outputs",
        ),
    ]


def test_validate_resume_run_rejects_completed_latest_attempt(tmp_path: Path, monkeypatch) -> None:
    state = MagicMock(
        training_critical_config_hash="train_hash",
        late_stage_config_hash="late_hash",
        model_dataset_config_hash="md_hash",
        pipeline_status="failed",
    )
    latest_attempt = MagicMock(
        status="completed",
        enabled_stage_names=["Dataset Validator"],
        stage_runs={"Dataset Validator": MagicMock(status="completed")},
    )
    state.attempts = [latest_attempt]

    monkeypatch.setattr("src.tui.adapters.launch_backend.load_pipeline_state", lambda _run_dir: state)
    monkeypatch.setattr(
        "src.tui.adapters.launch_backend.resolve_config_path_for_run",
        lambda run_dir, config_path=None: tmp_path / "cfg.yaml",
    )
    mock_config = MagicMock()
    monkeypatch.setattr("src.tui.adapters.launch_backend.load_config", lambda _path: mock_config)
    monkeypatch.setattr(
        "src.tui.adapters.launch_backend.compute_config_hashes",
        lambda _config: {"training_critical": "train_hash", "late_stage": "late_hash", "model_dataset": "md_hash"},
    )

    with pytest.raises(ValueError, match="Nothing to resume"):
        validate_resume_run(tmp_path / "runs" / "existing")


def test_execute_launch_subprocess_returns_readable_failure_tail(tmp_path: Path, monkeypatch) -> None:
    request = LaunchRequest(
        mode="fresh",
        run_dir=tmp_path / "runs" / "fresh_run",
        config_path=tmp_path / "config.yaml",
    )

    class DummyProcess:
        pid = 123

        def wait(self) -> int:
            log_path = request.run_dir.resolve() / "tui_launch.log"
            log_path.write_text(
                "bootstrap line\nPipeline failed: [CONFIG_DRIFT] training_critical config changed\n",
                encoding="utf-8",
            )
            return 1

    monkeypatch.setattr("src.tui.launch.subprocess.Popen", lambda *args, **kwargs: DummyProcess())

    result = execute_launch_subprocess(request, python_executable="python3")

    assert result.return_code == 1
    assert any("launcher_log=" in line for line in result.output_tail)
    assert any("Pipeline failed: [CONFIG_DRIFT]" in line for line in result.output_tail)


def test_execute_launch_subprocess_passes_log_level_via_environment(tmp_path: Path, monkeypatch) -> None:
    request = LaunchRequest(
        mode="fresh",
        run_dir=tmp_path / "runs" / "fresh_run",
        config_path=tmp_path / "config.yaml",
        log_level="DEBUG",
    )
    captured_env: dict[str, str] = {}

    class DummyProcess:
        pid = 456

        def wait(self) -> int:
            return 0

    def _fake_popen(*args, **kwargs):
        del args
        captured_env.update(kwargs["env"])
        return DummyProcess()

    monkeypatch.setattr("src.tui.launch.subprocess.Popen", _fake_popen)

    result = execute_launch_subprocess(request, python_executable="python3")

    assert result.return_code == 0
    assert captured_env["LOG_LEVEL"] == "DEBUG"
    assert any("LOG_LEVEL: DEBUG" in line for line in result.output_tail)


def test_interrupt_launch_process_uses_process_group_for_session_leader(monkeypatch) -> None:
    killpg_calls: list[tuple[int, int]] = []
    kill_calls: list[tuple[int, int]] = []

    monkeypatch.setattr("src.tui.launch.os.getpgid", lambda pid: pid)
    monkeypatch.setattr("src.tui.launch.os.killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    monkeypatch.setattr("src.tui.launch.os.kill", lambda pid, sig: kill_calls.append((pid, sig)))

    assert interrupt_launch_process(12345) is True
    assert len(killpg_calls) == 2
    assert kill_calls == []


def test_interrupt_launch_process_falls_back_to_single_pid_for_external_run(monkeypatch) -> None:
    killpg_calls: list[tuple[int, int]] = []
    kill_calls: list[tuple[int, int]] = []

    monkeypatch.setattr("src.tui.launch.os.getpgid", lambda _pid: 99999)
    monkeypatch.setattr("src.tui.launch.os.killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    monkeypatch.setattr("src.tui.launch.os.kill", lambda pid, sig: kill_calls.append((pid, sig)))

    assert interrupt_launch_process(12345) is True
    assert killpg_calls == []
    assert len(kill_calls) == 2


def test_ryotenkai_app_blocks_parallel_launches(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("textual")
    from src.tui.apps import RyotenkaiApp

    app = RyotenkaiApp(runs_dir=tmp_path)
    request = LaunchRequest(
        mode="fresh",
        run_dir=tmp_path / "runs" / "fresh_run",
        config_path=tmp_path / "config.yaml",
    )

    monkeypatch.setattr(app, "notify", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "open_attempt_for_run", lambda run_dir, attempt_no=None: None)
    monkeypatch.setattr(app, "_run_launch_worker", lambda _request: object())

    assert app.start_launch(request) is True
    assert app.active_launch is not None
    assert app.active_launch.status == "launching"
    assert app.start_launch(request) is False


def test_ryotenkai_app_uses_triple_notification_timeout(tmp_path: Path) -> None:
    pytest.importorskip("textual")
    from src.tui.apps import RyotenkaiApp

    app = RyotenkaiApp(runs_dir=tmp_path)

    assert app.NOTIFICATION_TIMEOUT == 9.0


def test_ryotenkai_app_opens_predicted_next_attempt_for_existing_run(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("textual")
    from src.tui.apps import RyotenkaiApp
    from src.pipeline.state.models import PipelineAttemptState, PipelineState, StageRunState

    app = RyotenkaiApp(runs_dir=tmp_path)
    request = LaunchRequest(
        mode="resume",
        run_dir=tmp_path / "runs" / "existing_run",
    )
    state = PipelineState(
        schema_version=1,
        logical_run_id="run_1",
        run_directory="runs/run_1",
        config_path="config.yaml",
        active_attempt_id=None,
        pipeline_status=StageRunState.STATUS_COMPLETED,
        training_critical_config_hash="",
        late_stage_config_hash="",
        attempts=[
            PipelineAttemptState(
                attempt_id="attempt-1",
                attempt_no=1,
                runtime_name="runtime",
                requested_action="fresh",
                effective_action="fresh",
                restart_from_stage=None,
                status=StageRunState.STATUS_COMPLETED,
                started_at="2026-03-21T00:00:00+00:00",
            ),
            PipelineAttemptState(
                attempt_id="attempt-2",
                attempt_no=2,
                runtime_name="runtime",
                requested_action="resume",
                effective_action="resume",
                restart_from_stage=None,
                status=StageRunState.STATUS_COMPLETED,
                started_at="2026-03-22T00:00:00+00:00",
            ),
        ],
    )
    store = MagicMock()
    store.load.return_value = state
    opened: list[tuple[Path, int | None]] = []

    monkeypatch.setattr("src.tui.adapters.state.PipelineStateStore", lambda _run_dir: store)
    monkeypatch.setattr(app, "notify", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "open_attempt_for_run", lambda run_dir, attempt_no=None: opened.append((run_dir, attempt_no)))
    monkeypatch.setattr(app, "_run_launch_worker", lambda _request: object())

    assert app.start_launch(request) is True
    assert opened == [(request.run_dir.resolve(), 3)]


def test_ryotenkai_app_opens_first_attempt_for_new_run_without_state(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("textual")
    from src.tui.apps import RyotenkaiApp

    app = RyotenkaiApp(runs_dir=tmp_path)
    request = LaunchRequest(
        mode="new_run",
        run_dir=tmp_path / "runs" / "new_run",
        config_path=tmp_path / "config.yaml",
    )
    opened: list[tuple[Path, int | None]] = []

    def _missing_store(_run_dir):
        store = MagicMock()
        store.load.side_effect = FileNotFoundError("missing state")
        return store

    monkeypatch.setattr("src.tui.adapters.state.PipelineStateStore", _missing_store)
    monkeypatch.setattr(app, "notify", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "open_attempt_for_run", lambda run_dir, attempt_no=None: opened.append((run_dir, attempt_no)))
    monkeypatch.setattr(app, "_run_launch_worker", lambda _request: object())

    assert app.start_launch(request) is True
    assert opened == [(request.run_dir.resolve(), 1)]


def test_ryotenkai_app_allows_parallel_new_run_when_another_run_is_active(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("textual")
    from src.tui.apps import RyotenkaiApp

    app = RyotenkaiApp(runs_dir=tmp_path)
    first_request = LaunchRequest(
        mode="fresh",
        run_dir=tmp_path / "runs" / "existing_run",
        config_path=tmp_path / "config.yaml",
    )
    second_request = LaunchRequest(
        mode="new_run",
        run_dir=tmp_path / "runs" / "new_run",
        config_path=tmp_path / "config.yaml",
    )

    monkeypatch.setattr(app, "notify", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "open_attempt_for_run", lambda run_dir, attempt_no=None: None)
    monkeypatch.setattr(app, "_run_launch_worker", lambda _request: object())

    assert app.start_launch(first_request) is True
    assert app.start_launch(second_request) is True
    assert app.get_active_launch_for_run(first_request.run_dir.resolve()) is not None
    assert app.get_active_launch_for_run(second_request.run_dir.resolve()) is not None


def test_ryotenkai_app_interrupts_active_launch(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("textual")
    from src.tui.apps import RyotenkaiApp
    from src.tui.launch import ActiveLaunch

    app = RyotenkaiApp(runs_dir=tmp_path)
    run_dir = (tmp_path / "runs" / "active_run").resolve()
    request = LaunchRequest(
        mode="resume",
        run_dir=run_dir,
    )
    app._active_launches[run_dir] = ActiveLaunch(
        request=request,
        command=("python3", "-m", "src.main", "train"),
        status="running",
        pid=43210,
    )

    monkeypatch.setattr("src.tui.apps.interrupt_launch_process", lambda pid: pid == 43210)
    monkeypatch.setattr(app, "notify", lambda *args, **kwargs: None)

    assert app.can_interrupt_run(run_dir) is True
    assert app.interrupt_active_launch_for_run(run_dir) is True
    assert app.active_launch is not None
    assert app.active_launch.status == "stopping"


def test_resolve_running_attempt_no_prefers_active_attempt_id() -> None:
    from src.pipeline.state.models import PipelineAttemptState, PipelineState, StageRunState
    from src.tui.adapters.state import find_running_attempt_no

    running_attempt = PipelineAttemptState(
        attempt_id="attempt-2",
        attempt_no=2,
        runtime_name="runtime",
        requested_action="resume",
        effective_action="resume",
        restart_from_stage=None,
        status=StageRunState.STATUS_RUNNING,
        started_at="2026-03-22T00:00:00+00:00",
    )
    older_attempt = PipelineAttemptState(
        attempt_id="attempt-1",
        attempt_no=1,
        runtime_name="runtime",
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=None,
        status=StageRunState.STATUS_COMPLETED,
        started_at="2026-03-21T00:00:00+00:00",
        completed_at="2026-03-21T01:00:00+00:00",
    )
    state = PipelineState(
        schema_version=1,
        logical_run_id="run_1",
        run_directory="runs/run_1",
        config_path="config.yaml",
        active_attempt_id="attempt-2",
        pipeline_status=StageRunState.STATUS_RUNNING,
        training_critical_config_hash="",
        late_stage_config_hash="",
        attempts=[older_attempt, running_attempt],
    )

    assert find_running_attempt_no(state) == 2


def test_resolve_running_attempt_no_falls_back_to_latest_running_attempt() -> None:
    from src.pipeline.state.models import PipelineAttemptState, PipelineState, StageRunState
    from src.tui.adapters.state import find_running_attempt_no

    completed_attempt = PipelineAttemptState(
        attempt_id="attempt-1",
        attempt_no=1,
        runtime_name="runtime",
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=None,
        status=StageRunState.STATUS_COMPLETED,
        started_at="2026-03-21T00:00:00+00:00",
        completed_at="2026-03-21T01:00:00+00:00",
    )
    running_attempt = PipelineAttemptState(
        attempt_id="attempt-3",
        attempt_no=3,
        runtime_name="runtime",
        requested_action="resume",
        effective_action="resume",
        restart_from_stage=None,
        status=StageRunState.STATUS_RUNNING,
        started_at="2026-03-22T00:00:00+00:00",
    )
    state = PipelineState(
        schema_version=1,
        logical_run_id="run_1",
        run_directory="runs/run_1",
        config_path="config.yaml",
        active_attempt_id=None,
        pipeline_status=StageRunState.STATUS_RUNNING,
        training_critical_config_hash="",
        late_stage_config_hash="",
        attempts=[completed_attempt, running_attempt],
    )

    assert find_running_attempt_no(state) == 3
