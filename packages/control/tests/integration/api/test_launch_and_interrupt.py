from __future__ import annotations

from pathlib import Path

import pytest

from src.api.services import launch_service
from src.pipeline import launch as pipeline_launch


def _fake_spawn(monkeypatch: pytest.MonkeyPatch, *, pid: int = 42424, command: tuple[str, ...] = ("python3",)) -> list[dict]:
    calls: list[dict] = []

    def _spawn(request, *, python_executable=None, extra_env=None, attach_stdio=False, **_ignored):  # type: ignore[no-untyped-def]
        calls.append({"request": request, "python": python_executable, "extra_env": extra_env, "attach_stdio": attach_stdio})
        launcher_log = request.run_dir / "tui_launch.log"
        launcher_log.parent.mkdir(parents=True, exist_ok=True)
        launcher_log.write_text("[fake] spawned\n", encoding="utf-8")
        return pid, command, launcher_log

    monkeypatch.setattr(launch_service, "spawn_launch", _spawn)
    return calls


def test_launch_spawns_and_returns_pid(client, seed_completed_run, monkeypatch, tmp_path: Path) -> None:
    seed_completed_run("run_launch_1")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model:\n  name: demo\n", encoding="utf-8")
    calls = _fake_spawn(monkeypatch)

    response = client.post(
        "/api/v1/runs/run_launch_1/launch",
        json={"mode": "fresh", "config_path": str(config_file)},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["pid"] == 42424
    assert body["launcher_log"].endswith("tui_launch.log")
    assert len(calls) == 1
    assert calls[0]["request"].mode == "fresh"


def test_launch_rejects_when_already_running(client, seed_running_run, monkeypatch) -> None:
    seed_running_run("run_launch_2", pid=1)  # pid=1 — init, always alive on UNIX
    monkeypatch.setattr(launch_service, "is_process_alive", lambda pid: True)
    calls = _fake_spawn(monkeypatch)

    response = client.post(
        "/api/v1/runs/run_launch_2/launch",
        json={"mode": "resume"},
    )
    assert response.status_code == 409
    assert calls == []


def test_launch_fresh_requires_config(client, seed_completed_run, monkeypatch) -> None:
    seed_completed_run("run_launch_3")
    calls = _fake_spawn(monkeypatch)

    response = client.post(
        "/api/v1/runs/run_launch_3/launch",
        json={"mode": "fresh"},
    )
    assert response.status_code == 422
    assert calls == []


def test_interrupt_stale_lock_removes_lock(client, seed_running_run, monkeypatch) -> None:
    run_dir = seed_running_run("run_launch_4", pid=123456)
    monkeypatch.setattr(launch_service, "is_process_alive", lambda pid: False)

    response = client.post("/api/v1/runs/run_launch_4/interrupt")
    assert response.status_code == 200
    body = response.json()
    assert body["interrupted"] is False
    assert body["reason"] == "process_not_found"
    assert not (run_dir / "run.lock").exists()


def test_interrupt_running_sends_signal(client, seed_running_run, monkeypatch) -> None:
    seed_running_run("run_launch_5", pid=7777)
    monkeypatch.setattr(launch_service, "is_process_alive", lambda pid: True)
    called = {}

    def _interrupt(pid: int) -> bool:
        called["pid"] = pid
        return True

    monkeypatch.setattr(launch_service, "interrupt_launch_process", _interrupt)

    response = client.post("/api/v1/runs/run_launch_5/interrupt")
    assert response.status_code == 200
    body = response.json()
    assert body["interrupted"] is True
    assert body["pid"] == 7777
    assert called == {"pid": 7777}


def test_interrupt_no_lock_returns_false(client, seed_completed_run) -> None:
    seed_completed_run("run_launch_6")
    response = client.post("/api/v1/runs/run_launch_6/interrupt")
    assert response.status_code == 200
    assert response.json()["reason"] == "no_lock_file"


def test_launch_inside_project_workspace_passes_RYOTENKAI_env_vars(
    client, seed_completed_run, monkeypatch, tmp_path: Path,
) -> None:
    """Regression for the previous "Web-launched runs lose metadata" bug.

    A run inside a project workspace must spawn with
    ``RYOTENKAI_PROJECT_ID/ACTOR/CONFIG_VERSION_HASH/RUNS_BASE_DIR``
    in extra_env so the worker's bootstrap can stamp them onto state
    and MLflow tags.
    """
    from src.workspace.projects.adapter import ResolvedProject

    seed_completed_run("run_proj")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model:\n  name: x\n", encoding="utf-8")

    # Stub the resolver: pretend run_dir is inside a project.
    project_runs_base = tmp_path / "proj" / "runs"
    project_runs_base.mkdir(parents=True)
    fake_resolved = ResolvedProject(
        config_path=tmp_path / "proj" / "configs" / "current.yaml",
        runs_base_dir=project_runs_base,
        env={"HF_TOKEN": "tok-from-env-json"},
        metadata={
            "project_id": "myproj",
            "actor": "agent:web-ui",
            "config_version_hash": "deadbeef",
        },
    )

    def _stub_resolver(run_dir, registry=None):
        return fake_resolved

    monkeypatch.setattr(
        launch_service,
        "resolve_project_launch_inputs_from_run_dir",
        _stub_resolver,
    )

    calls = _fake_spawn(monkeypatch)

    response = client.post(
        "/api/v1/runs/run_proj/launch",
        json={"mode": "fresh", "config_path": str(config_file)},
    )
    assert response.status_code == 202, response.text
    assert len(calls) == 1
    extra_env = calls[0]["extra_env"]
    assert extra_env["RYOTENKAI_PROJECT_ID"] == "myproj"
    assert extra_env["RYOTENKAI_ACTOR"] == "agent:web-ui"
    assert extra_env["RYOTENKAI_CONFIG_VERSION_HASH"] == "deadbeef"
    assert extra_env["RYOTENKAI_RUNS_BASE_DIR"] == str(project_runs_base)
    # env.json keys come through too.
    assert extra_env["HF_TOKEN"] == "tok-from-env-json"


def test_launch_outside_project_yields_empty_extra_env(
    client, seed_completed_run, monkeypatch, tmp_path: Path,
) -> None:
    """Ad-hoc run (not inside any project workspace) → no RYOTENKAI_*
    env vars injected. Worker treats absence as anonymous run."""
    seed_completed_run("run_adhoc")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model:\n  name: demo\n", encoding="utf-8")

    # Stub resolver to return None (ad-hoc, no project).
    monkeypatch.setattr(
        launch_service,
        "resolve_project_launch_inputs_from_run_dir",
        lambda run_dir, registry=None: None,
    )
    calls = _fake_spawn(monkeypatch)

    response = client.post(
        "/api/v1/runs/run_adhoc/launch",
        json={"mode": "fresh", "config_path": str(config_file)},
    )
    assert response.status_code == 202
    assert len(calls) == 1
    extra_env = calls[0]["extra_env"]
    assert "RYOTENKAI_PROJECT_ID" not in extra_env
    assert "RYOTENKAI_ACTOR" not in extra_env


def test_pipeline_launch_read_lock_pid_parses_valid(tmp_path: Path) -> None:
    (tmp_path / "run.lock").write_text("pid=4242\nstarted_at=2026\n", encoding="utf-8")
    assert pipeline_launch.read_lock_pid(tmp_path) == 4242


def test_pipeline_launch_read_lock_pid_missing(tmp_path: Path) -> None:
    assert pipeline_launch.read_lock_pid(tmp_path) is None
