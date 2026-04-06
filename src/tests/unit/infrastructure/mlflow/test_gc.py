from __future__ import annotations

import sys
from subprocess import CompletedProcess

import pytest

from src.infrastructure.mlflow.gc import MLflowCliGcAdapter, MLflowGcConfig


def test_mlflow_gc_config_uses_explicit_fields_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_GC_BACKEND_STORE_URI", raising=False)
    monkeypatch.delenv("MLFLOW_GC_ARTIFACTS_DESTINATION", raising=False)
    cfg = type(
        "Cfg",
        (),
        {
            "gc_backend_store_uri": "postgresql://mlflow:pass@localhost:5432/mlflow_db",
            "gc_artifacts_destination": "s3://mlflow",
        },
    )()

    resolved = MLflowGcConfig.from_runtime(config=cfg, tracking_uri="http://localhost:5002")

    assert resolved.backend_store_uri == "postgresql://mlflow:pass@localhost:5432/mlflow_db"
    assert resolved.artifacts_destination == "s3://mlflow"
    assert resolved.tracking_uri == "http://localhost:5002"


def test_mlflow_gc_adapter_invokes_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = MLflowCliGcAdapter()
    observed: dict[str, object] = {}

    def fake_run(command, check, capture_output, text, env):
        observed["command"] = command
        observed["check"] = check
        observed["capture_output"] = capture_output
        observed["text"] = text
        observed["env_tracking_uri"] = env["MLFLOW_TRACKING_URI"]
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("src.infrastructure.mlflow.gc.subprocess.run", fake_run)

    adapter.hard_delete_runs(
        ["run_a", "run_b"],
        config=MLflowGcConfig(
            backend_store_uri="postgresql://mlflow:pass@localhost:5432/mlflow_db",
            artifacts_destination="s3://mlflow",
            tracking_uri="http://localhost:5002",
        ),
    )

    assert observed["check"] is False
    assert observed["capture_output"] is True
    assert observed["text"] is True
    assert observed["env_tracking_uri"] == "http://localhost:5002"
    assert observed["command"] == [
        sys.executable,
        "-m",
        "mlflow",
        "gc",
        "--backend-store-uri",
        "postgresql://mlflow:pass@localhost:5432/mlflow_db",
        "--artifacts-destination",
        "s3://mlflow",
        "--tracking-uri",
        "http://localhost:5002",
        "--run-ids",
        "run_a,run_b",
    ]


def test_mlflow_gc_adapter_raises_on_cli_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = MLflowCliGcAdapter()

    monkeypatch.setattr(
        "src.infrastructure.mlflow.gc.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args[0], 1, stdout="", stderr="boom"),
    )

    with pytest.raises(RuntimeError, match="mlflow gc failed: boom"):
        adapter.hard_delete_runs(
            ["run_a"],
            config=MLflowGcConfig(
                backend_store_uri="postgresql://mlflow:pass@localhost:5432/mlflow_db",
                artifacts_destination="s3://mlflow",
                tracking_uri="http://localhost:5002",
            ),
        )
