#!/usr/bin/env python3
"""
Test MLflow Stack (PostgreSQL + MinIO)
Verifies artifacts are logged correctly via the REST API.

Important:
- This test requires a running MLflow server (and backing services) at TRACKING_URI.
- When the stack is down, MLflow client performs long HTTP retries, which looks like "hang".
  We do a fast port check and skip (or fail fast if explicitly required).
"""

import json
import os
import socket
import sys
import tempfile
from urllib.parse import urlparse

import mlflow

TRACKING_URI = "http://localhost:5002"


def _is_port_open(uri: str, *, timeout_seconds: float = 0.2) -> bool:
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def _mlflow_stack_check() -> bool:
    """Run MLflow/MinIO stack check. Returns True on success, False on failure."""

    print("🧪 Testing MLflow Stack...")
    print("")

    # Setup MLflow client — tracking_uri only
    tracking_uri = TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    print(f"✅ Tracking URI: {tracking_uri}")

    # Create / reuse experiment
    # NOTE: MLflow can keep experiments in a deleted state; handle that for repeatable tests.
    experiment_name = "test_minio_stack"
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        exp = client.get_experiment_by_name(experiment_name)
        if exp and exp.lifecycle_stage == "deleted":
            client.restore_experiment(exp.experiment_id)
    except Exception:
        # If anything goes wrong, fall back to mlflow.set_experiment which will raise
        pass

    mlflow.set_experiment(experiment_name)
    print(f"✅ Experiment: {experiment_name}")
    print("")

    # Start run
    with mlflow.start_run(run_name="test_from_mac") as run:
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri

        print(f"📊 Run ID: {run_id}")
        print(f"📦 Artifact URI: {artifact_uri}")
        print("")

        # Check artifact URI scheme
        if "mlflow-artifacts" in artifact_uri:
            print("✅ Uses mlflow-artifacts:// scheme (MinIO via REST API)")
        elif artifact_uri.startswith("s3://"):
            print("✅ Uses s3:// scheme (MinIO via boto3)")
        elif artifact_uri.startswith("/"):
            print("❌ FAIL: Still uses local filesystem path!")
            return False
        else:
            print(f"⚠️  Unknown scheme: {artifact_uri}")

        print("")

        # Log parameters
        mlflow.log_param("test_param", "value123")
        mlflow.log_param("environment", "development")
        print("✅ Logged parameters")

        # Log metrics
        mlflow.log_metric("accuracy", 0.95)
        mlflow.log_metric("loss", 0.05)
        print("✅ Logged metrics")

        # Log artifact (MAIN TEST!)
        print("")
        print("📤 Logging artifact to MinIO...")

        test_data = {
            "status": "success",
            "message": "MinIO + PostgreSQL stack works!",
            "artifact_uri": artifact_uri,
            "run_id": run_id,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f, indent=2)
            temp_path = f.name

        try:
            mlflow.log_artifact(temp_path, artifact_path="test_results")
            print("✅ Artifact logged successfully via REST API!")
        except Exception as e:
            print(f"❌ FAIL: Failed to log artifact: {e}")
            return False
        finally:
            import os

            os.remove(temp_path)

        print("")
        print("─" * 60)
        print("🎉 SUCCESS! MLflow + PostgreSQL + MinIO stack is working!")
        print("─" * 60)
        print("")
        print("🌐 Check results:")
        print("   MLflow UI:    http://localhost:5002")
        print("   MinIO UI:     http://localhost:9001")
        print("   Experiment:   http://localhost:5002/#/experiments/2")
        print(f"   Run:          http://localhost:5002/#/experiments/2/runs/{run_id}")
        print("")

        return True


def test_mlflow_stack() -> None:
    """Test artifact logging to MinIO via MLflow REST API."""
    import pytest

    if not _is_port_open(TRACKING_URI):
        msg = (
            f"MLflow stack is not reachable at {TRACKING_URI}. "
            "Start the stack to run this test."
        )
        if os.getenv("REQUIRE_MLFLOW_STACK", "").lower() in {"1", "true", "yes"}:
            pytest.fail(msg)
        pytest.skip(msg)
    assert _mlflow_stack_check() is True


if __name__ == "__main__":
    try:
        success = _mlflow_stack_check()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
