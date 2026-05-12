"""L6 smoke: fake-mlflow sidecar REST surface.

Covers a single end-to-end run lifecycle (create → log-metric → update)
and the chaos surface that flips the sidecar into "MLflow unavailable"
mode, mapping to a 503 on subsequent calls.
"""

from __future__ import annotations

import httpx
import pytest

from tests._harness.stack import Stack

pytestmark = [pytest.mark.asyncio, pytest.mark.stack]


async def test_create_run_via_mlflow_sidecar(stack: Stack) -> None:
    mlflow = stack.mlflow_url
    async with httpx.AsyncClient(timeout=5.0) as client:
        # /api/setup activates the in-memory store. is_active flips to True
        # and a fake tracking URI is published.
        setup = await client.post(mlflow + "/api/setup")
        assert setup.status_code == 200
        body = setup.json()
        assert body["is_active"] is True
        assert body["tracking_uri"] == "fake://in-memory"

        # Create a run, log a metric, finish it.
        create = await client.post(
            mlflow + "/api/2.0/mlflow/runs/create",
            json={"run_name": "smoke", "description": "L6 smoke run"},
        )
        assert create.status_code == 200
        run_id = create.json()["run"]["info"]["run_id"]
        assert run_id

        log = await client.post(
            mlflow + "/api/2.0/mlflow/runs/log-metric",
            json={"run_id": run_id, "key": "loss", "value": 0.5, "step": 1},
        )
        assert log.status_code == 200

        update = await client.post(
            mlflow + "/api/2.0/mlflow/runs/update",
            json={"run_id": run_id, "status": "FINISHED"},
        )
        assert update.status_code == 200
        assert update.json() == {"status": "FINISHED"}

        # /control/state should reflect the run with its metric.
        state = await client.get(mlflow + "/control/state")
        assert state.status_code == 200
        snapshot = state.json()
        assert run_id in snapshot["runs"]
        recorded_metrics = snapshot["runs"][run_id]["metrics"]
        assert any(m["key"] == "loss" and m["value"] == 0.5 for m in recorded_metrics)


async def test_mlflow_unavailable_returns_503(stack: Stack) -> None:
    mlflow = stack.mlflow_url
    async with httpx.AsyncClient(timeout=5.0) as client:
        # Flip the sidecar into "unavailable" mode.
        flip = await client.post(mlflow + "/control/set_unavailable", params={"value": "true"})
        assert flip.status_code == 200
        assert flip.json() == {"unavailable": True}

        # Any run-creation attempt must now return 503 with the
        # MLflowUnavailableError mapping.
        create = await client.post(
            mlflow + "/api/2.0/mlflow/runs/create",
            json={"run_name": "should-fail"},
        )
        assert create.status_code == 503
        assert create.json()["error"] == "unavailable"
