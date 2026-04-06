from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class MLflowGcConfig:
    backend_store_uri: str
    artifacts_destination: str
    tracking_uri: str

    @classmethod
    def from_runtime(
        cls,
        *,
        config: object,
        tracking_uri: str,
    ) -> "MLflowGcConfig":
        backend_store_uri = (
            getattr(config, "gc_backend_store_uri", None)
            or os.getenv("MLFLOW_GC_BACKEND_STORE_URI")
            or os.getenv("BACKEND_STORE_URI")
        )
        artifacts_destination = (
            getattr(config, "gc_artifacts_destination", None)
            or os.getenv("MLFLOW_GC_ARTIFACTS_DESTINATION")
            or os.getenv("ARTIFACTS_DESTINATION")
        )
        if not tracking_uri:
            raise ValueError("MLflow GC requires a non-empty tracking URI")
        if not backend_store_uri or not artifacts_destination:
            raise ValueError(
                "MLflow hard-delete requires gc_backend_store_uri and gc_artifacts_destination "
                "(or equivalent environment variables)"
            )
        return cls(
            backend_store_uri=str(backend_store_uri),
            artifacts_destination=str(artifacts_destination),
            tracking_uri=tracking_uri,
        )


class IMLflowGcAdapter(Protocol):
    def hard_delete_runs(self, run_ids: list[str], *, config: MLflowGcConfig) -> None:
        """Permanently remove previously soft-deleted runs and their artifacts."""


class MLflowCliGcAdapter:
    """Hard-delete MLflow runs via the official `mlflow gc` CLI."""

    def hard_delete_runs(self, run_ids: list[str], *, config: MLflowGcConfig) -> None:
        normalized_run_ids = [run_id.strip() for run_id in run_ids if run_id and run_id.strip()]
        if not normalized_run_ids:
            return
        command = [
            sys.executable,
            "-m",
            "mlflow",
            "gc",
            "--backend-store-uri",
            config.backend_store_uri,
            "--artifacts-destination",
            config.artifacts_destination,
            "--tracking-uri",
            config.tracking_uri,
            "--run-ids",
            ",".join(normalized_run_ids),
        ]
        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = config.tracking_uri
        completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
        if completed.returncode == 0:
            return
        details = completed.stderr.strip() or completed.stdout.strip() or f"exit_code={completed.returncode}"
        raise RuntimeError(f"mlflow gc failed: {details}")


__all__ = ["IMLflowGcAdapter", "MLflowCliGcAdapter", "MLflowGcConfig"]
