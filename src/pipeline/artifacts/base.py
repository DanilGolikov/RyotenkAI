"""StageArtifactEnvelope, StageArtifactCollector and save_stage_artifact helper."""

from __future__ import annotations

import contextlib
import json
import logging
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Artifact status values
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_INTERRUPTED = "interrupted"


@dataclass
class StageArtifactEnvelope:
    """Unified envelope written by every pipeline stage as a JSON artifact.

    Fields:
        stage: snake_case stage identifier, e.g. "dataset_validator"
        status: one of passed / failed / skipped / interrupted
        started_at: ISO-8601 timestamp of stage start
        duration_seconds: wall-clock seconds from start to finish
        error: human-readable error message when status=="failed", else None
        data: stage-specific payload, typed via TypedDict schemas
    """

    stage: str
    status: str
    started_at: str
    duration_seconds: float
    error: str | None
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageArtifactEnvelope:
        return cls(
            stage=d.get("stage", ""),
            status=d.get("status", STATUS_PASSED),
            started_at=d.get("started_at", ""),
            duration_seconds=float(d.get("duration_seconds", 0.0)),
            error=d.get("error"),
            data=d.get("data", {}),
        )


class StageArtifactCollector:
    """Accumulates stage data during execution and flushes to a JSON envelope.

    Usage (simple stages — data comes from context after stage.run()):
        collector.put(upload_duration_seconds=33.4, deps_duration_seconds=7.1)
        collector.flush_ok(started_at=ts, duration_seconds=402.1, context=ctx)

    Usage (complex stages with callbacks — e.g. DatasetValidator):
        # In callbacks:
        collector.append("datasets", dataset_entry)
        # On completion callback:
        collector.flush_ok(started_at=ts, duration_seconds=1.3, context=ctx)

    The collector is flushed exactly once. After flush, is_flushed == True and
    the orchestrator skips automatic flush for that stage.
    """

    def __init__(self, stage: str, artifact_name: str) -> None:
        self._stage = stage
        self._artifact_name = artifact_name
        self._data: dict[str, Any] = {}
        self._is_flushed = False
        self._started_at: str | None = None

    @property
    def is_flushed(self) -> bool:
        return self._is_flushed

    @property
    def artifact_name(self) -> str:
        return self._artifact_name

    def put(self, **kwargs: Any) -> None:
        """Add/update flat key-value pairs in the data payload."""
        self._data.update(kwargs)

    def append(self, key: str, item: Any) -> None:
        """Append an item to a list in the data payload."""
        self._data.setdefault(key, []).append(item)

    def set_started_at(self, started_at: str) -> None:
        """Record stage start timestamp (called at stage loop start)."""
        self._started_at = started_at

    def flush_ok(
        self,
        *,
        started_at: str,
        duration_seconds: float,
        context: dict[str, Any],
    ) -> StageArtifactEnvelope | None:
        """Flush as passed. Writes artifact via save_stage_artifact."""
        if self._is_flushed:
            return None
        envelope = StageArtifactEnvelope(
            stage=self._stage,
            status=STATUS_PASSED,
            started_at=started_at,
            duration_seconds=round(duration_seconds, 3),
            error=None,
            data=self._data,
        )
        self._is_flushed = True
        save_stage_artifact(context, envelope, self._artifact_name)
        return envelope

    def flush_error(
        self,
        *,
        error: str,
        started_at: str,
        duration_seconds: float,
        context: dict[str, Any],
    ) -> StageArtifactEnvelope | None:
        """Flush as failed with an error message."""
        if self._is_flushed:
            return None
        envelope = StageArtifactEnvelope(
            stage=self._stage,
            status=STATUS_FAILED,
            started_at=started_at,
            duration_seconds=round(duration_seconds, 3),
            error=error,
            data=self._data,
        )
        self._is_flushed = True
        save_stage_artifact(context, envelope, self._artifact_name)
        return envelope

    def flush_interrupted(
        self,
        *,
        started_at: str,
        duration_seconds: float,
        context: dict[str, Any],
    ) -> StageArtifactEnvelope | None:
        """Flush as interrupted (SIGINT / unhandled exception in finally block)."""
        if self._is_flushed:
            return None
        envelope = StageArtifactEnvelope(
            stage=self._stage,
            status=STATUS_INTERRUPTED,
            started_at=started_at,
            duration_seconds=round(duration_seconds, 3),
            error=None,
            data=self._data,
        )
        self._is_flushed = True
        save_stage_artifact(context, envelope, self._artifact_name)
        return envelope

    def flush_skipped(
        self,
        *,
        started_at: str,
        context: dict[str, Any],
    ) -> StageArtifactEnvelope | None:
        """Flush as skipped (stage disabled in config)."""
        if self._is_flushed:
            return None
        envelope = StageArtifactEnvelope(
            stage=self._stage,
            status=STATUS_SKIPPED,
            started_at=started_at,
            duration_seconds=0.0,
            error=None,
            data={},
        )
        self._is_flushed = True
        save_stage_artifact(context, envelope, self._artifact_name)
        return envelope


def save_stage_artifact(
    context: dict[str, Any],
    envelope: StageArtifactEnvelope,
    artifact_name: str,
    artifact_path: str = "",
) -> None:
    """Write envelope JSON to MLflow as a run artifact (best-effort).

    Uses the MLflowManager instance from pipeline context to reuse the
    already-open MLflow run without creating a new connection.

    Args:
        context: pipeline context dict (must contain MLFLOW_MANAGER and
            MLFLOW_PARENT_RUN_ID keys)
        envelope: the envelope to serialise
        artifact_name: filename for the artifact, e.g.
            "dataset_validator_results.json"
        artifact_path: optional subfolder within the MLflow artifact store,
            e.g. "evaluation" → "evaluation/sub_results.json"
    """
    try:
        from src.pipeline.stages.constants import PipelineContextKeys
        from src.training.managers.mlflow_manager import MLflowManager

        mlflow_mgr: Any = context.get(PipelineContextKeys.MLFLOW_MANAGER)
        run_id: Any = context.get(PipelineContextKeys.MLFLOW_PARENT_RUN_ID)

        if not isinstance(mlflow_mgr, MLflowManager) or not mlflow_mgr.is_enabled:
            logger.debug(
                "[ARTIFACT] MLflowManager not available — skipping artifact write for %s",
                artifact_name,
            )
            return

        if not isinstance(run_id, str) or not run_id:
            logger.debug(
                "[ARTIFACT] No MLflow run_id in context — skipping artifact write for %s",
                artifact_name,
            )
            return

        payload = json.dumps(envelope.to_dict(), ensure_ascii=False, indent=2)

        # Write to a temp dir with the exact artifact_name so MLflow stores it
        # under the correct filename. NamedTemporaryFile adds a random prefix
        # which becomes the artifact name in MLflow (tmp2u1d_xxx.json).
        tmp_dir = Path(tempfile.mkdtemp())
        tmp_path = tmp_dir / artifact_name
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            mlflow_mgr.log_artifact(
                str(tmp_path),
                artifact_path=artifact_path,
                run_id=run_id,
            )
            logger.debug(
                "[ARTIFACT] Wrote %s/%s (status=%s)",
                artifact_path or ".",
                artifact_name,
                envelope.status,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                tmp_dir.rmdir()

    except Exception as exc:
        logger.warning(
            "[ARTIFACT] Failed to write artifact %s (non-fatal): %s",
            artifact_name,
            exc,
        )


def utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.utcnow().isoformat()
