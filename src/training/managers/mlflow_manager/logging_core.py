"""
MLflowLoggingMixin — Core MLflow logging primitives.

Responsibilities:
  - log_params()          — training parameters
  - log_metrics()         — numeric metrics
  - log_artifact()        — file artifacts
  - log_dict()            — JSON artifact
  - log_text()            — text artifact
  - set_tags() / set_tag() — run tags
  - log_summary_artifact() — consolidated summary artifact
  - log_llm_evaluation()  — LLM eval result logging
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from src.training.constants import (
    MLFLOW_TRUNCATE_FEEDBACK,
    MLFLOW_TRUNCATE_PROMPT,
    MLFLOW_TRUNCATE_RESPONSE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLflowLoggingMixin:
    """
    Mixin: core MLflow logging primitives.

    Assumes the following attributes exist on self (set by MLflowManager.__init__):
      _mlflow, _run, _run_id, _mlflow_config, client (property), _get_active_run_id()
    """

    # ------------------------------------------------------------------
    # Parameters & metrics
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to current run."""
        if self._mlflow is None or self._run is None:  # type: ignore[attr-defined]
            return
        try:
            clean_params = {k: str(v) if v is not None else "None" for k, v in params.items()}
            self._mlflow.log_params(clean_params)  # type: ignore[attr-defined]
            logger.debug(f"[MLFLOW:PARAMS] {len(clean_params)} params logged")
        except Exception as e:
            logger.warning(f"[MLFLOW] log_params failed: {e}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to current run."""
        if self._mlflow is None or self._run is None:  # type: ignore[attr-defined]
            return
        try:
            for key, value in metrics.items():
                if value is not None:
                    self._mlflow.log_metric(key, float(value), step=step)  # type: ignore[attr-defined]
            logger.debug(f"[MLFLOW:METRICS] {metrics}")
        except Exception as e:
            logger.warning(f"[MLFLOW] log_metrics failed: {e}")

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
        run_id: str | None = None,
    ) -> bool:
        """Log artifact file to MLflow run via HTTP API."""
        target_run_id = self._get_active_run_id(run_id)  # type: ignore[attr-defined]
        if not target_run_id or self.client is None:  # type: ignore[attr-defined]
            return False

        path = Path(local_path)
        if not path.exists():
            logger.warning(f"[MLFLOW] Artifact not found: {local_path}")
            return False

        try:
            content = path.read_text(encoding="utf-8")
            artifact_name = path.name
            if artifact_path:
                artifact_name = f"{artifact_path}/{artifact_name}"
            self.client.log_text(target_run_id, content, artifact_name)  # type: ignore[attr-defined]
            logger.debug(f"[MLFLOW:ARTIFACT] {local_path} -> {artifact_name}")
            return True
        except UnicodeDecodeError:
            logger.debug(f"[MLFLOW:ARTIFACT] Skipping binary file: {local_path}")
            return False
        except Exception as e:
            logger.warning(f"[MLFLOW] log_artifact failed: {e}")
            return False

    def log_dict(
        self,
        dictionary: dict[str, Any],
        artifact_file: str,
        run_id: str | None = None,
    ) -> bool:
        """Log dict as JSON artifact to specific run."""
        target_run_id = self._get_active_run_id(run_id)  # type: ignore[attr-defined]
        if not target_run_id:
            logger.debug(f"[MLFLOW:DICT] Skipped {artifact_file} - no active run")
            return False

        try:
            self.client.log_dict(target_run_id, dictionary, artifact_file)  # type: ignore[attr-defined]
            logger.debug(f"[MLFLOW:DICT] {artifact_file} -> run_id={target_run_id[:8]}...")
            return True
        except Exception as e:
            logger.error(f"[MLFLOW:DICT] Failed to log {artifact_file}: {e}")
            logger.error(f"[MLFLOW:DICT] Traceback:\n{traceback.format_exc()}")
            return False

    def log_text(
        self,
        text: str,
        artifact_file: str,
        run_id: str | None = None,
    ) -> bool:
        """Log text content as artifact to specific run."""
        target_run_id = self._get_active_run_id(run_id)  # type: ignore[attr-defined]
        if not target_run_id:
            return False
        try:
            self.client.log_text(target_run_id, text, artifact_file)  # type: ignore[attr-defined]
            logger.debug(f"[MLFLOW:TEXT] {artifact_file} -> run_id={target_run_id[:8]}...")
            return True
        except Exception as e:
            logger.error(f"[MLFLOW:TEXT] Failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags on current run."""
        if self._mlflow is None or self._run is None:  # type: ignore[attr-defined]
            return
        try:
            self._mlflow.set_tags(tags)  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(f"[MLFLOW] set_tags failed: {e}")

    def set_tag(self, key: str, value: str) -> None:
        """Set a single tag on current run."""
        self.set_tags({key: value})  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Summary artifact
    # ------------------------------------------------------------------

    def log_summary_artifact(
        self,
        events_artifact_name: str = "training_events.json",
        parent_run_id: str | None = None,
    ) -> bool:
        """Generate and log summary as MLflow artifact."""
        target_run_id = self._get_active_run_id(parent_run_id)  # type: ignore[attr-defined]
        try:
            events_ok = self.log_events_artifact(events_artifact_name, run_id=target_run_id)  # type: ignore[attr-defined]
            if events_ok:
                logger.info("[MLFLOW:SUMMARY] Events artifact logged")
                return True
            logger.warning("[MLFLOW:SUMMARY] No artifacts were logged")
            return False
        except Exception as e:
            logger.warning(f"[MLFLOW:SUMMARY] Failed to log summary: {e}")
            return False

    # ------------------------------------------------------------------
    # LLM evaluation
    # ------------------------------------------------------------------

    def log_llm_evaluation(
        self,
        prompt: str,
        response: str,
        expected: str | None = None,
        score: float | None = None,
        feedback: str | None = None,
        evaluator: str = "human",
    ) -> None:
        """Log LLM evaluation result."""
        if self._mlflow is None:  # type: ignore[attr-defined]
            return

        try:
            evaluation_data: dict[str, Any] = {
                "prompt": prompt[:MLFLOW_TRUNCATE_PROMPT],
                "response": response[:MLFLOW_TRUNCATE_RESPONSE],
                "evaluator": evaluator,
            }
            if expected:
                evaluation_data["expected"] = expected[:MLFLOW_TRUNCATE_RESPONSE]
            if feedback:
                evaluation_data["feedback"] = feedback[:MLFLOW_TRUNCATE_FEEDBACK]

            run_id_prefix = self._run_id[:8] if self._run_id else "unknown"  # type: ignore[attr-defined]
            artifact_name = f"evaluation_{evaluator}_{run_id_prefix}.json"
            self.log_dict(evaluation_data, artifact_name)  # type: ignore[attr-defined]

            if score is not None:
                self._mlflow.log_metric(f"eval_score_{evaluator}", score)  # type: ignore[attr-defined]

            logger.debug(f"[MLFLOW:EVAL] evaluator={evaluator}, score={score}")

        except Exception as e:
            logger.debug(f"[MLFLOW:EVAL] Failed to log evaluation: {e}")


__all__ = ["MLflowLoggingMixin"]
