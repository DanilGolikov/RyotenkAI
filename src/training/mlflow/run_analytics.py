"""
MLflowRunAnalytics — run search, comparison and summary generation.

Responsibilities (Single Responsibility):
  - Query runs by metric, filter, experiment
  - Compare multiple runs
  - Get metric history
  - Get experiment-level statistics
  - Generate Markdown summary report combining MLflow data and in-memory events

Depends on: IMLflowGateway (for direct client access), mlflow module
(for search_runs), MLflowEventLog (for events section in summary).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.training.constants import (
    MLFLOW_DF_EMPTY_ATTR,
    MLFLOW_DF_TO_DICT,
    MLFLOW_KEY_TAGS,
    MLFLOW_MD_SEPARATOR,
)
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.infrastructure.mlflow.gateway import IMLflowGateway
    from src.training.mlflow.event_log import MLflowEventLog

logger = get_logger(__name__)


class MLflowRunAnalytics:
    """
    MLflow run search, comparison and summary report generation.

    Args:
        gateway: MLflow gateway for client-based API access
        mlflow_module: The imported mlflow module (for search_runs)
        experiment_name: Default experiment name for searches
        event_log: Event log instance for summary report events section
    """

    def __init__(
        self,
        gateway: IMLflowGateway,
        mlflow_module: Any,
        experiment_name: str | None = None,
        event_log: MLflowEventLog | None = None,
    ) -> None:
        self._gateway = gateway
        self._mlflow = mlflow_module
        self._experiment_name = experiment_name
        self._event_log = event_log

    def _resolve_experiment_name(self, experiment_name: str | None) -> str | None:
        return experiment_name or self._experiment_name

    # =========================================================================
    # RUN SEARCH
    # =========================================================================

    def get_best_run(
        self,
        metric: str = "eval_loss",
        mode: str = "min",
        experiment_name: str | None = None,
        filter_string: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Get the best run from experiment based on a metric.

        Args:
            metric: Metric name to optimize (default: "eval_loss")
            mode: "min" or "max"
            experiment_name: Experiment name (default: current)
            filter_string: Optional MLflow filter string

        Returns:
            Dict with best run info or None
        """
        if self._mlflow is None:
            return None

        exp_name = self._resolve_experiment_name(experiment_name)
        if not exp_name:
            logger.warning("[MLFLOW:ANALYTICS] No experiment name specified")
            return None

        try:
            order_by = f"metrics.{metric} {'ASC' if mode == 'min' else 'DESC'}"
            search_kwargs: dict[str, Any] = {
                "experiment_names": [exp_name],
                "order_by": [order_by],
                "max_results": 1,
            }
            if filter_string:
                search_kwargs["filter_string"] = filter_string

            runs = self._mlflow.search_runs(**search_kwargs)

            if hasattr(runs, MLFLOW_DF_EMPTY_ATTR) and getattr(runs, MLFLOW_DF_EMPTY_ATTR):
                return None

            if hasattr(runs, MLFLOW_DF_TO_DICT):
                run_list = getattr(runs, MLFLOW_DF_TO_DICT)(orient="records")
                if run_list:
                    best_run = run_list[0]
                    logger.info(
                        f"[MLFLOW:ANALYTICS] Best run: {best_run.get('run_id', 'N/A')[:8]}... "
                        f"({metric}={best_run.get(f'metrics.{metric}', 'N/A')})"
                    )
                    return best_run

            return None
        except Exception as e:
            logger.warning(f"[MLFLOW:ANALYTICS] Failed to get best run: {e}")
            return None

    def compare_runs(
        self,
        run_ids: list[str],
        metrics: list[str] | None = None,
        params: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Compare multiple runs by specified metrics and params.

        Args:
            run_ids: List of run IDs to compare
            metrics: List of metric names to include (default: all)
            params: List of param names to include (default: all)

        Returns:
            List of dicts with run info, metrics, and params
        """
        if self._mlflow is None:
            return []

        try:
            client = self._gateway.get_client()
            results = []

            for run_id in run_ids:
                try:
                    run = client.get_run(run_id)

                    run_metrics = dict(run.data.metrics)
                    if metrics:
                        run_metrics = {k: v for k, v in run_metrics.items() if k in metrics}

                    run_params = dict(run.data.params)
                    if params:
                        run_params = {k: v for k, v in run_params.items() if k in params}

                    results.append(
                        {
                            "run_id": run_id,
                            "run_name": run.info.run_name,
                            "status": run.info.status,
                            "start_time": run.info.start_time,
                            "end_time": run.info.end_time,
                            "metrics": run_metrics,
                            "params": run_params,
                            "tags": dict(run.data.tags),
                        }
                    )
                except Exception as e:
                    logger.warning(f"[MLFLOW:ANALYTICS] Failed to get run {run_id}: {e}")

            return results
        except Exception as e:
            logger.warning(f"[MLFLOW:ANALYTICS] Failed to compare runs: {e}")
            return []

    def search_runs(
        self,
        filter_string: str | None = None,
        experiment_name: str | None = None,
        order_by: list[str] | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Search runs with filters and ordering.

        Args:
            filter_string: MLflow filter string (e.g., "metrics.loss < 0.5")
            experiment_name: Experiment name (default: current)
            order_by: List of order columns
            max_results: Maximum number of results

        Returns:
            List of run dicts
        """
        if self._mlflow is None:
            return []

        exp_name = self._resolve_experiment_name(experiment_name)
        if not exp_name:
            return []

        try:
            search_kwargs: dict[str, Any] = {
                "experiment_names": [exp_name],
                "max_results": max_results,
            }
            if filter_string:
                search_kwargs["filter_string"] = filter_string
            if order_by:
                search_kwargs["order_by"] = order_by

            runs = self._mlflow.search_runs(**search_kwargs)

            if hasattr(runs, MLFLOW_DF_EMPTY_ATTR) and getattr(runs, MLFLOW_DF_EMPTY_ATTR):
                return []

            if hasattr(runs, MLFLOW_DF_TO_DICT):
                return getattr(runs, MLFLOW_DF_TO_DICT)(orient="records")

            return []
        except Exception as e:
            logger.warning(f"[MLFLOW:ANALYTICS] Failed to search runs: {e}")
            return []

    def get_run_metrics_history(
        self,
        run_id: str,
        metric: str,
    ) -> list[dict[str, Any]]:
        """
        Get metric history for a run (all logged values).

        Args:
            run_id: Run ID
            metric: Metric name

        Returns:
            List of dicts with step, timestamp, value
        """
        if self._mlflow is None:
            return []

        try:
            client = self._gateway.get_client()
            history = client.get_metric_history(run_id, metric)
            return [{"step": m.step, "timestamp": m.timestamp, "value": m.value} for m in history]
        except Exception as e:
            logger.warning(f"[MLFLOW:ANALYTICS] Failed to get metric history: {e}")
            return []

    def get_experiment_summary(
        self,
        experiment_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Get summary statistics for an experiment.

        Args:
            experiment_name: Experiment name (default: current)

        Returns:
            Dict with summary stats
        """
        if self._mlflow is None:
            return {}

        exp_name = self._resolve_experiment_name(experiment_name)
        if not exp_name:
            return {}

        try:
            runs = self._mlflow.search_runs(
                experiment_names=[exp_name],
                max_results=1000,
            )

            if hasattr(runs, MLFLOW_DF_EMPTY_ATTR) and getattr(runs, MLFLOW_DF_EMPTY_ATTR):
                return {"total_runs": 0}

            total = len(runs) if hasattr(runs, "__len__") else 0
            summary: dict[str, Any] = {
                "experiment_name": exp_name,
                "total_runs": total,
            }

            if hasattr(runs, "columns"):
                metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
                for col in metric_cols:
                    metric_name = col.replace("metrics.", "")
                    values = runs[col].dropna()
                    if len(values) > 0:
                        summary[f"best_{metric_name}"] = float(values.min())
                        summary[f"worst_{metric_name}"] = float(values.max())
                        summary[f"avg_{metric_name}"] = float(values.mean())

            return summary
        except Exception as e:
            logger.warning(f"[MLFLOW:ANALYTICS] Failed to get experiment summary: {e}")
            return {}

    def get_child_runs(self, parent_run_id: str) -> list[dict[str, Any]]:
        """
        Get all child runs for a parent run.

        Args:
            parent_run_id: Parent run ID

        Returns:
            List of child run info dicts
        """
        if self._mlflow is None or not parent_run_id:
            return []

        try:
            runs = self._mlflow.search_runs(
                filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
                order_by=["start_time ASC"],
            )

            if hasattr(runs, MLFLOW_DF_EMPTY_ATTR) and getattr(runs, MLFLOW_DF_EMPTY_ATTR):
                return []

            if hasattr(runs, MLFLOW_DF_TO_DICT):
                return getattr(runs, MLFLOW_DF_TO_DICT)(orient="records")

            return []
        except Exception as e:
            logger.warning(f"[MLFLOW:ANALYTICS] Failed to get child runs: {e}")
            return []

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def get_run_data(self, run_id: str) -> dict[str, Any] | None:
        """
        Get run data (params, metrics, tags) by run_id.

        Args:
            run_id: MLflow run ID

        Returns:
            Dict with params, metrics, tags, info or None
        """
        if self._mlflow is None or not run_id:
            return None

        try:
            client = self._gateway.get_client()
            run = client.get_run(run_id)
            return {
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
                "tags": dict(run.data.tags),
                "info": {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                },
            }
        except Exception as e:
            logger.warning(f"[MLFLOW:ANALYTICS] Failed to get run data: {e}")
            return None

    def generate_summary_markdown(self, run_id: str | None = None) -> str:
        """
        Generate Markdown summary report combining MLflow run data and event log.

        Args:
            run_id: Run ID to pull params/metrics from

        Returns:
            Markdown formatted report string
        """
        from datetime import datetime

        lines = [
            "# Training Summary Report",
            "",
            f"**Run ID**: {run_id or 'N/A'}",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            MLFLOW_MD_SEPARATOR,
            "",
        ]

        run_data = self.get_run_data(run_id) if run_id else None

        # Overview
        lines.append("## Overview")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")

        if run_data:
            params = run_data.get("params", {})
            lines.append(f"| Model | {params.get('model_name', 'N/A')} |")
            lines.append(f"| Training Type | {params.get('training_type', 'N/A')} |")
            lines.append(f"| GPU | {params.get('gpu_name', 'N/A')} |")
            lines.append(f"| GPU Tier | {params.get('gpu_tier', 'N/A')} |")
            tags = run_data.get(MLFLOW_KEY_TAGS, {})
            lines.append(f"| Strategy Chain | {tags.get('strategy_chain', 'N/A')} |")

        lines.append("")
        lines.append(MLFLOW_MD_SEPARATOR)
        lines.append("")

        # Events section — delegated to event_log if available
        if self._event_log is not None:
            lines.extend(self._event_log.generate_summary_section())
        else:
            lines.append("## Events Timeline")
            lines.append("")
            lines.append("*(Event log not available)*")
            lines.append("")

        # Results
        lines.append("## Results")
        lines.append("")

        if run_data:
            metrics = run_data.get("metrics", {})
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            if "train_loss" in metrics:
                lines.append(f"| Train Loss | {metrics['train_loss']:.4f} |")
            if "eval_loss" in metrics:
                lines.append(f"| Eval Loss | {metrics['eval_loss']:.4f} |")
            if "global_step" in metrics:
                lines.append(f"| Total Steps | {int(metrics['global_step']):,} |")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Generated by RyotenkAI Training Pipeline*")

        return "\n".join(lines)


__all__ = ["MLflowRunAnalytics"]
