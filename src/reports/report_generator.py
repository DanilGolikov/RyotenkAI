"""
Experiment Report Generator (v2).

Main facade for report generation using Domain-Centric architecture.
Architecture: MLflow -> Adapter -> Domain Data -> Builder -> Report.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
from mlflow.tracking import MlflowClient

from src.reports.adapters.mlflow_adapter import MLflowAdapter
from src.reports.core.builder import ReportBuilder
from src.community.catalog import catalog
from src.reports.plugins.composer import ReportComposer
from src.reports.plugins.interfaces import IReportBlockPlugin, ReportPluginContext
from src.reports.plugins.markdown_block_renderer import MarkdownBlockRenderer
from src.reports.plugins.registry import build_report_plugins
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.infrastructure.mlflow.gateway import IMLflowGateway
    from src.reports.domain.entities import ExperimentData
    from src.reports.domain.interfaces import IExperimentDataProvider
    from src.reports.models.report import ExperimentReport

logger = get_logger(__name__)


class ExperimentReportGenerator:
    """
    Main class for experiment report generation.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        *,
        gateway: IMLflowGateway | None = None,
        adapter: IExperimentDataProvider | None = None,
        plugins: list[IReportBlockPlugin] | None = None,
    ):
        """
        Initialize generator.

        Args:
            tracking_uri: MLflow tracking URI (legacy, used when gateway is not provided)
            gateway:      IMLflowGateway instance. Takes precedence over tracking_uri.
            adapter:      Data provider (defaults to MLflowAdapter)
            plugins:      Ordered list of report block plugins (defaults to builtins)
        """
        if gateway is not None:
            self._tracking_uri = gateway.uri
            self._client = gateway.get_client()
            mlflow.set_tracking_uri(gateway.uri)
            self._adapter = adapter or MLflowAdapter(gateway=gateway)
        elif tracking_uri is not None:
            self._tracking_uri = tracking_uri
            self._client = MlflowClient(tracking_uri=tracking_uri)
            mlflow.set_tracking_uri(tracking_uri)
            self._adapter = adapter or MLflowAdapter(tracking_uri)
        else:
            raise ValueError("Either tracking_uri or gateway must be provided")

        if plugins is None:
            catalog.ensure_loaded()
            self._plugins = build_report_plugins()
        else:
            self._plugins = plugins
        self._composer = ReportComposer(self._plugins)
        self._block_renderer = MarkdownBlockRenderer()

        logger.debug(f"[GENERATOR] Initialized with tracking_uri={self._tracking_uri}")

    def generate(
        self,
        run_id: str,
        *,
        local_logs_dir: Path | None = None,
        artifact_name: str = "experiment_report.md",
    ) -> str:
        """
        Generate experiment report.
        """
        logger.info(f"[GENERATOR] Generating report for run {run_id[:8]}...")

        # 1. Fetch Data (Adapter Layer)
        data: ExperimentData = self._adapter.load(run_id)

        # 2. Build Report (Domain -> View)
        builder = ReportBuilder(data)
        report: ExperimentReport = builder.build()

        # 3. Render (Plugins -> Blocks -> Markdown)
        ctx = ReportPluginContext(
            run_id=run_id,
            data_provider=self._adapter,
            data=data,
            report=report,
            logger=logger,
        )
        blocks, records = self._composer.compose(ctx)
        markdown = self._block_renderer.render(blocks)

        failed_blocks = sum(1 for r in records if r.status == "failed")
        logger.info(
            f"[GENERATOR] Report rendered ({len(markdown)} chars, blocks={len(blocks)}, failed_blocks={failed_blocks})"
        )

        # 4. Upload & Save
        self._upload_to_mlflow(run_id, markdown, artifact_name)

        if local_logs_dir:
            self._copy_to_local(local_logs_dir, markdown, artifact_name)

        logger.info(f"[GENERATOR] Generation complete for {run_id[:8]}")
        return markdown

    def generate_report_model(self, run_id: str) -> ExperimentReport:
        """Generate report model without rendering."""
        data = self._adapter.load(run_id)
        return ReportBuilder(data).build()

    def generate_for_latest(
        self,
        experiment_name: str,
        *,
        local_logs_dir: Path | None = None,
        artifact_name: str = "experiment_report.md",
    ) -> str | None:
        """Generate report for latest run."""
        try:
            experiment = self._client.get_experiment_by_name(experiment_name)
            if not experiment:
                logger.warning(f"Experiment not found: {experiment_name}")
                return None

            runs = self._client.search_runs(
                experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1
            )

            if not runs:
                return None

            return self.generate(runs[0].info.run_id, local_logs_dir=local_logs_dir, artifact_name=artifact_name)
        except Exception as e:
            logger.error(f"Failed to generate for latest: {e}")
            return None

    def download_from_mlflow(
        self,
        *,
        run_id: str,
        local_dir: Path,
        artifact_name: str = "experiment_report.md",
    ) -> Path | None:
        """
        Download an existing report artifact from MLflow to a local directory.

        Returns:
            Path to downloaded file or None if artifact not found / download failed.
        """
        try:
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            downloaded_path = self._client.download_artifacts(run_id, artifact_name, str(local_dir))
            return Path(downloaded_path)
        except Exception as e:
            logger.error(f"Failed to download artifact '{artifact_name}' for run {run_id[:8]}: {e}")
            return None

    def _upload_to_mlflow(self, run_id: str, content: str, artifact_name: str) -> bool:
        try:
            self._client.log_text(run_id, content, artifact_name)
            return True
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    @staticmethod
    def _copy_to_local(local_dir: Path, content: str, filename: str) -> bool:
        try:
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / filename).write_text(content, encoding="utf-8")
            return True
        except Exception as e:
            logger.error(f"Local save failed: {e}")
            return False


__all__ = ["ExperimentReportGenerator"]
