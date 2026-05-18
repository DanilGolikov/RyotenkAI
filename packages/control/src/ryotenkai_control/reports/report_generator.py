"""
Experiment Report Generator (v2).

Main facade for report generation using Domain-Centric architecture.
Architecture:
    MLflow             -> MLflowAdapter        -> ExperimentData (metadata)
    events.jsonl       -> JournalReportAdapter -> ExperimentEventData
    [composed]         -> ReportBuilder        -> Report

Phase 7 split the adapter layer: MLflow metadata (params/tags/metric
history/stage envelopes) is still loaded by :class:`MLflowAdapter`,
but the runtime event stream (memory + timeline) is now provided by
:class:`JournalReportAdapter`, reading the typed event journal written
during the run.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ryotenkai_community.catalog import catalog
from ryotenkai_control.pipeline.mlflow.read.client import MlflowReadClient
from ryotenkai_control.reports.adapters.journal_adapter import JournalReportAdapter
from ryotenkai_control.reports.adapters.mlflow_adapter import MLflowAdapter
from ryotenkai_control.reports.core.builder import ReportBuilder
from ryotenkai_control.reports.plugins.composer import ReportComposer
from ryotenkai_control.reports.plugins.interfaces import IReportBlockPlugin, ReportPluginContext
from ryotenkai_control.reports.plugins.markdown_block_renderer import MarkdownBlockRenderer
from ryotenkai_control.reports.plugins.registry import build_report_plugins
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from ryotenkai_control.reports.domain.entities import ExperimentData
    from ryotenkai_control.reports.domain.interfaces import IExperimentDataProvider
    from ryotenkai_control.reports.models.report import ExperimentReport
    from ryotenkai_shared.infrastructure.mlflow.gateway import IMLflowGateway
    from ryotenkai_shared.infrastructure.mlflow.protocol import IMLflowManager

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
        run_query: MlflowReadClient | None = None,
        adapter: IExperimentDataProvider | None = None,
        journal_adapter: JournalReportAdapter | None = None,
        mlflow_manager: IMLflowManager | None = None,
        plugins: list[IReportBlockPlugin] | None = None,
        sections: Sequence[str] | None = None,
    ):
        """
        Initialize generator.

        Args:
            tracking_uri:     MLflow tracking URI (legacy, used when gateway is not provided).
            gateway:          IMLflowGateway instance. Used to derive the URI for the read
                              client.
            run_query:        Pre-built :class:`MlflowReadClient`. Phase M3.B preferred
                              entry point — when provided, no global tracking-URI
                              mutation is performed.
            adapter:          MLflow metadata provider (defaults to MLflowAdapter).
            journal_adapter:  Event-journal provider (defaults to JournalReportAdapter
                              bound to ``mlflow_manager`` for the artifact fallback path).
            mlflow_manager:   IMLflowManager used by the journal adapter for the
                              MLflow-artifact fallback when the workspace journal is
                              missing. Optional; the adapter degrades to "empty
                              events" when both are unavailable.
            plugins:          Ordered list of report block plugins (escape hatch for tests).
                              If passed, ``sections`` is ignored.
            sections:         Ordered list of plugin ids to render. ``None`` uses the
                              built-in default (see ``DEFAULT_REPORT_SECTIONS``).
        """
        # Phase M3.B: construct exactly one MlflowReadClient and route the
        # underlying client through it. No ``mlflow.set_tracking_uri``
        # mutation — the read client owns the URI.
        if run_query is not None:
            self._run_query = run_query
            self._tracking_uri = run_query.tracking_uri
        elif gateway is not None:
            self._tracking_uri = gateway.uri
            self._run_query = MlflowReadClient(tracking_uri=gateway.uri)
        elif tracking_uri is not None:
            self._tracking_uri = tracking_uri
            self._run_query = MlflowReadClient(tracking_uri=tracking_uri)
        else:
            raise ValueError(
                "One of run_query, gateway, or tracking_uri must be provided"
            )
        self._client = self._run_query.underlying_client
        self._adapter = adapter or MLflowAdapter(run_query=self._run_query)

        self._journal_adapter = journal_adapter or JournalReportAdapter(mlflow_manager=mlflow_manager)

        if plugins is None:
            catalog.ensure_loaded()
            self._plugins = build_report_plugins(sections)
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

        # 1b. Merge journal-derived events into the experiment data so
        # the builder can render the timeline + memory-management section.
        workspace_dir = local_logs_dir if local_logs_dir else None
        event_data = self._journal_adapter.load(run_id, workspace_dir=workspace_dir)
        data.memory_events = event_data.memory_events

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
        event_data = self._journal_adapter.load(run_id)
        data.memory_events = event_data.memory_events
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
