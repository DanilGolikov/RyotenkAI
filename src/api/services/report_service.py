from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from src.api.schemas.report import ReportResponse
from src.pipeline.state import PipelineStateStore


def get_or_generate_report(run_dir: Path, *, regenerate: bool = False) -> ReportResponse:
    report_path = run_dir / "report.md"
    if report_path.exists() and not regenerate:
        return ReportResponse(
            path=str(report_path),
            markdown=report_path.read_text(encoding="utf-8"),
            generated_at=datetime.fromtimestamp(report_path.stat().st_mtime, tz=UTC).isoformat(),
            regenerated=False,
        )
    state = PipelineStateStore(run_dir).load()
    if not state.root_mlflow_run_id:
        raise ValueError("root_mlflow_run_id not found in pipeline_state.json")

    # Import lazily to avoid pulling MLflow on every call.
    from src.reports.report_generator import ExperimentReportGenerator

    generator = ExperimentReportGenerator()
    markdown = generator.generate(state.root_mlflow_run_id, local_logs_dir=run_dir)
    report_path.write_text(markdown, encoding="utf-8")
    return ReportResponse(
        path=str(report_path),
        markdown=markdown,
        generated_at=datetime.now(UTC).isoformat(),
        regenerated=True,
    )


__all__ = ["get_or_generate_report"]
