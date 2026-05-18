from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ryotenkai_control.api.schemas.report import ReportResponse
from ryotenkai_control.pipeline.state import PipelineStateStore


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
    from ryotenkai_control.reports.report_generator import ExperimentReportGenerator

    # Phase M3.B: pass the persisted runtime tracking URI through the
    # new DI'd entry point so the generator owns exactly one
    # MlflowReadClient for the lifetime of the call.
    tracking_uri = state.mlflow_runtime_tracking_uri or "http://localhost:5002"
    generator = ExperimentReportGenerator(tracking_uri=tracking_uri)
    markdown = generator.generate(state.root_mlflow_run_id, local_logs_dir=run_dir)
    report_path.write_text(markdown, encoding="utf-8")
    return ReportResponse(
        path=str(report_path),
        markdown=markdown,
        generated_at=datetime.now(UTC).isoformat(),
        regenerated=True,
    )


__all__ = ["get_or_generate_report"]
