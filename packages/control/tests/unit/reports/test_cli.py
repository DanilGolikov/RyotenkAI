from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import CliRunner

import src.reports.cli as reports_cli


@dataclass
class FakeGenerator:
    tracking_uri: str
    mode: str = "ok"
    report_text: str = "REPORT"
    downloaded_path: Path | None = None

    # capture
    last_generate: dict | None = None
    last_generate_latest: dict | None = None
    last_download: dict | None = None

    def generate(self, *, run_id: str, local_logs_dir: Path | None = None) -> str | None:
        self.last_generate = {"run_id": run_id, "local_logs_dir": local_logs_dir}
        if self.mode == "none":
            return None
        if self.mode == "value_error":
            raise ValueError("bad input")
        if self.mode == "crash":
            raise RuntimeError("boom")
        return self.report_text

    def generate_for_latest(self, *, experiment_name: str, local_logs_dir: Path | None = None) -> str | None:
        self.last_generate_latest = {"experiment_name": experiment_name, "local_logs_dir": local_logs_dir}
        return self.report_text

    def download_from_mlflow(self, *, run_id: str, local_dir: Path) -> Path | None:
        self.last_download = {"run_id": run_id, "local_dir": local_dir}
        return self.downloaded_path


class FakeGeneratorFactory:
    """Helper to swap ExperimentReportGenerator and access last instance."""

    last: FakeGenerator | None = None

    def __init__(self, tracking_uri: str):
        # Default ok mode; tests can mutate FakeGeneratorFactory.last after invoke if needed.
        FakeGeneratorFactory.last = FakeGenerator(tracking_uri=tracking_uri)

    def __getattr__(self, item: str):
        assert FakeGeneratorFactory.last is not None
        return getattr(FakeGeneratorFactory.last, item)


def test_generate_requires_run_id_or_experiment_latest() -> None:
    runner = CliRunner()
    result = runner.invoke(reports_cli.cli, ["generate"])
    assert result.exit_code == 2
    assert "Either --run-id or (--experiment + --latest) is required" in result.output


def test_generate_run_id_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reports_cli, "ExperimentReportGenerator", FakeGeneratorFactory)

    runner = CliRunner()
    result = runner.invoke(
        reports_cli.cli,
        [
            "generate",
            "--run-id",
            "abc123",
            "--tracking-uri",
            "http://mlflow:5000",
        ],
    )

    assert result.exit_code == 0
    assert "✅ Report generated successfully" in result.output
    assert "REPORT" in result.output

    gen = FakeGeneratorFactory.last
    assert gen is not None
    assert gen.last_generate == {"run_id": "abc123", "local_logs_dir": None}


def test_generate_latest_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reports_cli, "ExperimentReportGenerator", FakeGeneratorFactory)

    runner = CliRunner()
    result = runner.invoke(
        reports_cli.cli,
        [
            "generate",
            "--experiment",
            "helix-training",
            "--latest",
        ],
    )

    assert result.exit_code == 0
    gen = FakeGeneratorFactory.last
    assert gen is not None
    assert gen.last_generate_latest == {"experiment_name": "helix-training", "local_logs_dir": None}


def test_generate_writes_output_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(reports_cli, "ExperimentReportGenerator", FakeGeneratorFactory)

    out = tmp_path / "report.md"
    runner = CliRunner()
    result = runner.invoke(
        reports_cli.cli,
        [
            "generate",
            "--run-id",
            "abc123",
            "--output",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert out.exists()
    assert out.read_text(encoding="utf-8") == "REPORT"
    assert "✅ Report saved to:" in result.output
    assert "REPORT" not in result.output  # report is not printed when --output is used


def test_generate_no_runs_found_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    class FactoryNone(FakeGeneratorFactory):
        def __init__(self, tracking_uri: str):
            FakeGeneratorFactory.last = FakeGenerator(tracking_uri=tracking_uri, mode="none")

    monkeypatch.setattr(reports_cli, "ExperimentReportGenerator", FactoryNone)

    runner = CliRunner()
    result = runner.invoke(reports_cli.cli, ["generate", "--run-id", "abc123"])

    assert result.exit_code == 1
    assert "❌ No runs found" in result.output


def test_generate_value_error_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    class FactoryValueError(FakeGeneratorFactory):
        def __init__(self, tracking_uri: str):
            FakeGeneratorFactory.last = FakeGenerator(tracking_uri=tracking_uri, mode="value_error")

    monkeypatch.setattr(reports_cli, "ExperimentReportGenerator", FactoryValueError)

    runner = CliRunner()
    result = runner.invoke(reports_cli.cli, ["generate", "--run-id", "abc123"])

    assert result.exit_code == 1
    assert "❌ Error: bad input" in result.output


def test_generate_unexpected_error_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    class FactoryCrash(FakeGeneratorFactory):
        def __init__(self, tracking_uri: str):
            FakeGeneratorFactory.last = FakeGenerator(tracking_uri=tracking_uri, mode="crash")

    monkeypatch.setattr(reports_cli, "ExperimentReportGenerator", FactoryCrash)

    runner = CliRunner()
    result = runner.invoke(reports_cli.cli, ["generate", "--run-id", "abc123"])

    assert result.exit_code == 1
    assert "❌ Unexpected error: boom" in result.output


def test_download_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FactoryDownload(FakeGeneratorFactory):
        def __init__(self, tracking_uri: str):
            FakeGeneratorFactory.last = FakeGenerator(
                tracking_uri=tracking_uri,
                downloaded_path=tmp_path / "experiment_report.md",
            )

    monkeypatch.setattr(reports_cli, "ExperimentReportGenerator", FactoryDownload)

    runner = CliRunner()
    result = runner.invoke(
        reports_cli.cli,
        ["download", "--run-id", "abc123", "--local-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "✅ Report downloaded to:" in result.output


def test_download_not_found_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FactoryDownloadNone(FakeGeneratorFactory):
        def __init__(self, tracking_uri: str):
            FakeGeneratorFactory.last = FakeGenerator(tracking_uri=tracking_uri, downloaded_path=None)

    monkeypatch.setattr(reports_cli, "ExperimentReportGenerator", FactoryDownloadNone)

    runner = CliRunner()
    result = runner.invoke(
        reports_cli.cli,
        ["download", "--run-id", "abc123", "--local-dir", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "❌ Report not found in MLflow artifacts" in result.output
