"""
CLI for Experiment Report Generation.

Usage:
    # Generate report for specific run
    python -m src.reports generate --run-id abc123

    # Generate and save locally
    python -m src.reports generate --run-id abc123 --local-dir runs/20251208_xxx/

    # Generate for latest run in experiment
    python -m src.reports generate --experiment "ryotenkai-training" --latest

    # Download existing report from MLflow
    python -m src.reports download --run-id abc123 --local-dir runs/xxx/
"""

from pathlib import Path
from typing import cast

import click

from src.reports.report_generator import ExperimentReportGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


@click.group()
def cli():
    """Experiment report generation commands."""
    pass


@cli.command()
@click.option(
    "--run-id",
    help="MLflow run ID",
)
@click.option(
    "--experiment",
    help="Experiment name (use with --latest)",
)
@click.option(
    "--latest",
    is_flag=True,
    help="Use latest run from experiment",
)
@click.option(
    "--local-dir",
    type=click.Path(path_type=Path),
    help="Local directory to save report (e.g., runs/20251208_xxx/)",
)
@click.option(
    "--tracking-uri",
    default="http://localhost:5002",
    help="MLflow tracking URI",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path (alternative to --local-dir)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Don't print report to stdout",
)
def generate(
    run_id: str | None,
    experiment: str | None,
    latest: bool,
    local_dir: Path | None,
    tracking_uri: str,
    output: Path | None,
    quiet: bool,
):
    """
    Generate experiment report from MLflow run.

    Report is saved to MLflow artifacts and optionally to local directory.

    Examples:

        # Generate for specific run
        python -m src.reports generate --run-id abc123def456

        # Generate for latest run in experiment
        python -m src.reports generate --experiment ryotenkai-training --latest

        # Save to local logs directory
        python -m src.reports generate --run-id abc123 --local-dir runs/20251208_xxx/
    """
    # Validate arguments
    if not run_id and not (experiment and latest):
        raise click.UsageError("Either --run-id or (--experiment + --latest) is required")

    generator = ExperimentReportGenerator(tracking_uri)

    try:
        if run_id:
            report_opt = cast(
                "str | None",
                generator.generate(
                    run_id=run_id,
                    local_logs_dir=local_dir,
                ),
            )
            if report_opt is None:
                click.echo("❌ No runs found", err=True)
                raise SystemExit(1)
            report = report_opt
        else:
            # experiment + latest
            latest_report = generator.generate_for_latest(
                experiment_name=experiment,  # type: ignore
                local_logs_dir=local_dir,
            )
            if latest_report is None:
                click.echo("❌ No runs found", err=True)
                raise SystemExit(1)
            report = latest_report

        # Save to output file if specified
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report, encoding="utf-8")
            click.echo(f"✅ Report saved to: {output}")

        # Print to stdout unless quiet
        if not quiet and not output:
            click.echo(report)

        click.echo("✅ Report generated successfully", err=True)

    except ValueError as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option(
    "--run-id",
    required=True,
    help="MLflow run ID",
)
@click.option(
    "--local-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Local directory to save report",
)
@click.option(
    "--tracking-uri",
    default="http://localhost:5002",
    help="MLflow tracking URI",
)
def download(
    run_id: str,
    local_dir: Path,
    tracking_uri: str,
):
    """
    Download existing report from MLflow.

    Use this when report was already generated and you want a local copy.

    Example:
        python -m src.reports download --run-id abc123 --local-dir runs/xxx/
    """
    generator = ExperimentReportGenerator(tracking_uri)

    try:
        local_path = generator.download_from_mlflow(
            run_id=run_id,
            local_dir=local_dir,
        )

        if local_path:
            click.echo(f"✅ Report downloaded to: {local_path}")
        else:
            click.echo("❌ Report not found in MLflow artifacts", err=True)
            raise SystemExit(1)

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise SystemExit(1)


# Entry point for python -m src.reports
def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
