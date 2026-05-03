"""
E2E Test for Report Generation.
Connects to real MLflow server if available.
"""

import pytest
import requests

from src.reports.report_generator import ExperimentReportGenerator

TRACKING_URI = "http://localhost:5002"
# Run ID provided by user for reproduction
REAL_RUN_ID = "28b01411736546cd8d41bbb49ecb9228"


def is_mlflow_available():
    try:
        response = requests.get(TRACKING_URI, timeout=1)
        return response.status_code == 200
    except Exception:
        return False


@pytest.mark.e2e
@pytest.mark.skip(reason="Outdated run ID (28b01411...) no longer exists in MLflow")
@pytest.mark.skipif(not is_mlflow_available(), reason="MLflow server not running")
def test_generate_report_real_run():
    """
    Test report generation against a real local MLflow instance.
    Uses the run ID specified by user for verification.
    """
    print(f"\nConnecting to {TRACKING_URI}...")
    generator = ExperimentReportGenerator(TRACKING_URI)

    # Generate model only to check data
    report = generator.generate_report_model(REAL_RUN_ID)

    # 1. Check Structure
    print(f"Found {len(report.phases)} phases")
    assert len(report.phases) == 3, "Expected 3 phases (CPT, SFT, COT)"

    # 2. Check Strategies
    strategies = [p.strategy for p in report.phases]
    assert strategies == ["CPT", "SFT", "COT"]

    # 3. Check Metrics
    # Based on our manual inspection, phase 0 had train_loss ~4.3
    assert report.phases[0].final_loss is not None
    assert report.phases[0].final_loss > 4.0

    # 4. Check Config & Model Info (CRITICAL FIX CHECK)
    assert report.config.learning_rate is not None
    assert report.model.total_parameters is not None, "Total Parameters missing"
    assert report.model.trainable_parameters is not None, "Trainable Parameters missing"
    assert report.model.name != "Unknown", "Model Name is Unknown"
    assert report.model.loading_time_seconds is not None, "Loading Time missing"

    print(f"Model: {report.model.name}")
    print(f"Params: Total={report.model.total_parameters}, Trainable={report.model.trainable_parameters}")
    print(f"Load Time: {report.model.loading_time_seconds}s")

    # 5. Check GPU Info (CRITICAL FIX CHECK)
    assert report.resources.gpu_name is not None, "GPU Name missing"
    assert (
        report.resources.gpu_vram_gb is not None or report.resources.total_vram_gb is not None
    ), "GPU VRAM missing (checked gpu_vram_gb and total_vram_gb)"
    # Note: resources object might have different field names based on my last edit, let's check
    # In builder I used: gpu_vram_gb=...
    print(f"GPU Info: Name={report.resources.gpu_name}, VRAM={report.resources.gpu_vram_gb}")

    # 6. Check Rendering
    markdown = generator.generate(REAL_RUN_ID)
    assert "Experiment Report" in markdown
    assert "CPT" in markdown
    assert "SFT" in markdown

    # Check that placeholders "—" are NOT present for critical fields
    assert "| Total Parameters | — |" not in markdown, "Markdown still shows empty Total Parameters"
    assert "| VRAM | — |" not in markdown, "Markdown still shows empty VRAM"

    print("✅ Real run report generated successfully with ALL fields")
