"""Default section order for the built-in report plugins.

This is the fallback the report generator uses when the pipeline config
doesn't declare ``reports.sections``. Changing a plugin's file in
``community/reports/<id>/`` does NOT change where it lands — order lives
here and in user config, not in plugin manifests.

Adding a new built-in: drop the plugin folder into ``community/reports/``
and add its id to this tuple at the position you want it to render.
"""

from __future__ import annotations

from typing import Final

DEFAULT_REPORT_SECTIONS: Final[tuple[str, ...]] = (
    "header",
    "summary",
    "issues",
    "dataset_validation",
    "evaluation_block",
    "model_configuration",
    "memory_management",
    "training_configuration",
    "phase_details",
    "metrics_analysis",
    "stage_timeline",
    "config_dump",
    "footer",
)

__all__ = ["DEFAULT_REPORT_SECTIONS"]
