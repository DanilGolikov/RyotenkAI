"""Schema for the ``reports:`` block in pipeline config YAML.

The block controls which report plugins appear in the post-run Markdown
report and in what order. It's a single source of truth — plugin
manifests do NOT carry section order any more. This means:

- Adding a plugin to ``community/reports/<id>/`` doesn't automatically
  get it into your report — you opt-in by listing its id in ``sections``.
- Two users with the same plugin set can render completely different
  reports just by reordering ``sections``.
- Conflicts between third-party plugins are impossible by design: order
  is your config's, not the author's.
"""

from __future__ import annotations

from pydantic import Field

from src.config.base import StrictBaseModel


class ReportsConfig(StrictBaseModel):
    """Pipeline-level report configuration."""

    sections: list[str] | None = Field(
        default=None,
        description=(
            "Ordered list of report plugin ids to render. "
            "If null/unset, the default built-in section order is used "
            "(see ``DEFAULT_REPORT_SECTIONS`` in src/reports/plugins/defaults.py). "
            "Plugin ids not present in the registry at runtime raise a clear error; "
            "to omit a built-in section, simply leave it out of the list."
        ),
    )


__all__ = ["ReportsConfig"]
