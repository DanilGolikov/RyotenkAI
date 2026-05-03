"""Cross-cutting pipeline run-context types.

Lived under ``ryotenkai_control.pipeline.state.run_context`` until ADR
row 9 (Phase C drift fix) — providers needed it for type annotations
(``connect(*, run: RunContext)``) but were forbidden from importing
control. Moved here as a shared dataclass.
"""

from __future__ import annotations

from ryotenkai_shared.pipeline_context.run_context import RunContext

__all__ = ["RunContext"]
