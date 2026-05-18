"""System prompt loading with bounded cache + explicit failure modes.

Public surface:
    - :class:`SystemPromptLoader` — caching loader that resolves a system
      prompt from MLflow Prompt Registry or a local file.
    - :class:`SystemPromptResult` — value object carrying ``text`` plus
      ``source`` audit metadata.

This package replaces the static loader at
``ryotenkai_shared.infrastructure.mlflow.system_prompt`` (kept as a
deprecated shim until M4). See
``docs/plans/vectorized-fluttering-mist.md`` section "D. system_prompt.py".
"""

from __future__ import annotations

from ryotenkai_shared.inference.prompts.system_prompt_loader import (
    SystemPromptLoader,
    SystemPromptResult,
)

__all__ = [
    "SystemPromptLoader",
    "SystemPromptResult",
]
