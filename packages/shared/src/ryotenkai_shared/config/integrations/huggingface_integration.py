"""Settings-level HuggingFace integration schema (PR1).

Reusable HuggingFace integration. Holds no secrets in plaintext — the
``HF_TOKEN`` value lives encrypted in ``token.enc`` and is fetched on
demand.

``repo_id`` / ``private`` stay project-local (on ``HuggingFaceRef`` in
PR3), because a single HF account typically pushes to many different
repos.
"""

from __future__ import annotations

from ..base import StrictBaseModel


class HuggingFaceIntegrationConfig(StrictBaseModel):
    """HuggingFace Hub integration (token-carrier)."""


__all__ = [
    "HuggingFaceIntegrationConfig",
]
