"""Phase 4 — Provider-agnostic ``IHFHubClient`` Protocol.

Extracted additively in 2026-05-11. HuggingFace Hub usage in production
is concentrated in
:class:`ryotenkai_control.pipeline.stages.model_retriever.hf_uploader.HFModelUploader`
(upload via SSH + ``huggingface-cli``, repo lifecycle via SDK
:class:`huggingface_hub.HfApi`). The Protocol narrows that into a
five-method surface large enough to drive component tests of upload
flows + repo lifecycle without booking real HF API calls.

**Definition-only**: production code still calls
:class:`huggingface_hub.HfApi` directly; the Protocol exists so the
fake (and a future thin wrapper) can satisfy DI for component tests.
The compliance test parametrizes over ``[fake, real]``; ``real`` is
``pytest.skip``-ed until a wrapper lands.

The five typed errors mirror the practical failure modes that the
:func:`classify_hf_upload_error` taxonomy already recognises:

* :class:`HFAuthError` — 401/403 / bad token
* :class:`HFNotFoundError` — 404 / unknown repo
* :class:`HFRateLimitedError` — 429
* :class:`HFTransientError` — 5xx / connection drop (retryable)
* :class:`HFHubError` — base / catch-all
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class HFHubError(Exception):
    """Base error for the ``IHFHubClient`` surface."""


class HFAuthError(HFHubError):
    """401 / 403 — invalid token or insufficient permissions."""


class HFNotFoundError(HFHubError):
    """404 — repo or file does not exist."""


class HFRateLimitedError(HFHubError):
    """429 — caller should back off and retry."""


class HFTransientError(HFHubError):
    """5xx / connection drop — caller should retry."""


@dataclass(frozen=True)
class HFRepoInfo:
    """Snapshot of a HF Hub repo's observable state."""

    repo_id: str
    private: bool
    sha: str | None = None
    revision: str = "main"


@runtime_checkable
class IHFHubClient(Protocol):
    """Async surface for HuggingFace Hub interactions.

    Methods mirror the subset of :class:`huggingface_hub.HfApi` actually
    used by production today (repo CRUD + file upload + file download +
    model-card read).
    """

    async def create_repo(
        self,
        repo_id: str,
        *,
        private: bool = False,
        exist_ok: bool = True,
    ) -> HFRepoInfo:
        """Create or fetch a repo; idempotent when ``exist_ok=True``."""
        ...

    async def repo_info(self, repo_id: str) -> HFRepoInfo:
        """Fetch metadata for ``repo_id``.

        Raises :class:`HFNotFoundError` when the repo does not exist.
        """
        ...

    async def upload_file(
        self,
        *,
        repo_id: str,
        path_in_repo: str,
        content: bytes,
        commit_message: str | None = None,
    ) -> None:
        """Upload ``content`` to ``path_in_repo`` in ``repo_id``."""
        ...

    async def download_file(
        self,
        *,
        repo_id: str,
        path_in_repo: str,
        revision: str = "main",
    ) -> bytes:
        """Download ``path_in_repo`` from ``repo_id`` at ``revision``.

        Raises :class:`HFNotFoundError` when the file does not exist.
        """
        ...

    async def get_model_card(self, repo_id: str) -> str:
        """Return the rendered README.md content for ``repo_id``.

        Raises :class:`HFNotFoundError` if the repo has no README.
        """
        ...


__all__ = [
    "HFAuthError",
    "HFHubError",
    "HFNotFoundError",
    "HFRateLimitedError",
    "HFRepoInfo",
    "HFTransientError",
    "IHFHubClient",
]
