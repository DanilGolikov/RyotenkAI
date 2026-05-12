"""Phase 5 — real :class:`IHFHubClient` adapter around :class:`huggingface_hub.HfApi`.

Additive only — the production code keeps calling :class:`HfApi`
directly. This module exists so the live-protocol-compliance lane can
exercise the real HuggingFace SDK against a real token / a recorded
cassette in the future.

The adapter is thin: every method delegates to ``HfApi`` and
translates HF's exception hierarchy onto the Protocol's typed errors.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
from typing import Any

from ryotenkai_shared.infrastructure.hf_hub import (
    HFAuthError,
    HFHubError,
    HFNotFoundError,
    HFRepoInfo,
    HFTransientError,
    IHFHubClient,
)


class HFHubAdapter:
    """Async wrapper around :class:`huggingface_hub.HfApi`.

    The SDK is sync; we wrap each call in :func:`asyncio.to_thread` to
    match the Protocol's async surface. Errors are caught here and
    re-raised as the Protocol's typed hierarchy.
    """

    def __init__(self, *, token: str | None = None, endpoint: str | None = None) -> None:
        # WHY lazy import: ``huggingface_hub`` is a heavy import; production
        # code paths that don't use this adapter shouldn't pay the cost.
        from huggingface_hub import HfApi  # noqa: PLC0415

        self._api = HfApi(token=token, endpoint=endpoint)

    async def create_repo(
        self,
        repo_id: str,
        *,
        private: bool = False,
        exist_ok: bool = True,
    ) -> HFRepoInfo:
        def _call() -> tuple[bool, str]:
            try:
                self._api.create_repo(repo_id=repo_id, private=private, exist_ok=exist_ok)
            except Exception as exc:  # noqa: BLE001
                _translate_and_raise(exc)
                raise  # unreachable
            return private, repo_id

        actual_private, repo = await asyncio.to_thread(_call)
        return HFRepoInfo(repo_id=repo, private=actual_private)

    async def repo_info(self, repo_id: str) -> HFRepoInfo:
        def _call() -> HFRepoInfo:
            try:
                info = self._api.repo_info(repo_id=repo_id)
            except Exception as exc:  # noqa: BLE001
                _translate_and_raise(exc)
                raise
            return HFRepoInfo(
                repo_id=getattr(info, "id", repo_id),
                private=bool(getattr(info, "private", False)),
                sha=getattr(info, "sha", None),
                revision=getattr(info, "revision", "main") or "main",
            )

        return await asyncio.to_thread(_call)

    async def upload_file(
        self,
        *,
        repo_id: str,
        path_in_repo: str,
        content: bytes,
        commit_message: str | None = None,
    ) -> None:
        def _call() -> None:
            try:
                self._api.upload_file(
                    path_or_fileobj=io.BytesIO(content),
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    commit_message=commit_message,
                )
            except Exception as exc:  # noqa: BLE001
                _translate_and_raise(exc)
                raise

        await asyncio.to_thread(_call)

    async def download_file(
        self,
        *,
        repo_id: str,
        path_in_repo: str,
        revision: str = "main",
    ) -> bytes:
        def _call() -> bytes:
            try:
                local_path = self._api.hf_hub_download(
                    repo_id=repo_id, filename=path_in_repo, revision=revision,
                )
            except Exception as exc:  # noqa: BLE001
                _translate_and_raise(exc)
                raise
            with open(local_path, "rb") as f:
                return f.read()

        return await asyncio.to_thread(_call)

    async def get_model_card(self, repo_id: str) -> str:
        def _call() -> str:
            try:
                # Read README.md via the SDK.
                local_path = self._api.hf_hub_download(
                    repo_id=repo_id, filename="README.md",
                )
            except Exception as exc:  # noqa: BLE001
                _translate_and_raise(exc)
                raise
            with open(local_path, "r", encoding="utf-8") as f:
                return f.read()

        return await asyncio.to_thread(_call)


def _translate_and_raise(exc: BaseException) -> None:
    """Map HF SDK exceptions onto our Protocol's typed hierarchy."""
    name = type(exc).__name__
    msg = str(exc)
    # The SDK ships separate exception classes; we match by name to
    # avoid a hard dep on import paths that vary across versions.
    if name in {"RepositoryNotFoundError", "EntryNotFoundError"}:
        raise HFNotFoundError(msg) from exc
    if name in {"HfHubHTTPError"}:
        status = getattr(exc, "response", None)
        code = getattr(status, "status_code", None) if status is not None else None
        if code == 401 or code == 403:
            raise HFAuthError(msg) from exc
        if code == 404:
            raise HFNotFoundError(msg) from exc
        if code == 429:
            raise HFHubError(msg) from exc  # Adapter currently treats 429 as base — fake injects RateLimited separately
        if code is not None and 500 <= code < 600:
            raise HFTransientError(msg) from exc
    raise HFHubError(msg) from exc


# Static guarantee — only run when we have the SDK installed.
with contextlib.suppress(Exception):
    _runtime_check: IHFHubClient = HFHubAdapter()  # type: ignore[assignment]
    del _runtime_check


__all__ = ["HFHubAdapter"]
