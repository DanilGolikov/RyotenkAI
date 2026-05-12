"""``FakeHFHubClient`` — canonical fake for :class:`IHFHubClient`.

In-memory model registry. State:

* repos dict (``repo_id -> _FakeRepo``)
* files dict per repo (``path_in_repo -> bytes``)
* commit log (every upload appends an entry)

Chaos surface:

* :meth:`inject_rate_limit` — next N calls raise :class:`HFRateLimitedError`
* :meth:`inject_5xx` — next N calls raise :class:`HFTransientError`
* :meth:`inject_auth_failure` — next call raises :class:`HFAuthError`
* :meth:`inject_corrupted_download` — next ``download_file`` returns
  garbage bytes (a deliberate copy of mangled content)
* :meth:`reset_chaos`
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Any

from ryotenkai_shared.infrastructure.hf_hub import (
    HFAuthError,
    HFNotFoundError,
    HFRateLimitedError,
    HFRepoInfo,
    HFTransientError,
    IHFHubClient,
)
from tests._harness.clock import Clock, RealClock


@dataclass
class _Commit:
    sha: str
    path_in_repo: str
    size_bytes: int
    message: str | None


@dataclass
class _FakeRepo:
    repo_id: str
    private: bool
    files: dict[str, bytes] = field(default_factory=dict)
    commits: list[_Commit] = field(default_factory=list)
    model_card: str = ""


@dataclass
class _ChaosState:
    rate_limited_remaining: int = 0
    transient_remaining: int = 0
    auth_failure_next: bool = False
    corrupted_download_remaining: int = 0


class FakeHFHubClient:
    """Deterministic in-memory fake for :class:`IHFHubClient`."""

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock: Clock = clock if clock is not None else RealClock()
        self._repos: dict[str, _FakeRepo] = {}
        self._chaos = _ChaosState()
        self._sha_counter = itertools.count(start=1)

    # ------------------------------------------------------------------
    # Chaos surface
    # ------------------------------------------------------------------

    def inject_rate_limit(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._chaos.rate_limited_remaining = count

    def inject_5xx(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._chaos.transient_remaining = count

    def inject_auth_failure(self) -> None:
        self._chaos.auth_failure_next = True

    def inject_corrupted_download(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._chaos.corrupted_download_remaining = count

    def reset_chaos(self) -> None:
        self._chaos = _ChaosState()

    # ------------------------------------------------------------------
    # Inspection helpers (test-only)
    # ------------------------------------------------------------------

    def list_repos(self) -> list[str]:
        return sorted(self._repos.keys())

    def get_repo(self, repo_id: str) -> _FakeRepo:
        return self._repos[repo_id]

    def set_model_card(self, repo_id: str, content: str) -> None:
        if repo_id not in self._repos:
            raise HFNotFoundError(f"unknown repo: {repo_id!r}")
        self._repos[repo_id].model_card = content

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "repos": {
                rid: {
                    "private": repo.private,
                    "files": {
                        path: {"size": len(content)} for path, content in repo.files.items()
                    },
                    "commits": [
                        {
                            "sha": c.sha,
                            "path": c.path_in_repo,
                            "size": c.size_bytes,
                            "message": c.message,
                        }
                        for c in repo.commits
                    ],
                    "model_card_size": len(repo.model_card),
                }
                for rid, repo in self._repos.items()
            },
            "chaos": {
                "rate_limited_remaining": self._chaos.rate_limited_remaining,
                "transient_remaining": self._chaos.transient_remaining,
                "auth_failure_next": self._chaos.auth_failure_next,
                "corrupted_download_remaining": self._chaos.corrupted_download_remaining,
            },
        }

    # ------------------------------------------------------------------
    # IHFHubClient surface
    # ------------------------------------------------------------------

    async def create_repo(
        self,
        repo_id: str,
        *,
        private: bool = False,
        exist_ok: bool = True,
    ) -> HFRepoInfo:
        self._fire_chaos()
        if repo_id in self._repos:
            if not exist_ok:
                raise HFAuthError(f"repo already exists: {repo_id!r}")
            existing = self._repos[repo_id]
            return HFRepoInfo(
                repo_id=existing.repo_id, private=existing.private, sha=self._latest_sha(existing),
            )
        repo = _FakeRepo(repo_id=repo_id, private=private)
        self._repos[repo_id] = repo
        return HFRepoInfo(repo_id=repo_id, private=private, sha=None)

    async def repo_info(self, repo_id: str) -> HFRepoInfo:
        self._fire_chaos()
        if repo_id not in self._repos:
            raise HFNotFoundError(f"unknown repo: {repo_id!r}")
        repo = self._repos[repo_id]
        return HFRepoInfo(
            repo_id=repo.repo_id, private=repo.private, sha=self._latest_sha(repo),
        )

    async def upload_file(
        self,
        *,
        repo_id: str,
        path_in_repo: str,
        content: bytes,
        commit_message: str | None = None,
    ) -> None:
        self._fire_chaos()
        if repo_id not in self._repos:
            raise HFNotFoundError(f"unknown repo: {repo_id!r}")
        repo = self._repos[repo_id]
        repo.files[path_in_repo] = content
        sha = self._new_sha(content)
        repo.commits.append(
            _Commit(
                sha=sha,
                path_in_repo=path_in_repo,
                size_bytes=len(content),
                message=commit_message,
            ),
        )
        # Mirror HfApi's README convention: uploading ``README.md``
        # populates the model card so ``get_model_card`` returns it.
        if path_in_repo == "README.md":
            with contextlib.suppress(UnicodeDecodeError):
                repo.model_card = content.decode("utf-8")

    async def download_file(
        self,
        *,
        repo_id: str,
        path_in_repo: str,
        revision: str = "main",
    ) -> bytes:
        self._fire_chaos()
        if repo_id not in self._repos:
            raise HFNotFoundError(f"unknown repo: {repo_id!r}")
        repo = self._repos[repo_id]
        if path_in_repo not in repo.files:
            raise HFNotFoundError(
                f"unknown file in repo {repo_id!r}: {path_in_repo!r}",
            )
        content = repo.files[path_in_repo]
        if self._chaos.corrupted_download_remaining > 0:
            self._chaos.corrupted_download_remaining -= 1
            # Return mangled content: prefix + truncate to simulate a
            # partial download race.
            return b"\x00CORRUPT\x00" + content[: max(0, len(content) // 2)]
        return content

    async def get_model_card(self, repo_id: str) -> str:
        self._fire_chaos()
        if repo_id not in self._repos:
            raise HFNotFoundError(f"unknown repo: {repo_id!r}")
        repo = self._repos[repo_id]
        if not repo.model_card:
            raise HFNotFoundError(f"no README.md in repo {repo_id!r}")
        return repo.model_card

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fire_chaos(self) -> None:
        # Order: auth (cheapest, fast-fail), rate-limit, transient.
        if self._chaos.auth_failure_next:
            self._chaos.auth_failure_next = False
            raise HFAuthError("fake injected auth failure (401)")
        if self._chaos.rate_limited_remaining > 0:
            self._chaos.rate_limited_remaining -= 1
            raise HFRateLimitedError("fake injected 429 rate limit")
        if self._chaos.transient_remaining > 0:
            self._chaos.transient_remaining -= 1
            raise HFTransientError("fake injected 5xx transient")

    def _new_sha(self, content: bytes) -> str:
        # Mix in the counter so repeated identical uploads still produce
        # distinct commit SHAs (mirrors HF's commit-per-upload model).
        n = next(self._sha_counter)
        h = hashlib.sha1(content + str(n).encode()).hexdigest()
        return h[:12]

    def _latest_sha(self, repo: _FakeRepo) -> str | None:
        if not repo.commits:
            return None
        return repo.commits[-1].sha


# Static guarantee.
_runtime_check: IHFHubClient = FakeHFHubClient()


__all__ = [
    "FakeHFHubClient",
]
