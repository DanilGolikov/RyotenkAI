"""HTTP-based replacement for the legacy SSH ``LogManager``.

Phase 2 PR-2.3 of transport-unification-v2 — pulls remote log
files from the pod via ``GET /api/v1/logs/{name}`` instead of the
old ``ssh stat`` + ``tail -c`` protocol. Same on-disk artifact
contract: writes the live trainer / runner output into
``runs/<id>/attempts/<n>/logs/{trainer.stdio,runner}.log`` so the
web UI's ``LogDock`` keeps tailing the same files.

Why a new class instead of mutating ``LogManager``: SSH-side
implementation hard-codes a paramiko :class:`SSHClient`. Threading
HTTP through that class would force every test fixture to grow a
JobClient; cleaner to introduce a sibling and delete the SSH one
in PR-2.3 along with its tests.

API parity with ``LogManager``:

* ``download(silent=True) -> bool``
* ``download_on_error(error_context="")``
* ``get_last_lines(n=30) -> list[str]``
* ``local_path`` property
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ryotenkai_shared.utils.logger import get_logger, get_run_log_layout

if TYPE_CHECKING:
    from pathlib import Path

    from ryotenkai_shared.contracts.runner_api.logs import LogName
    from ryotenkai_shared.utils.clients.job_client import JobClient

logger = get_logger(__name__)

ENCODING_UTF8 = "utf-8"
LOG_REMOTE_NOT_FOUND_MSG = "⚠️ Remote log not found yet"
DEFAULT_RANGE_CHUNK_BYTES = 64 * 1024  # 64 KB per HTTP poll
TAIL_LOOKBACK_BYTES = 64 * 1024        # for get_last_lines: read last 64 KB


class LogFetcher:
    """Mirror of :class:`LogManager` but driven by HTTP via JobClient.

    Same on-disk contract: appends only newly-arrived bytes between
    polls so the local file is a faithful artifact of the remote
    log. Handles remote rotation/truncation by re-downloading from
    offset 0 (same recovery as LogManager).
    """

    def __init__(
        self,
        client: "JobClient",
        *,
        name: "LogName",
        local_path: "Path",
    ) -> None:
        from ryotenkai_shared.contracts.runner_api.logs import LogName as _LN

        if not isinstance(name, _LN):
            raise TypeError(f"name must be LogName enum, got {type(name).__name__}")
        self._client = client
        self._name = name
        layout = get_run_log_layout()
        layout.ensure_logs_dir()
        self._local_path = local_path
        # Track BYTE offset on the remote log — single source of truth
        # for incremental polling. Initialised from local file size so
        # a runner restart picks up where it left off.
        self._offset = self._local_size_bytes()

    # ----- public surface ------------------------------------------------

    @property
    def local_path(self) -> "Path":
        return self._local_path

    @property
    def name(self) -> "LogName":
        return self._name

    def download(self, *, silent: bool = True) -> bool:
        """Synchronous wrapper — reads the remote tail via HTTP and
        appends the new bytes to ``self._local_path``.

        Returns True on success (including "no new bytes" — that's a
        legitimate poll outcome). False when the remote file isn't
        available yet.
        """
        try:
            return asyncio.run(self._download_async(silent=silent))
        except Exception as exc:  # noqa: BLE001 — defensive (sync seam)
            logger.debug(f"[LOG_FETCHER] {self._name.value}: download failed: {exc}")
            return False

    def download_on_error(self, error_context: str = "") -> None:
        if error_context:
            logger.info(
                f"📋 Downloading {self._local_path.name} due to: {error_context}"
            )
        self.download(silent=False)

    def get_last_lines(self, n: int = 30) -> list[str]:
        """Return the last ``n`` lines of the remote log.

        Approximation: fetch the last ``TAIL_LOOKBACK_BYTES`` bytes,
        split, return the last ``n`` lines. For typical training
        logs (one line per training step) 64 KB ≫ 30 lines so the
        approximation is exact in practice. If the file is shorter
        than the lookback window we read the whole thing.
        """
        try:
            return asyncio.run(self._get_last_lines_async(n=n))
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.debug(f"[LOG_FETCHER] {self._name.value}: get_last_lines failed: {exc}")
            return []

    # ----- async core ----------------------------------------------------

    async def _download_async(self, *, silent: bool) -> bool:
        from ryotenkai_shared.utils.clients.problem_details import APIException

        try:
            chunk = await self._client.read_log(
                self._name.value,
                offset=self._offset,
                limit_bytes=DEFAULT_RANGE_CHUNK_BYTES,
            )
        except APIException as exc:
            from ryotenkai_shared.contracts.problem_details import ErrorCode

            if exc.code == ErrorCode.LOG_NOT_AVAILABLE:
                logger.debug(LOG_REMOTE_NOT_FOUND_MSG)
                return False
            if exc.code == ErrorCode.LOG_OFFSET_OUT_OF_RANGE:
                # Remote log rotated/truncated — reset cursor + retry
                # in one go. ``LogManager`` had the same recovery (the
                # ``remote_size < local_size`` branch).
                logger.debug(
                    f"[LOG_FETCHER] {self._name.value}: cursor out of range; "
                    f"resetting from offset 0"
                )
                self._offset = 0
                # Truncate local file so the artifact stays consistent
                # with the post-rotation remote.
                self._local_path.write_text("", encoding=ENCODING_UTF8)
                chunk = await self._client.read_log(
                    self._name.value,
                    offset=0,
                    limit_bytes=DEFAULT_RANGE_CHUNK_BYTES,
                )
            else:
                raise

        if chunk.content:
            with self._local_path.open("a", encoding=ENCODING_UTF8) as fh:
                fh.write(chunk.content)
            self._offset = chunk.next_offset
            if not silent:
                logger.info(
                    f"📥 {self._local_path.name} updated: "
                    f"{chunk.total_size:,} bytes ({len(chunk.content)} new)"
                )
        elif not silent:
            logger.info(
                f"📥 {self._local_path.name}: no new bytes "
                f"({chunk.total_size:,} total)"
            )
        return True

    async def _get_last_lines_async(self, *, n: int) -> list[str]:
        size_resp = await self._client.get_log_size(self._name.value)
        total = size_resp.size_bytes
        if total <= 0:
            return []
        offset = max(0, total - TAIL_LOOKBACK_BYTES)
        chunk = await self._client.read_log(
            self._name.value,
            offset=offset,
            limit_bytes=TAIL_LOOKBACK_BYTES,
        )
        if not chunk.content:
            return []
        # Strip torn first line (we may have started mid-line) by
        # discarding the bit before the first newline. Only safe when
        # offset > 0.
        body = chunk.content
        if offset > 0:
            nl = body.find("\n")
            if nl >= 0:
                body = body[nl + 1 :]
        return body.splitlines()[-n:]

    # ----- helpers -------------------------------------------------------

    def _local_size_bytes(self) -> int:
        try:
            return int(self._local_path.stat().st_size)
        except (FileNotFoundError, OSError):
            return 0


__all__ = ["LogFetcher"]
