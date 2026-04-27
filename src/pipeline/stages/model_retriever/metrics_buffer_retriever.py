"""
Phase 12.A — retrieve ``metrics_buffer.jsonl`` from a remote pod.

Bridges the gap between the trainer's
:class:`~src.training.mlflow.metrics_buffer.MetricsBuffer` (which writes
to ``/workspace/metrics_buffer.jsonl`` on the pod) and the Mac-side
:class:`~src.pipeline.stages.model_retriever.metrics_replay.BufferedMetricsReplay`
which replays the file's contents into MLflow.

Design constraints
------------------
* **Best-effort** — never raises into the caller. Returns a
  :class:`FetchResult` describing what happened so the stage can log
  + emit a callback but continue with HF upload either way.
* **Size-capped** — refuses to download files larger than
  :data:`MAX_BUFFER_SIZE_BYTES` (default 100 MiB) to protect the
  Mac's disk against a malformed / runaway buffer. Practical buffer
  files are < 10 MiB even on a 24-hour run with ``keep_all=true``.
* **Single-file SCP** — uses :meth:`SSHClient.download_file` rather
  than the tar-pipeline ``download_directory``. No directory
  semantics needed; one tiny JSONL is faster as plain SCP.
* **Stage-quiet on missing file** — the typical happy path is
  "buffer absent" (Phase 11.A's :meth:`CompletionCallback.on_train_end`
  successfully drained on natural completion). Missing file is NOT
  an error; it's the expected outcome for healthy runs.

Caller contract
---------------
:class:`~src.pipeline.stages.model_retriever.retriever.ModelRetriever`
calls :meth:`fetch` after HF upload completes but BEFORE
:meth:`provider.cleanup_pod` runs (which would otherwise reclaim the
volume). Phase 12.A.1 plan § 3.2 wires this in
``_execute_retrieval`` after the local download / HF upload branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from src.utils.logger import get_logger
from src.utils.ssh_client import SSHClient

logger = get_logger(__name__)


@dataclass(frozen=True)
class FetchResult:
    """Outcome of a single buffer-fetch attempt.

    Attributes:
        local_path: Absolute path of the downloaded file on the Mac;
                    ``None`` when the file was absent or oversized
                    (skip path).
        size_bytes: Size of the remote file in bytes; ``0`` when the
                    file was absent.
        line_count: Number of lines in the local file (cheap cross-
                    check). ``0`` when ``local_path is None``.
        missing:    ``True`` iff the remote file did not exist.
                    Distinct from "exists but oversized" — a missing
                    file means the trainer's drain already succeeded
                    (or no metrics were ever buffered), which is the
                    healthy case.
        oversized:  ``True`` iff the file exists on the pod but
                    exceeds :data:`MAX_BUFFER_SIZE_BYTES` and was
                    NOT downloaded. Surfaced separately from
                    ``missing`` so the operator can see "we deliberately
                    skipped a runaway buffer" in logs.
        error:      Human-readable error message when the fetch
                    failed for an unexpected reason (SSH dead, disk
                    full on Mac, etc.). ``None`` on the happy path
                    OR on the missing/oversized skip paths.
    """

    local_path: Path | None
    size_bytes: int = 0
    line_count: int = 0
    missing: bool = False
    oversized: bool = False
    error: str | None = None


class MetricsBufferRetriever:
    """Fetch the trainer's metrics buffer from a remote pod over SSH."""

    REMOTE_BUFFER_FILENAME: ClassVar[str] = "metrics_buffer.jsonl"
    """File name written by :class:`MetricsBuffer` (mirrors
    ``_DEFAULT_BUFFER_FILENAME`` in ``metrics_buffer.py``)."""

    REMOTE_FLUSH_OFFSET_RELPATH: ClassVar[str] = ".runner/buffer.flush_offset"
    """Marker file written by :class:`ResilientMLflowTransport` after
    each successful drain. Best-effort retrieved alongside the buffer
    for forensics; not required for replay correctness (per
    phase-12 § 2.7 invariant: buffer file = un-flushed only)."""

    MAX_BUFFER_SIZE_BYTES: ClassVar[int] = 100 * 1024 * 1024  # 100 MiB
    """Hard cap above which we skip download. Practical buffer files
    are < 10 MiB even on multi-hour runs. Anything larger suggests a
    bug or attack and we'd rather skip than risk the Mac's disk."""

    SSH_QUICK_TIMEOUT_S: ClassVar[int] = 15
    """Timeout for ``test -f`` / ``stat`` probe commands."""

    SCP_DOWNLOAD_TIMEOUT_S: ClassVar[int] = 300
    """Timeout for the SCP download itself. 5 min is generous for a
    100 MiB file even on a slow link."""

    def __init__(
        self,
        ssh_client: SSHClient,
        *,
        workspace_path: str = "/workspace",
    ) -> None:
        """
        Args:
            ssh_client:     Established SSH connection to the pod
                            (from :class:`ModelRetriever._ssh_client`).
            workspace_path: Remote workspace root (``/workspace`` on
                            both RunPod and single_node by default).
        """
        self._ssh = ssh_client
        self._workspace = workspace_path.rstrip("/")
        self._remote_buffer_path = (
            f"{self._workspace}/{self.REMOTE_BUFFER_FILENAME}"
        )
        self._remote_flush_offset_path = (
            f"{self._workspace}/{self.REMOTE_FLUSH_OFFSET_RELPATH}"
        )

    @property
    def remote_buffer_path(self) -> str:
        """Remote path of the buffer file (for logging / tests)."""
        return self._remote_buffer_path

    def fetch(self, *, local_dir: Path) -> FetchResult:
        """Probe + download the buffer file into ``local_dir``.

        Steps:
        1. ``test -f`` probe — missing → return early.
        2. ``stat -c %s`` — oversized → return early.
        3. SCP download → ``local_dir / metrics_buffer.jsonl``.
        4. Best-effort fetch the optional ``.flush_offset`` marker
           into ``local_dir / buffer.flush_offset.json`` (forensics
           only; missing is normal).
        5. Count lines in the local file (cross-check).

        Args:
            local_dir: Directory to write the local copy into.
                       Created if missing. Typically the per-attempt
                       directory (``attempts/<n>/``).

        Returns:
            :class:`FetchResult`. Never raises.
        """
        # 0. Ensure the local directory exists. Defensive — caller is
        # expected to pass an existing attempts dir, but treating this
        # as best-effort means a single missing parent doesn't tank
        # the whole stage.
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return FetchResult(
                local_path=None,
                error=f"local_dir mkdir failed: {exc}",
            )

        # 1. Probe — file exists?
        if not self._ssh.file_exists(self._remote_buffer_path):
            return FetchResult(local_path=None, missing=True)

        # 2. Probe — size acceptable?
        size_bytes = self._remote_size_bytes(self._remote_buffer_path)
        if size_bytes is None:
            # Probe itself failed (SSH dead). Treat as missing rather
            # than blocking the stage, but capture the error.
            return FetchResult(
                local_path=None,
                missing=True,
                error="failed to stat remote buffer (SSH probe error)",
            )

        if size_bytes > self.MAX_BUFFER_SIZE_BYTES:
            logger.warning(
                "[METRICS_REPLAY] remote buffer oversized: %d bytes "
                "(>%d cap) — skipping download to protect Mac disk",
                size_bytes, self.MAX_BUFFER_SIZE_BYTES,
            )
            return FetchResult(
                local_path=None,
                size_bytes=size_bytes,
                oversized=True,
            )

        # 3. SCP the buffer file.
        local_path = local_dir / self.REMOTE_BUFFER_FILENAME
        download_result = self._ssh.download_file(
            self._remote_buffer_path,
            local_path,
            timeout=self.SCP_DOWNLOAD_TIMEOUT_S,
        )
        if download_result.is_failure():
            err = download_result.unwrap_err()
            err_msg = str(err) if err is not None else "unknown SCP error"
            logger.warning(
                "[METRICS_REPLAY] SCP download failed: %s", err_msg,
            )
            return FetchResult(
                local_path=None,
                size_bytes=size_bytes,
                error=err_msg,
            )

        # 4. Best-effort fetch the flush_offset marker. NOT required
        #    for replay correctness — purely forensic.
        offset_local = local_dir / "buffer.flush_offset.json"
        if self._ssh.file_exists(self._remote_flush_offset_path):
            self._ssh.download_file(
                self._remote_flush_offset_path,
                offset_local,
                timeout=self.SSH_QUICK_TIMEOUT_S,
            )  # ignored — pure forensics

        # 5. Local sanity check — count lines.
        line_count = self._count_lines(local_path)

        logger.info(
            "[METRICS_REPLAY] fetched buffer: %d bytes, %d lines -> %s",
            size_bytes, line_count, local_path,
        )

        return FetchResult(
            local_path=local_path,
            size_bytes=size_bytes,
            line_count=line_count,
            missing=False,
            oversized=False,
            error=None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _remote_size_bytes(self, remote_path: str) -> int | None:
        """Return file size in bytes via ``stat -c %s``.

        Returns ``None`` when the SSH command fails or the output
        cannot be parsed — caller treats that as a probe failure and
        skips the download.
        """
        # ``stat -c %s`` works on every Linux pod we ship to. macOS
        # uses ``stat -f %z`` but pods are all Linux.
        success, stdout, _ = self._ssh.exec_command(
            f"stat -c %s {remote_path}",
            timeout=self.SSH_QUICK_TIMEOUT_S,
            silent=True,
        )
        if not success:
            return None
        try:
            return int(stdout.strip())
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _count_lines(path: Path) -> int:
        """Count non-empty lines in a local file. Best-effort; returns
        ``0`` on any read error."""
        try:
            with path.open(encoding="utf-8") as fh:
                return sum(1 for line in fh if line.strip())
        except OSError:
            return 0


__all__ = ["FetchResult", "MetricsBufferRetriever"]
