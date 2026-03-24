"""
Log Manager - downloads and manages logs from remote training.

Provider-agnostic log management for any GPU provider (RunPod, SingleNode, etc.)

Responsibilities:
- Download training log from remote server (incremental delta updates)
- Save to run-specific directory (runs/{timestamp}/)
- Called periodically during monitoring and on errors

Usage:
    log_manager = LogManager(ssh_client, remote_path="/workspace/training.log")
    log_manager.download()  # Downloads to runs/{timestamp}/training.log
    last_lines = log_manager.get_last_lines(30)  # For display
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.pipeline.constants import ENCODING_UTF8, LOG_REMOTE_NOT_FOUND_MSG
from src.utils.logger import get_logger, get_run_log_dir

if TYPE_CHECKING:
    from pathlib import Path

    from src.utils.ssh_client import SSHClient

logger = get_logger(__name__)


class LogManager:
    """
    Manages downloading logs from remote training.

    Provider-agnostic: works with any SSH-accessible server.
    All logs are saved to the run-specific directory:
    runs/{pipeline_timestamp}/training.log
    """

    DEFAULT_REMOTE_PATH = "/workspace/training.log"
    LOCAL_LOG_NAME = "training.log"

    def __init__(self, ssh_client: SSHClient, remote_path: str | None = None):
        """
        Initialize log manager.

        Args:
            ssh_client: SSH client connected to remote host
            remote_path: Path to training log on remote (default: /workspace/training.log)
        """
        self.ssh = ssh_client
        self._remote_path = remote_path or self.DEFAULT_REMOTE_PATH
        self._log_dir = get_run_log_dir()
        self._local_path = self._log_dir / self.LOCAL_LOG_NAME
        # Track last downloaded size in BYTES (not characters).
        # Used to incrementally append only new bytes on subsequent downloads.
        self._last_size = self._get_local_size_bytes()

    @property
    def local_path(self) -> Path:
        """Path to local training log file (run-scoped)."""
        return self._local_path

    def _get_local_size_bytes(self) -> int:
        try:
            return int(self._local_path.stat().st_size)
        except FileNotFoundError:
            return 0
        except OSError:
            return 0

    def _get_remote_size_bytes(self) -> int | None:
        """
        Get remote log file size in bytes.

        Returns None when file is missing/unreadable.
        """
        # Prefer stat (metadata-only, O(1)) to avoid scanning large logs on each poll.
        for cmd in (
            f"stat -c%s {self._remote_path} 2>/dev/null || echo 'LOG_NOT_FOUND'",
            f"wc -c < {self._remote_path} 2>/dev/null || echo 'LOG_NOT_FOUND'",
        ):
            success, out, _ = self.ssh.exec_command(command=cmd, silent=True, timeout=10)
            if not success or "LOG_NOT_FOUND" in out:
                continue
            s = (out or "").strip()
            try:
                return int(s)
            except ValueError:
                continue

        return None

    def _download_full(self) -> tuple[bool, str]:
        """Download full remote log content (fallback path)."""
        success, content, _ = self.ssh.exec_command(
            command=f"cat {self._remote_path} 2>/dev/null || echo 'LOG_NOT_FOUND'",
            silent=True,
            timeout=60,
        )
        if not success or "LOG_NOT_FOUND" in content:
            return False, ""
        return True, content

    def download(self, silent: bool = True) -> bool:
        """
        Download training log from remote training workspace.

        Optimized:
        - First download writes the full file.
        - Subsequent downloads append only newly added bytes (delta) to reduce SSH traffic.

        Args:
            silent: If True, don't log success message (for periodic downloads)

        Returns:
            True if download successful, False otherwise
        """
        local_size = self._get_local_size_bytes()
        prev_size = self._last_size

        # First download (or local file was deleted) -> full fetch.
        if local_size <= 0:
            ok, content = self._download_full()
            if not ok:
                logger.debug(LOG_REMOTE_NOT_FOUND_MSG)
                return False
            self._local_path.write_text(content, encoding=ENCODING_UTF8)
            current_size = self._get_local_size_bytes()
            if not silent:
                logger.info(f"📥 Downloaded training log: {self._local_path} ({current_size:,} bytes)")
            elif current_size != prev_size:
                logger.debug(f"📥 Training log updated: {current_size:,} bytes")
            self._last_size = current_size
            return True

        # Incremental path (append only new bytes).
        remote_size = self._get_remote_size_bytes()
        if remote_size is None:
            logger.debug(LOG_REMOTE_NOT_FOUND_MSG)
            return False

        # Remote log truncated/rotated -> re-download full to keep local artifact consistent.
        if remote_size < local_size:
            ok, content = self._download_full()
            if not ok:
                logger.debug(LOG_REMOTE_NOT_FOUND_MSG)
                return False
            self._local_path.write_text(content, encoding=ENCODING_UTF8)
            current_size = self._get_local_size_bytes()
            if not silent:
                logger.info(f"📥 Downloaded training log: {self._local_path} ({current_size:,} bytes)")
            elif current_size != prev_size:
                logger.debug(f"📥 Training log updated: {current_size:,} bytes")
            self._last_size = current_size
            return True

        # No new content.
        if remote_size == local_size:
            self._last_size = local_size
            return True

        delta_bytes = remote_size - local_size
        # Fetch only newly appended bytes (fast path).
        success, delta, _ = self.ssh.exec_command(
            command=f"tail -c {delta_bytes} {self._remote_path} 2>/dev/null || echo 'LOG_NOT_FOUND'",
            silent=True,
            timeout=60,
        )
        if not success or "LOG_NOT_FOUND" in delta:
            # Fallback: if tail is unavailable or failed, fetch full file to avoid losing logs.
            ok, content = self._download_full()
            if not ok:
                logger.debug(LOG_REMOTE_NOT_FOUND_MSG)
                return False
            self._local_path.write_text(content, encoding=ENCODING_UTF8)
        else:
            # Append delta (keeps local log a full artifact for this run).
            with self._local_path.open("a", encoding=ENCODING_UTF8) as f:
                f.write(delta)

        current_size = self._get_local_size_bytes()

        if not silent:
            logger.info(f"📥 Downloaded training log: {self._local_path} ({current_size:,} bytes)")
        elif current_size != self._last_size:
            # Log only when size changes (new content)
            logger.debug(f"📥 Training log updated: {current_size:,} bytes")

        self._last_size = current_size
        return True

    def get_last_lines(self, n: int = 30) -> list[str]:
        """
        Get last N lines from remote training log.

        Fetches directly from remote to get latest content.

        Args:
            n: Number of lines to get

        Returns:
            List of last N lines (empty if log not found)
        """
        success, content, _ = self.ssh.exec_command(
            command=f"tail -n {n} {self._remote_path} 2>/dev/null || echo ''",
            silent=True,
        )

        if not success or not content.strip():
            return []

        return content.strip().split("\n")

    def download_on_error(self, error_context: str = "") -> None:
        """
        Download log on error - always logs the download.

        Args:
            error_context: Description of the error for logging
        """
        if error_context:
            logger.info(f"📋 Downloading training log due to: {error_context}")

        self.download(silent=False)


# Backward compatibility alias
RunPodLogManager = LogManager

__all__ = ["LogManager", "RunPodLogManager"]
