"""Phase 12.A.1 — :class:`MetricsBufferRetriever` contract.

7-category coverage for the SSH-side fetcher that downloads
``/workspace/metrics_buffer.jsonl`` from the pod into a local
attempt directory before MLflow replay.

Uses an in-memory fake :class:`SSHClient` rather than touching real
SSH — the retriever's full logic (probe, size cap, SCP, line
count) is stdlib + a thin wrapper around ``exec_command`` /
``file_exists`` / ``download_file``, so the fake is sufficient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ryotenkai_control.pipeline.stages.model_retriever.metrics_buffer_retriever import (
    FetchResult,
    MetricsBufferRetriever,
)
from ryotenkai_shared.errors import SSHTransferFailedError


# ---------------------------------------------------------------------------
# Fake SSHClient
# ---------------------------------------------------------------------------


class _FakeSSHClient:
    """In-memory stand-in for :class:`SSHClient`.

    Configured per test:
    * ``files_present`` — set of remote paths that ``test -f`` reports
                          as present.
    * ``sizes``         — map remote_path → bytes returned by stat probe.
    * ``contents``      — map remote_path → string contents written into
                          the local destination on ``download_file``.
    * ``download_error``— optional :class:`SSHTransferFailedError` to raise.
    """

    def __init__(self) -> None:
        self.files_present: set[str] = set()
        self.sizes: dict[str, int] = {}
        self.contents: dict[str, str] = {}
        self.download_error: SSHTransferFailedError | None = None
        self.exec_calls: list[tuple[str, dict[str, Any]]] = []
        self.download_calls: list[tuple[str, Path]] = []

    def file_exists(self, remote_path: str) -> bool:
        return remote_path in self.files_present

    def exec_command(
        self,
        command: str,
        background: bool = False,
        timeout: int = 30,
        silent: bool = False,
    ) -> tuple[bool, str, str]:
        self.exec_calls.append(
            (command, {"timeout": timeout, "silent": silent})
        )
        # Recognise our `stat -c %s <path>` pattern.
        if command.startswith("stat -c %s "):
            remote = command[len("stat -c %s ") :].strip()
            if remote in self.sizes:
                return True, str(self.sizes[remote]) + "\n", ""
            return False, "", "stat: No such file"
        return False, "", "unrecognised in fake"

    def download_file(
        self,
        remote_path: str,
        local_path: Path,
        timeout: int = 300,
    ) -> None:
        self.download_calls.append((remote_path, local_path))
        if self.download_error is not None:
            raise self.download_error
        if remote_path in self.contents:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(self.contents[remote_path], encoding="utf-8")
            return
        raise SSHTransferFailedError(
            detail=f"file not in fake contents: {remote_path}",
            context={"op": "download_file"},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_REMOTE_BUFFER = "/workspace/metrics_buffer.jsonl"
_REMOTE_OFFSET = "/workspace/.runner/buffer.flush_offset"


def _make(ssh: _FakeSSHClient) -> MetricsBufferRetriever:
    return MetricsBufferRetriever(ssh, workspace_path="/workspace")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_fetch_existing_file_returns_local_path(
        self, tmp_path: Path
    ) -> None:
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER}
        ssh.sizes = {_REMOTE_BUFFER: 42}
        ssh.contents = {
            _REMOTE_BUFFER: (
                '{"key":"loss","value":0.5,"step":1,"timestamp":1.0}\n'
                '{"key":"loss","value":0.4,"step":2,"timestamp":2.0}\n'
            ),
        }

        retriever = _make(ssh)
        result = retriever.fetch(local_dir=tmp_path)

        assert result.local_path == tmp_path / "metrics_buffer.jsonl"
        assert result.size_bytes == 42
        assert result.line_count == 2
        assert not result.missing
        assert not result.oversized
        assert result.error is None

    def test_fetch_optionally_pulls_flush_offset(
        self, tmp_path: Path
    ) -> None:
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER, _REMOTE_OFFSET}
        ssh.sizes = {_REMOTE_BUFFER: 1}
        ssh.contents = {
            _REMOTE_BUFFER: '{"key":"loss","value":0.5,"step":1,"timestamp":1.0}\n',
            _REMOTE_OFFSET: '{"v":1,"drained_count":12,"drained_at_ms":1700000000000}',
        }

        retriever = _make(ssh)
        retriever.fetch(local_dir=tmp_path)

        # Two download_file calls: buffer + offset marker.
        called_remotes = [c[0] for c in ssh.download_calls]
        assert _REMOTE_BUFFER in called_remotes
        assert _REMOTE_OFFSET in called_remotes
        # Offset marker preserved as forensic file.
        assert (tmp_path / "buffer.flush_offset.json").exists()


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_missing_remote_file_returns_missing_true(
        self, tmp_path: Path
    ) -> None:
        ssh = _FakeSSHClient()
        # No files present at all.
        retriever = _make(ssh)
        result = retriever.fetch(local_dir=tmp_path)

        assert result.missing is True
        assert result.local_path is None
        assert result.error is None  # missing != error
        assert ssh.download_calls == []  # no SCP attempt

    def test_stat_probe_failure_returns_missing_with_error(
        self, tmp_path: Path
    ) -> None:
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER}
        # No size in `sizes` → stat returns failure.
        retriever = _make(ssh)
        result = retriever.fetch(local_dir=tmp_path)

        assert result.missing is True
        assert result.error is not None
        assert "stat" in result.error.lower() or "ssh" in result.error.lower()
        assert ssh.download_calls == []

    def test_scp_download_failure_returns_error(
        self, tmp_path: Path
    ) -> None:
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER}
        ssh.sizes = {_REMOTE_BUFFER: 100}
        ssh.download_error = SSHTransferFailedError(
            detail="SCP boom",
            context={"op": "download_file"},
        )

        retriever = _make(ssh)
        result = retriever.fetch(local_dir=tmp_path)

        assert result.local_path is None
        assert not result.missing
        assert not result.oversized
        assert result.error is not None
        assert "SCP" in result.error or "boom" in result.error


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_size_at_cap_proceeds(self, tmp_path: Path) -> None:
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER}
        ssh.sizes = {_REMOTE_BUFFER: MetricsBufferRetriever.MAX_BUFFER_SIZE_BYTES}
        ssh.contents = {_REMOTE_BUFFER: ""}

        retriever = _make(ssh)
        result = retriever.fetch(local_dir=tmp_path)

        # Exactly at cap → still allowed.
        assert not result.oversized
        assert result.size_bytes == MetricsBufferRetriever.MAX_BUFFER_SIZE_BYTES

    def test_size_above_cap_skips_with_oversized_flag(
        self, tmp_path: Path
    ) -> None:
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER}
        ssh.sizes = {
            _REMOTE_BUFFER: MetricsBufferRetriever.MAX_BUFFER_SIZE_BYTES + 1
        }

        retriever = _make(ssh)
        result = retriever.fetch(local_dir=tmp_path)

        assert result.oversized is True
        assert result.local_path is None
        assert ssh.download_calls == []  # never even attempted SCP

    def test_zero_byte_file_is_fetched(self, tmp_path: Path) -> None:
        # Empty buffer file is a valid edge case (decimation discarded
        # all records). Replay will return replayed=0 cleanly.
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER}
        ssh.sizes = {_REMOTE_BUFFER: 0}
        ssh.contents = {_REMOTE_BUFFER: ""}

        retriever = _make(ssh)
        result = retriever.fetch(local_dir=tmp_path)

        assert result.local_path is not None
        assert result.line_count == 0
        assert result.size_bytes == 0


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_local_dir_created_if_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "attempts"
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER}
        ssh.sizes = {_REMOTE_BUFFER: 1}
        ssh.contents = {_REMOTE_BUFFER: '{"key":"l","value":1,"step":1,"timestamp":1}\n'}

        retriever = _make(ssh)
        retriever.fetch(local_dir=target)

        assert target.exists()
        assert (target / "metrics_buffer.jsonl").exists()

    def test_remote_buffer_path_uses_workspace(
        self, tmp_path: Path
    ) -> None:
        ssh = _FakeSSHClient()
        retriever = MetricsBufferRetriever(  # type: ignore[arg-type]
            ssh, workspace_path="/data/workspace"
        )
        assert retriever.remote_buffer_path == (
            "/data/workspace/metrics_buffer.jsonl"
        )

    def test_workspace_trailing_slash_normalised(
        self, tmp_path: Path
    ) -> None:
        ssh = _FakeSSHClient()
        retriever = MetricsBufferRetriever(  # type: ignore[arg-type]
            ssh, workspace_path="/workspace/"
        )
        assert retriever.remote_buffer_path == "/workspace/metrics_buffer.jsonl"


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_local_mkdir_failure_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "blocked"
        ssh = _FakeSSHClient()
        retriever = _make(ssh)

        # Force mkdir to fail.
        original_mkdir = Path.mkdir

        def _raise_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
            if self == target:
                raise PermissionError("simulated EACCES")
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", _raise_mkdir)

        result = retriever.fetch(local_dir=target)

        assert result.local_path is None
        assert result.error is not None
        assert "mkdir" in result.error


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_offset_marker_optional_no_failure_when_absent(
        self, tmp_path: Path
    ) -> None:
        # Healthy retrieval — flush_offset marker absent (trainer
        # never had a successful flush, which is the typical "Mac
        # asleep entire training" case). MUST NOT fail.
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER}  # no offset marker
        ssh.sizes = {_REMOTE_BUFFER: 1}
        ssh.contents = {_REMOTE_BUFFER: '{"k":"l","value":1,"step":1,"timestamp":1}\n'}

        retriever = _make(ssh)
        result = retriever.fetch(local_dir=tmp_path)

        assert result.local_path is not None
        assert result.error is None
        # Marker file NOT created locally — that's expected.
        assert not (tmp_path / "buffer.flush_offset.json").exists()


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_line_count_skips_empty_lines(self, tmp_path: Path) -> None:
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER}
        ssh.sizes = {_REMOTE_BUFFER: 1}
        ssh.contents = {
            _REMOTE_BUFFER: (
                '{"key":"loss","value":0.5,"step":1,"timestamp":1.0}\n'
                "\n"
                '{"key":"loss","value":0.4,"step":2,"timestamp":2.0}\n'
                "\n"
            ),
        }
        retriever = _make(ssh)

        result = retriever.fetch(local_dir=tmp_path)

        assert result.line_count == 2

    def test_silent_flag_used_for_probe_calls(
        self, tmp_path: Path
    ) -> None:
        # Probe commands are noisy if logged on every retrieval.
        # Verify they pass `silent=True` to keep pipeline.log clean.
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER}
        ssh.sizes = {_REMOTE_BUFFER: 1}
        ssh.contents = {_REMOTE_BUFFER: ""}

        retriever = _make(ssh)
        retriever.fetch(local_dir=tmp_path)

        # All exec_command (stat) calls in the fake should have
        # silent=True.
        for cmd, kwargs in ssh.exec_calls:
            assert kwargs["silent"] is True
