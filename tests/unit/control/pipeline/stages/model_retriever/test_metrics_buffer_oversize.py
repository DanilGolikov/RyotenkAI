"""Post-Phase-10 B-СРЕД fix — env-tunable buffer threshold + typed event.

Covers the three branches introduced when the silent-drop policy on
``MAX_BUFFER_SIZE_BYTES`` was replaced with an honest-failure mode:

* below threshold              → no event, downloads as before
* exactly at threshold         → no event, downloads (boundary preserved)
* above threshold, below cap   → emits :class:`MetricsBufferOversizedEvent`,
                                 returns ``oversized=True``
* above ultra-large hard cap   → raises :class:`MetricsBufferTooLargeError`
* env override honoured        → ``RYOTENKAI_METRICS_BUFFER_MAX_MB=500``
                                 bumps the per-instance threshold

The tests use a tiny in-memory ``SSHClient`` fake (same shape as
``test_metrics_buffer_retriever.py``) plus the canonical
``FakeEventEmitter`` so the typed-event surface is exercised end-to-end
without touching the orchestrator or the bus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ryotenkai_control.pipeline.stages.model_retriever.metrics_buffer_retriever import (
    DEFAULT_BUFFER_MAX_MIB,
    ENV_METRICS_BUFFER_MAX_MB,
    MetricsBufferRetriever,
    _MAX_ENV_THRESHOLD_MIB,
)
from ryotenkai_shared.errors import (
    MetricsBufferTooLargeError,
    SSHTransferFailedError,
)
from ryotenkai_shared.events.types.control_model import (
    MetricsBufferOversizedEvent,
)

from tests._fakes.event_emitter import FakeEventEmitter

_REMOTE_BUFFER = "/workspace/metrics_buffer.jsonl"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSSHClient:
    """In-memory stand-in mirroring ``test_metrics_buffer_retriever``."""

    def __init__(self) -> None:
        self.files_present: set[str] = set()
        self.sizes: dict[str, int] = {}
        self.contents: dict[str, str] = {}
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
        if command.startswith("stat -c %s "):
            remote = command[len("stat -c %s ") :].strip()
            if remote in self.sizes:
                return True, str(self.sizes[remote]) + "\n", ""
        return False, "", "n/a"

    def download_file(
        self,
        remote_path: str,
        local_path: Path,
        timeout: int = 300,
    ) -> None:
        self.download_calls.append((remote_path, local_path))
        if remote_path in self.contents:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(self.contents[remote_path], encoding="utf-8")
            return
        raise SSHTransferFailedError(
            detail=f"not in fake: {remote_path}",
            context={"op": "download_file"},
        )


def _make(
    ssh: _FakeSSHClient,
    *,
    emitter: FakeEventEmitter | None = None,
    env: dict[str, str] | None = None,
) -> MetricsBufferRetriever:
    return MetricsBufferRetriever(  # type: ignore[arg-type]
        ssh,
        workspace_path="/workspace",
        emitter=emitter,
        run_id="run-test",
        env=env,
    )


# ---------------------------------------------------------------------------
# 1. Positive — under threshold: no event, normal retrieval.
# ---------------------------------------------------------------------------


class TestPositive:
    def test_below_threshold_no_event(self, tmp_path: Path) -> None:
        emitter = FakeEventEmitter()
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER}
        ssh.sizes = {_REMOTE_BUFFER: 1024}
        ssh.contents = {_REMOTE_BUFFER: '{"k":"v","value":1,"step":1,"timestamp":1}\n'}

        retriever = _make(ssh, emitter=emitter)
        result = retriever.fetch(local_dir=tmp_path)

        assert result.local_path is not None
        assert not result.oversized
        # No oversized event when below threshold.
        assert not any(
            isinstance(e, MetricsBufferOversizedEvent)
            for e in emitter.emitted
        )


# ---------------------------------------------------------------------------
# 2. Negative — above threshold, below ultra-large cap: emit + skip.
# ---------------------------------------------------------------------------


class TestNegative:
    def test_above_threshold_emits_event_and_skips(
        self, tmp_path: Path
    ) -> None:
        emitter = FakeEventEmitter()
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER}
        # 200 MiB > 100 MiB default threshold; way under 1 GiB cap.
        threshold = DEFAULT_BUFFER_MAX_MIB * 1024 * 1024
        ssh.sizes = {_REMOTE_BUFFER: threshold + 1024}

        retriever = _make(ssh, emitter=emitter)
        result = retriever.fetch(local_dir=tmp_path)

        assert result.oversized is True
        assert result.local_path is None
        assert ssh.download_calls == []
        events = [
            e for e in emitter.emitted
            if isinstance(e, MetricsBufferOversizedEvent)
        ]
        assert len(events) == 1
        assert events[0].payload.discarded is True
        assert events[0].payload.size_bytes == threshold + 1024
        assert events[0].payload.threshold_bytes == threshold
        assert events[0].payload.pod_path == _REMOTE_BUFFER
        assert events[0].severity == "warning"


# ---------------------------------------------------------------------------
# 3. Boundary — exactly at threshold proceeds without event.
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_size_exactly_at_threshold_proceeds(self, tmp_path: Path) -> None:
        emitter = FakeEventEmitter()
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER}
        threshold = DEFAULT_BUFFER_MAX_MIB * 1024 * 1024
        ssh.sizes = {_REMOTE_BUFFER: threshold}
        ssh.contents = {_REMOTE_BUFFER: ""}

        retriever = _make(ssh, emitter=emitter)
        result = retriever.fetch(local_dir=tmp_path)

        assert not result.oversized
        assert result.size_bytes == threshold
        assert not any(
            isinstance(e, MetricsBufferOversizedEvent) for e in emitter.emitted
        )


# ---------------------------------------------------------------------------
# 4. Regressions — ultra-large (> hard cap) raises typed error.
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_ultra_large_raises_typed_error(self, tmp_path: Path) -> None:
        emitter = FakeEventEmitter()
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER}
        hard_cap = _MAX_ENV_THRESHOLD_MIB * 1024 * 1024
        ssh.sizes = {_REMOTE_BUFFER: hard_cap + 1}

        retriever = _make(ssh, emitter=emitter)

        with pytest.raises(MetricsBufferTooLargeError) as excinfo:
            retriever.fetch(local_dir=tmp_path)

        err = excinfo.value
        assert err.context["size_bytes"] == hard_cap + 1
        assert err.context["legacy_code"] == "METRICS_BUFFER_OVERSIZE"
        # Honest-failure: also emits the oversized event with
        # ``discarded=True`` before raising.
        events = [
            e for e in emitter.emitted
            if isinstance(e, MetricsBufferOversizedEvent)
        ]
        assert len(events) == 1
        assert events[0].payload.discarded is True

    def test_no_emitter_does_not_crash(self, tmp_path: Path) -> None:
        # Phase-5 optional-emitter contract: passing ``emitter=None``
        # must not change the retrieval semantics for the oversized
        # path — the typed event simply doesn't fire.
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER}
        threshold = DEFAULT_BUFFER_MAX_MIB * 1024 * 1024
        ssh.sizes = {_REMOTE_BUFFER: threshold + 1024}

        retriever = _make(ssh, emitter=None)
        result = retriever.fetch(local_dir=tmp_path)

        assert result.oversized is True
        assert result.local_path is None


# ---------------------------------------------------------------------------
# 5. Logic-specific — env var honoured.
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_env_var_bumps_threshold(self, tmp_path: Path) -> None:
        emitter = FakeEventEmitter()
        ssh = _FakeSSHClient()
        ssh.files_present = {_REMOTE_BUFFER}
        # File would normally trip the 100 MiB default cap...
        size = 200 * 1024 * 1024
        ssh.sizes = {_REMOTE_BUFFER: size}
        ssh.contents = {_REMOTE_BUFFER: ""}

        # ... but the operator opted into a 500 MiB threshold.
        retriever = _make(
            ssh,
            emitter=emitter,
            env={ENV_METRICS_BUFFER_MAX_MB: "500"},
        )
        assert retriever.threshold_bytes == 500 * 1024 * 1024

        result = retriever.fetch(local_dir=tmp_path)
        assert not result.oversized
        assert result.size_bytes == size
        # No oversized event because we stayed under the bumped threshold.
        assert not any(
            isinstance(e, MetricsBufferOversizedEvent) for e in emitter.emitted
        )

    def test_env_var_capped_at_hard_max(self, tmp_path: Path) -> None:
        # Asking for 5 TiB clamps to the safe upper bound.
        ssh = _FakeSSHClient()
        retriever = _make(
            ssh,
            env={ENV_METRICS_BUFFER_MAX_MB: str(_MAX_ENV_THRESHOLD_MIB * 100)},
        )
        assert (
            retriever.threshold_bytes == _MAX_ENV_THRESHOLD_MIB * 1024 * 1024
        )

    @pytest.mark.parametrize("bad", ["", "not-a-number", "-1", "0"])
    def test_env_var_falls_back_to_default(
        self, tmp_path: Path, bad: str
    ) -> None:
        ssh = _FakeSSHClient()
        retriever = _make(ssh, env={ENV_METRICS_BUFFER_MAX_MB: bad})
        assert (
            retriever.threshold_bytes == DEFAULT_BUFFER_MAX_MIB * 1024 * 1024
        )
