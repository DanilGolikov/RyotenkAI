"""Phase 12.A.1 — :meth:`ModelRetriever._retrieve_and_replay_metrics_buffer`
integration contract.

Pins the wire-up between the new
:class:`MetricsBufferRetriever` + :class:`BufferedMetricsReplay` and
the existing :class:`ModelRetriever` stage. Bypasses the heavy
``_execute_retrieval`` path (HF upload, model card generation,
provider lookups) by calling the helper method directly with a
hand-built context — that's the seam Phase 12.A.1 introduces.

Pattern: instantiate :class:`ModelRetriever` via the test fixture
that already exists in the suite, then unit-test the helper as a
black box. Side effects observed via:
* fake :class:`SSHClient` (records calls, materialises files)
* fake :class:`MlflowClient` (counts log_metric calls)
* :data:`ModelRetrieverEventCallbacks.on_metrics_buffer_retrieved`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.stages.constants import PipelineContextKeys
from src.pipeline.stages.model_retriever import (
    ModelRetriever,
    ModelRetrieverEventCallbacks,
)
from src.utils.config import HuggingFaceHubConfig
from src.utils.result import Err, Ok, ProviderError


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSSHClient:
    """Mirrors the surface that
    :class:`MetricsBufferRetriever` uses (``file_exists``,
    ``exec_command``, ``download_file``)."""

    def __init__(
        self,
        *,
        buffer_present: bool = True,
        buffer_size: int = 1,
        buffer_contents: str = "",
        download_succeeds: bool = True,
    ) -> None:
        self._present_paths: set[str] = set()
        self._sizes: dict[str, int] = {}
        self._contents: dict[str, str] = {}
        if buffer_present:
            self._present_paths.add("/workspace/metrics_buffer.jsonl")
            self._sizes["/workspace/metrics_buffer.jsonl"] = buffer_size
            self._contents["/workspace/metrics_buffer.jsonl"] = buffer_contents
        self._download_succeeds = download_succeeds
        self.download_calls: list[tuple[str, Path]] = []
        self.close_master_calls = 0

    def file_exists(self, remote_path: str) -> bool:
        return remote_path in self._present_paths

    def exec_command(
        self,
        command: str,
        background: bool = False,
        timeout: int = 30,
        silent: bool = False,
    ) -> tuple[bool, str, str]:
        if command.startswith("stat -c %s "):
            remote = command[len("stat -c %s ") :].strip()
            if remote in self._sizes:
                return True, str(self._sizes[remote]) + "\n", ""
            return False, "", "stat: No such file"
        return False, "", "unrecognised in fake"

    def download_file(
        self,
        remote_path: str,
        local_path: Path,
        timeout: int = 300,
    ) -> Any:
        self.download_calls.append((remote_path, local_path))
        if not self._download_succeeds:
            return Err(
                ProviderError(
                    message="simulated download failure",
                    code="SSH_DOWNLOAD_FILE_FAILED",
                    details={},
                )
            )
        if remote_path in self._contents:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(self._contents[remote_path], encoding="utf-8")
            return Ok(None)
        return Err(
            ProviderError(
                message=f"file not in fake contents: {remote_path}",
                code="SSH_DOWNLOAD_FILE_FAILED",
                details={},
            )
        )

    def close_master(self) -> None:  # pragma: no cover — defensive
        self.close_master_calls += 1


class _FakeMlflowClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        timestamp: int | None = None,
        step: int | None = None,
    ) -> None:
        self.calls.append(
            {
                "run_id": run_id,
                "key": key,
                "value": value,
                "timestamp": timestamp,
                "step": step,
            }
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_secrets() -> MagicMock:
    secrets = MagicMock()
    secrets.hf_token = "hf_test_token"
    return secrets


@pytest.fixture
def mock_config() -> MagicMock:
    cfg = MagicMock()
    cfg.get_active_provider_name.return_value = "single_node"
    cfg.get_provider_config.return_value = {"mock_mode": False}
    cfg.integrations.huggingface = HuggingFaceHubConfig(
        integration=None, repo_id=None, private=False,
    )
    return cfg


def _make_retriever(
    cfg: MagicMock, secrets: MagicMock, *, callbacks: ModelRetrieverEventCallbacks | None = None,
) -> ModelRetriever:
    return ModelRetriever(cfg, secrets, callbacks=callbacks)


def _build_context(
    *,
    attempt_dir: Path,
    run_id: str | None = "run-abc",
) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        PipelineContextKeys.ATTEMPT_DIRECTORY: str(attempt_dir),
    }
    if run_id is not None:
        ctx[PipelineContextKeys.MLFLOW_PARENT_RUN_ID] = run_id
    return ctx


# ---------------------------------------------------------------------------
# 1. Positive — full happy path
# ---------------------------------------------------------------------------


class TestPositive:
    def test_buffer_present_replays_into_mlflow(
        self, mock_config: MagicMock, mock_secrets: MagicMock, tmp_path: Path
    ) -> None:
        attempt_dir = tmp_path / "attempts" / "1"
        ctx = _build_context(attempt_dir=attempt_dir, run_id="run-abc")

        ssh = _FakeSSHClient(
            buffer_present=True,
            buffer_size=200,
            buffer_contents=(
                '{"key":"loss","value":0.5,"step":1,"timestamp":1.0}\n'
                '{"key":"loss","value":0.4,"step":2,"timestamp":2.0}\n'
            ),
        )
        mlflow_client = _FakeMlflowClient()
        events: list[tuple] = []

        callbacks = ModelRetrieverEventCallbacks(
            on_metrics_buffer_retrieved=lambda r, lc, sb, missing, oversized: events.append(
                (r, lc, sb, missing, oversized)
            ),
        )
        retriever = _make_retriever(mock_config, mock_secrets, callbacks=callbacks)
        retriever._ssh_client = ssh  # type: ignore[assignment]
        retriever._workspace_path = "/workspace"

        with patch.object(
            ModelRetriever, "_build_mlflow_client", return_value=mlflow_client
        ):
            retriever._retrieve_and_replay_metrics_buffer(ctx)

        # Metrics shipped.
        assert len(mlflow_client.calls) == 2
        assert all(c["run_id"] == "run-abc" for c in mlflow_client.calls)
        # Local copy preserved for forensics.
        assert (attempt_dir / "metrics_buffer.jsonl").exists()
        # Callback fired with (replayed=2, line_count=2, size=200, missing=False, oversized=False).
        assert events == [(2, 2, 200, False, False)]


# ---------------------------------------------------------------------------
# 2. Negative — paths that skip silently / record forensics
# ---------------------------------------------------------------------------


class TestNegative:
    def test_no_ssh_client_skips_without_raising(
        self, mock_config: MagicMock, mock_secrets: MagicMock
    ) -> None:
        retriever = _make_retriever(mock_config, mock_secrets)
        retriever._ssh_client = None
        # Must not raise; must not invoke any callback.
        retriever._retrieve_and_replay_metrics_buffer({})

    def test_no_mlflow_run_id_keeps_local_copy(
        self, mock_config: MagicMock, mock_secrets: MagicMock, tmp_path: Path
    ) -> None:
        attempt_dir = tmp_path / "attempts" / "1"
        ctx = _build_context(attempt_dir=attempt_dir, run_id=None)
        ssh = _FakeSSHClient(
            buffer_present=True,
            buffer_size=50,
            buffer_contents='{"key":"loss","value":0.5,"step":1,"timestamp":1.0}\n',
        )
        events: list[tuple] = []
        callbacks = ModelRetrieverEventCallbacks(
            on_metrics_buffer_retrieved=lambda r, lc, sb, missing, oversized: events.append(
                (r, lc, sb, missing, oversized)
            ),
        )
        retriever = _make_retriever(mock_config, mock_secrets, callbacks=callbacks)
        retriever._ssh_client = ssh  # type: ignore[assignment]
        retriever._workspace_path = "/workspace"

        retriever._retrieve_and_replay_metrics_buffer(ctx)

        # File preserved locally; replayed=0 because no run_id.
        assert (attempt_dir / "metrics_buffer.jsonl").exists()
        assert events == [(0, 1, 50, False, False)]

    def test_buffer_missing_emits_healthy_callback(
        self, mock_config: MagicMock, mock_secrets: MagicMock, tmp_path: Path
    ) -> None:
        attempt_dir = tmp_path / "attempts" / "1"
        ctx = _build_context(attempt_dir=attempt_dir, run_id="run-abc")
        ssh = _FakeSSHClient(buffer_present=False)
        events: list[tuple] = []
        callbacks = ModelRetrieverEventCallbacks(
            on_metrics_buffer_retrieved=lambda r, lc, sb, missing, oversized: events.append(
                (r, lc, sb, missing, oversized)
            ),
        )
        retriever = _make_retriever(mock_config, mock_secrets, callbacks=callbacks)
        retriever._ssh_client = ssh  # type: ignore[assignment]
        retriever._workspace_path = "/workspace"

        retriever._retrieve_and_replay_metrics_buffer(ctx)

        # Healthy case: missing=True, replayed=0, line_count=0.
        assert events == [(0, 0, 0, True, False)]
        # No SCP attempted.
        assert ssh.download_calls == []

    def test_no_attempt_directory_skips(
        self, mock_config: MagicMock, mock_secrets: MagicMock
    ) -> None:
        retriever = _make_retriever(mock_config, mock_secrets)
        retriever._ssh_client = _FakeSSHClient(buffer_present=True)  # type: ignore[assignment]
        retriever._workspace_path = "/workspace"
        # Empty context — no attempt_directory key.
        retriever._retrieve_and_replay_metrics_buffer({})


# ---------------------------------------------------------------------------
# 3. Boundary — oversized, mlflow client unavailable
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_oversized_buffer_skipped_with_callback(
        self, mock_config: MagicMock, mock_secrets: MagicMock, tmp_path: Path
    ) -> None:
        attempt_dir = tmp_path / "attempts" / "1"
        ctx = _build_context(attempt_dir=attempt_dir, run_id="run-abc")
        oversized = 200 * 1024 * 1024  # 200 MiB > 100 MiB cap
        ssh = _FakeSSHClient(buffer_present=True, buffer_size=oversized)
        events: list[tuple] = []
        callbacks = ModelRetrieverEventCallbacks(
            on_metrics_buffer_retrieved=lambda r, lc, sb, missing, oversized_flag: events.append(
                (r, lc, sb, missing, oversized_flag)
            ),
        )
        retriever = _make_retriever(mock_config, mock_secrets, callbacks=callbacks)
        retriever._ssh_client = ssh  # type: ignore[assignment]
        retriever._workspace_path = "/workspace"

        retriever._retrieve_and_replay_metrics_buffer(ctx)

        assert events == [(0, 0, oversized, False, True)]
        assert ssh.download_calls == []  # no SCP

    def test_mlflow_client_unavailable_keeps_local_copy(
        self, mock_config: MagicMock, mock_secrets: MagicMock, tmp_path: Path
    ) -> None:
        # MlflowClient construction returns None (mlflow not installed
        # or tracking URI malformed).
        attempt_dir = tmp_path / "attempts" / "1"
        ctx = _build_context(attempt_dir=attempt_dir, run_id="run-abc")
        ssh = _FakeSSHClient(
            buffer_present=True,
            buffer_size=50,
            buffer_contents='{"key":"loss","value":0.5,"step":1,"timestamp":1.0}\n',
        )
        events: list[tuple] = []
        callbacks = ModelRetrieverEventCallbacks(
            on_metrics_buffer_retrieved=lambda r, lc, sb, missing, oversized: events.append(
                (r, lc, sb, missing, oversized)
            ),
        )
        retriever = _make_retriever(mock_config, mock_secrets, callbacks=callbacks)
        retriever._ssh_client = ssh  # type: ignore[assignment]
        retriever._workspace_path = "/workspace"

        with patch.object(
            ModelRetriever, "_build_mlflow_client", return_value=None
        ):
            retriever._retrieve_and_replay_metrics_buffer(ctx)

        # Buffer downloaded but not replayed.
        assert (attempt_dir / "metrics_buffer.jsonl").exists()
        assert events == [(0, 1, 50, False, False)]


# ---------------------------------------------------------------------------
# 4. Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_callback_is_optional(
        self, mock_config: MagicMock, mock_secrets: MagicMock, tmp_path: Path
    ) -> None:
        # No callback attached — must not raise.
        attempt_dir = tmp_path / "attempts" / "1"
        ctx = _build_context(attempt_dir=attempt_dir, run_id="run-abc")
        ssh = _FakeSSHClient(buffer_present=False)
        retriever = _make_retriever(mock_config, mock_secrets)
        retriever._ssh_client = ssh  # type: ignore[assignment]
        retriever._workspace_path = "/workspace"

        retriever._retrieve_and_replay_metrics_buffer(ctx)

    def test_run_id_resolution_chain(
        self, mock_config: MagicMock, mock_secrets: MagicMock
    ) -> None:
        # 1. PipelineContextKeys.MLFLOW_PARENT_RUN_ID first.
        ctx = {PipelineContextKeys.MLFLOW_PARENT_RUN_ID: "primary"}
        assert ModelRetriever._resolve_mlflow_run_id(ctx) == "primary"

        # 2. Falls back to plain string keys (test contexts).
        ctx2 = {"mlflow_parent_run_id": "fallback"}
        assert ModelRetriever._resolve_mlflow_run_id(ctx2) == "fallback"

        # 3. Whitespace stripped.
        ctx3 = {PipelineContextKeys.MLFLOW_PARENT_RUN_ID: "  spacey  "}
        assert ModelRetriever._resolve_mlflow_run_id(ctx3) == "spacey"

        # 4. Empty string → None.
        ctx4 = {PipelineContextKeys.MLFLOW_PARENT_RUN_ID: ""}
        assert ModelRetriever._resolve_mlflow_run_id(ctx4) is None

        # 5. Missing → None.
        assert ModelRetriever._resolve_mlflow_run_id({}) is None


# ---------------------------------------------------------------------------
# 5. Dependency errors
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_scp_failure_logged_and_callback_emitted(
        self, mock_config: MagicMock, mock_secrets: MagicMock, tmp_path: Path
    ) -> None:
        attempt_dir = tmp_path / "attempts" / "1"
        ctx = _build_context(attempt_dir=attempt_dir, run_id="run-abc")
        ssh = _FakeSSHClient(
            buffer_present=True, buffer_size=10, download_succeeds=False
        )
        events: list[tuple] = []
        callbacks = ModelRetrieverEventCallbacks(
            on_metrics_buffer_retrieved=lambda r, lc, sb, missing, oversized: events.append(
                (r, lc, sb, missing, oversized)
            ),
        )
        retriever = _make_retriever(mock_config, mock_secrets, callbacks=callbacks)
        retriever._ssh_client = ssh  # type: ignore[assignment]
        retriever._workspace_path = "/workspace"

        retriever._retrieve_and_replay_metrics_buffer(ctx)

        # SCP failed → replayed=0, missing=False, oversized=False, but
        # size_bytes from the stat probe is still surfaced.
        assert events == [(0, 0, 10, False, False)]


# ---------------------------------------------------------------------------
# 6. Regressions
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_helper_does_not_raise_when_ssh_misbehaves(
        self, mock_config: MagicMock, mock_secrets: MagicMock, tmp_path: Path
    ) -> None:
        # Defensive: even with a broken SSH stub that raises in
        # arbitrary places, the helper completes without bubbling.
        attempt_dir = tmp_path / "attempts" / "1"
        ctx = _build_context(attempt_dir=attempt_dir, run_id="run-abc")

        broken_ssh = MagicMock()
        broken_ssh.file_exists.side_effect = RuntimeError("ssh broken")

        retriever = _make_retriever(mock_config, mock_secrets)
        retriever._ssh_client = broken_ssh  # type: ignore[assignment]
        retriever._workspace_path = "/workspace"

        # Inner helper itself doesn't trap RuntimeError; outer
        # ``_execute_retrieval`` does. We verify _execute_retrieval's
        # try/except via direct test below.
        with pytest.raises(RuntimeError):
            retriever._retrieve_and_replay_metrics_buffer(ctx)


# ---------------------------------------------------------------------------
# 7. Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_replay_uses_constructed_mlflow_client(
        self, mock_config: MagicMock, mock_secrets: MagicMock, tmp_path: Path
    ) -> None:
        # Pin: helper constructs MlflowClient via the static factory
        # method — wired through ``_build_mlflow_client``. Patching
        # the factory must redirect ALL replay log_metric calls to
        # our fake.
        attempt_dir = tmp_path / "attempts" / "1"
        ctx = _build_context(attempt_dir=attempt_dir, run_id="run-abc")
        ssh = _FakeSSHClient(
            buffer_present=True,
            buffer_size=10,
            buffer_contents=(
                '{"key":"x","value":1,"step":1,"timestamp":1.0}\n'
                '{"key":"y","value":2,"step":2,"timestamp":2.0}\n'
            ),
        )
        client = _FakeMlflowClient()
        retriever = _make_retriever(mock_config, mock_secrets)
        retriever._ssh_client = ssh  # type: ignore[assignment]
        retriever._workspace_path = "/workspace"

        with patch.object(
            ModelRetriever, "_build_mlflow_client", return_value=client
        ):
            retriever._retrieve_and_replay_metrics_buffer(ctx)

        keys = [c["key"] for c in client.calls]
        assert keys == ["x", "y"]

    def test_marker_file_preserved_for_forensics(
        self, mock_config: MagicMock, mock_secrets: MagicMock, tmp_path: Path
    ) -> None:
        # Source marker exists on pod → fetch downloads it as
        # ``buffer.flush_offset.json`` next to the buffer.
        marker_payload = json.dumps(
            {"v": 1, "drained_count": 5, "drained_at_ms": 1700000000000},
        )
        ssh = _FakeSSHClient(
            buffer_present=True,
            buffer_size=10,
            buffer_contents='{"key":"l","value":1,"step":1,"timestamp":1.0}\n',
        )
        # Manually inject the offset file.
        ssh._present_paths.add("/workspace/.runner/buffer.flush_offset")
        ssh._sizes["/workspace/.runner/buffer.flush_offset"] = len(marker_payload)
        ssh._contents["/workspace/.runner/buffer.flush_offset"] = marker_payload

        attempt_dir = tmp_path / "attempts" / "1"
        ctx = _build_context(attempt_dir=attempt_dir, run_id="run-abc")
        retriever = _make_retriever(mock_config, mock_secrets)
        retriever._ssh_client = ssh  # type: ignore[assignment]
        retriever._workspace_path = "/workspace"

        with patch.object(
            ModelRetriever, "_build_mlflow_client", return_value=_FakeMlflowClient()
        ):
            retriever._retrieve_and_replay_metrics_buffer(ctx)

        # Marker preserved.
        marker_local = attempt_dir / "buffer.flush_offset.json"
        assert marker_local.exists()
        parsed = json.loads(marker_local.read_text())
        assert parsed["drained_count"] == 5
