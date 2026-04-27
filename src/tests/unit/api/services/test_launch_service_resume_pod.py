"""Phase 11.C-2 — :func:`launch_service.resume_pod_for_run` tests.

The service is a thin facade over :class:`PodAvailabilityProbe` +
:func:`resume_pod_with_retry`. We pin the response shape (the new
``ResumePodResponse`` is what the REST router returns) and the
branching logic against the 5 :class:`PodAvailability` outcomes.

Tests stub the RunPod transport at module level via monkeypatching
``RUNPOD_API_KEY`` env + injecting a fake ``RunPodAPIClient`` so no
network is touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.api.services.launch_service import (
    ResumePodResponse,
    resume_pod_for_run,
)
from src.pipeline.state.attempt_controller import AttemptController


# RunPodAPIClient transitively imports the ``runpod`` SDK package
# which isn't installed in the slim dev venv. Tests that patch the
# client module directly need it on sys.path; tests that exercise
# only the legacy / missing-creds paths don't.
_RUNPOD_SDK_AVAILABLE = True
try:
    import importlib
    importlib.import_module("src.providers.runpod.training.api_client")
except Exception:  # noqa: BLE001
    _RUNPOD_SDK_AVAILABLE = False

requires_runpod_sdk = pytest.mark.skipif(
    not _RUNPOD_SDK_AVAILABLE,
    reason="runpod SDK not installed (slim dev venv)",
)
from src.pipeline.state.models import (
    PipelineAttemptState,
    PipelineState,
    StageRunState,
    utc_now_iso,
)
from src.pipeline.state.store import PipelineStateStore


def _seed_run(tmp_path: Path, *, with_pod: bool, provider: str = "runpod") -> Path:
    """Persist a state file at tmp_path with optional pod_metadata."""
    store = PipelineStateStore(tmp_path)
    state = PipelineState(
        schema_version=1,
        logical_run_id="run-test",
        run_directory=str(tmp_path),
        config_path="/tmp/test.yaml",
        active_attempt_id="att-1",
        pipeline_status=StageRunState.STATUS_RUNNING,
        training_critical_config_hash="",
        late_stage_config_hash="",
    )
    state.attempts.append(
        PipelineAttemptState(
            attempt_id="att-1",
            attempt_no=1,
            runtime_name="test",
            requested_action="fresh",
            effective_action="fresh",
            restart_from_stage=None,
            status=StageRunState.STATUS_RUNNING,
            started_at=utc_now_iso(),
        ),
    )
    if with_pod:
        controller = AttemptController(save_fn=store.save, run_ctx=None)
        controller.adopt_state(state)
        controller.register_attempt(state.attempts[-1])
        controller.set_pod_metadata(
            pod_id="pod-abc",
            provider=provider,
            last_known_status="stopped",
        )
    else:
        store.save(state)
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Positive paths
# ---------------------------------------------------------------------------


@requires_runpod_sdk
class TestRunningPath:
    def test_pod_running_returns_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_dir = _seed_run(tmp_path, with_pod=True)
        monkeypatch.setenv("RUNPOD_API_KEY", "test-key")

        fake_client = MagicMock()
        fake_client.query_pod.return_value = MagicMock(
            is_failure=lambda: False,
            unwrap=lambda: {"desiredStatus": "RUNNING"},
        )

        with patch(
            "src.providers.runpod.training.api_client.RunPodAPIClient",
            return_value=fake_client,
        ):
            result = resume_pod_for_run(run_dir)

        assert isinstance(result, ResumePodResponse)
        assert result.ok is True
        assert result.availability == "running"


# ---------------------------------------------------------------------------
# 2. Legacy / no-metadata paths
# ---------------------------------------------------------------------------


class TestLegacyPath:
    def test_no_pod_metadata_returns_ok_running(self, tmp_path: Path) -> None:
        # Legacy attempt without pod_metadata → no-op, return ok=true
        # so the UI continues with normal flow.
        run_dir = _seed_run(tmp_path, with_pod=False)
        result = resume_pod_for_run(run_dir)
        assert result.ok is True
        assert result.availability == "running"
        assert "legacy" in result.message.lower() or "no pod" in result.message.lower()

    def test_non_runpod_provider_skipped(self, tmp_path: Path) -> None:
        run_dir = _seed_run(tmp_path, with_pod=True, provider="single_node")
        result = resume_pod_for_run(run_dir)
        assert result.ok is True
        assert result.availability == "running"
        assert "single_node" in result.message


# ---------------------------------------------------------------------------
# 3. Failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_missing_api_key_returns_probe_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        run_dir = _seed_run(tmp_path, with_pod=True)
        result = resume_pod_for_run(run_dir)
        assert result.ok is False
        assert result.availability == "probe_failed"
        assert "RUNPOD_API_KEY" in result.message

    @requires_runpod_sdk
    def test_pod_gone_returns_not_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_dir = _seed_run(tmp_path, with_pod=True)
        monkeypatch.setenv("RUNPOD_API_KEY", "test-key")

        fake_client = MagicMock()
        fake_client.query_pod.return_value = MagicMock(
            is_failure=lambda: False,
            unwrap=lambda: {"desiredStatus": "TERMINATED"},
        )

        with patch(
            "src.providers.runpod.training.api_client.RunPodAPIClient",
            return_value=fake_client,
        ):
            result = resume_pod_for_run(run_dir)

        assert result.ok is False
        assert result.availability == "gone"


# ---------------------------------------------------------------------------
# 4. Sleeping → wake path
# ---------------------------------------------------------------------------


@requires_runpod_sdk
class TestSleepingResume:
    def test_sleeping_pod_woken_successfully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_dir = _seed_run(tmp_path, with_pod=True)
        monkeypatch.setenv("RUNPOD_API_KEY", "test-key")

        fake_client = MagicMock()
        fake_client.query_pod.return_value = MagicMock(
            is_failure=lambda: False,
            unwrap=lambda: {"desiredStatus": "EXITED"},
        )
        fake_client.resume_pod.return_value = MagicMock(
            is_failure=lambda: False,
        )

        with patch(
            "src.providers.runpod.training.api_client.RunPodAPIClient",
            return_value=fake_client,
        ):
            result = resume_pod_for_run(run_dir)

        assert result.ok is True
        assert result.availability == "running"
        assert "resumed" in result.message.lower()


# ---------------------------------------------------------------------------
# 5. ResumePodResponse contract
# ---------------------------------------------------------------------------


class TestResponseContract:
    def test_to_dict_has_three_fields(self) -> None:
        # Pin the response shape — REST router returns exactly these
        # three fields and the Web UI's TypeScript hook depends on
        # them.
        r = ResumePodResponse(availability="running", ok=True, message="x")
        d = r.to_dict()
        assert set(d.keys()) == {"availability", "ok", "message"}
        assert d["availability"] == "running"
        assert d["ok"] is True
        assert d["message"] == "x"
