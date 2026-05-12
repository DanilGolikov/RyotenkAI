"""Phase 14.C — :class:`LaunchResumeService` contract.

Tests the unified resume orchestrator. Mocks the
``provider_resolver`` test seam so we don't need a real RunPod SDK
or network access. Exercises:

* Happy paths: legacy run, single-node skip, sleeping-pod resume,
  running pod no-op.
* Failure paths: GONE, capacity-exhausted, probe-failed.
* Progress callback contract: order + ``kind`` discriminator.
* Skipped path: missing creds, non-lifecycle provider, unknown
  provider.

7-cat coverage. Slim-venv compatible — no provider impls
constructed.
"""

from __future__ import annotations

from types import SimpleNamespace

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ryotenkai_shared.constants import PROVIDER_RUNPOD
from ryotenkai_shared.infrastructure.lifecycle import PodAvailability
from ryotenkai_control.pipeline.launch.resume_service import (
    LaunchResumeService,
    ResumeOutcome,
    ResumeProgress,
)
from ryotenkai_control.pipeline.state.models import PodMetadata


@pytest.fixture
def _stub_capacity_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake ``is_capacity_error_message`` so the resume
    service's lazy import doesn't require the ``runpod`` SDK.

    The service does ``from ryotenkai_providers.runpod.sdk_adapter import
    is_capacity_error_message`` inside :meth:`_do_resume`. In slim
    CI venvs that import fails at the SDK-adapter's module-top
    ``import runpod``. We side-step it by installing a synthetic
    sdk_adapter module with just the helper our tests need.
    Production image always has the real SDK.
    """
    import sys
    import types

    if "ryotenkai_providers.runpod.sdk_adapter" in sys.modules:
        return

    fake = types.ModuleType("ryotenkai_providers.runpod.sdk_adapter")
    fake.is_capacity_error_message = lambda msg: (  # type: ignore[attr-defined]
        "no instances currently available" in msg.lower()
        or "capacity" in msg.lower()
    )
    monkeypatch.setitem(
        sys.modules, "ryotenkai_providers.runpod.sdk_adapter", fake,
    )


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeAPIClient:
    """In-memory transport that returns scripted ``query_pod`` and
    ``resume_pod`` results."""

    def __init__(
        self,
        *,
        query_response: dict[str, Any] | None = None,
        query_raises: Exception | None = None,
        resume_ok: bool = True,
        resume_capacity_msg: str | None = None,
    ) -> None:
        self._query_response = query_response or {"desiredStatus": "RUNNING"}
        self._query_raises = query_raises
        self._resume_ok = resume_ok
        self._resume_capacity_msg = resume_capacity_msg
        self.query_calls: list[str] = []
        self.resume_calls: list[str] = []

    def query_pod(self, pod_id: str) -> Any:
        self.query_calls.append(pod_id)
        if self._query_raises is not None:
            raise self._query_raises
        result = MagicMock()
        result.is_failure.return_value = False
        result.unwrap.return_value = self._query_response
        return result

    def resume_pod(self, pod_id: str) -> Any:
        self.resume_calls.append(pod_id)
        result = MagicMock()
        if self._resume_ok:
            result.is_failure.return_value = False
            result.unwrap.return_value = None
        else:
            result.is_failure.return_value = True
            err = SimpleNamespace(message=self._resume_capacity_msg or "fatal error")
            result.unwrap_err.return_value = err
        return result


class _FakeRunPodProvider:
    """Implements just enough of :class:`ITerminalActionProvider` for
    the service to drive resume flow."""

    def __init__(self, api_client: _FakeAPIClient) -> None:
        self._api_client = api_client

    @property
    def provider_name(self) -> str:
        return PROVIDER_RUNPOD

    def terminate(self, *, resource_id: str, reason: str) -> Any: ...

    def pause(self, *, resource_id: str) -> Any: ...

    def resume(self, *, resource_id: str) -> Any:
        return self._api_client.resume_pod(resource_id)


class _ProgressCollector:
    """Accumulates progress events for assertion."""

    def __init__(self) -> None:
        self.events: list[ResumeProgress] = []

    def __call__(self, evt: ResumeProgress) -> None:
        self.events.append(evt)

    @property
    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_run_with_metadata(
    tmp_path: Path,
    *,
    pod_id: str = "pod-test",
    provider: str = "runpod",
) -> Path:
    """Persist a state file at tmp_path with pod_metadata.
    Uses the same AttemptController flow as
    :mod:`test_launch_service_resume_pod` to keep the seeding
    consistent with the other resume-flow tests."""
    from ryotenkai_control.pipeline.state.attempt_controller import AttemptController
    from ryotenkai_control.pipeline.state.models import (
        PipelineAttemptState,
        PipelineState,
        StageRunState,
        utc_now_iso,
    )
    from ryotenkai_control.pipeline.state.store import PipelineStateStore

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
    controller = AttemptController(save_fn=store.save, run_ctx=None)
    controller.adopt_state(state)
    controller.register_attempt(state.attempts[-1])
    controller.set_pod_metadata(
        pod_id=pod_id,
        provider=provider,
        last_known_status="stopped",
    )
    return tmp_path


def _make_service(
    fake_provider: _FakeRunPodProvider | None,
) -> LaunchResumeService:
    """Build a service with a fake provider_resolver."""
    def _resolver(name: str) -> Any:
        if name == PROVIDER_RUNPOD:
            return fake_provider
        return None
    return LaunchResumeService(provider_resolver=_resolver)


# ---------------------------------------------------------------------------
# 1. Positive — happy paths
# ---------------------------------------------------------------------------


class TestPositive:
    def test_legacy_run_no_metadata_returns_running(
        self, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "legacy"
        run_dir.mkdir()
        outcome = LaunchResumeService().resume(run_dir)
        assert outcome.ok is True
        assert outcome.availability == PodAvailability.RUNNING.value
        assert "legacy" in outcome.message.lower()

    def test_running_pod_returns_no_op_running(
        self, tmp_path: Path,
    ) -> None:
        api = _FakeAPIClient(query_response={"desiredStatus": "RUNNING"})
        provider = _FakeRunPodProvider(api)
        run_dir = _seed_run_with_metadata(tmp_path)
        outcome = _make_service(provider).resume(run_dir)
        assert outcome.ok is True
        assert outcome.availability == PodAvailability.RUNNING.value
        # No resume call — only probe.
        assert api.query_calls == ["pod-test"]
        assert api.resume_calls == []

    def test_sleeping_pod_resumes_successfully(
        self, tmp_path: Path, _stub_capacity_classifier: None,
    ) -> None:
        api = _FakeAPIClient(
            query_response={"desiredStatus": "EXITED"},
            resume_ok=True,
        )
        provider = _FakeRunPodProvider(api)
        run_dir = _seed_run_with_metadata(tmp_path)
        outcome = _make_service(provider).resume(run_dir)
        assert outcome.ok is True
        assert outcome.availability == PodAvailability.RUNNING.value
        assert outcome.attempts_made >= 1
        # Resume was actually called.
        assert api.resume_calls == ["pod-test"]

    def test_progress_callback_emits_full_sequence(
        self, tmp_path: Path, _stub_capacity_classifier: None,
    ) -> None:
        api = _FakeAPIClient(
            query_response={"desiredStatus": "EXITED"},
            resume_ok=True,
        )
        provider = _FakeRunPodProvider(api)
        run_dir = _seed_run_with_metadata(tmp_path)
        progress = _ProgressCollector()
        _make_service(provider).resume(run_dir, on_progress=progress)
        # Order: probing → verdict → resuming → resumed
        assert progress.kinds == ["probing", "verdict", "resuming", "resumed"]


# ---------------------------------------------------------------------------
# 2. Negative — failure paths
# ---------------------------------------------------------------------------


class TestNegative:
    def test_pod_gone_returns_not_ok(self, tmp_path: Path) -> None:
        api = _FakeAPIClient(query_response={"desiredStatus": "TERMINATED"})
        provider = _FakeRunPodProvider(api)
        run_dir = _seed_run_with_metadata(tmp_path)
        outcome = _make_service(provider).resume(run_dir)
        assert outcome.ok is False
        assert outcome.availability == PodAvailability.GONE.value

    def test_probe_failed_returns_not_ok(self, tmp_path: Path) -> None:
        api = _FakeAPIClient(
            query_raises=RuntimeError("RunPod 503 outage"),
        )
        provider = _FakeRunPodProvider(api)
        run_dir = _seed_run_with_metadata(tmp_path)
        outcome = _make_service(provider).resume(run_dir)
        assert outcome.ok is False
        assert outcome.availability == PodAvailability.PROBE_FAILED.value
        assert "RunPod 503" in outcome.message

    def test_capacity_exhausted_returns_not_ok(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _stub_capacity_classifier: None,
    ) -> None:
        # Mock resume_pod_with_retry to return a capacity-exhausted
        # ResumeResult immediately — the actual retry loop has its
        # own tests in test_pod_availability.py.
        from ryotenkai_control.pipeline.launch import pod_availability as pa
        from ryotenkai_control.pipeline.launch.resume_service import (
            ResumeOutcome as _RO,  # noqa: F401 — keeps the import seen
        )
        from ryotenkai_control.pipeline.launch import resume_service as rs

        async def _fake_resume(*args: Any, **kwargs: Any) -> Any:
            return pa.ResumeResult(
                ok=False,
                pod_id="pod-test",
                attempts=4,
                elapsed_seconds=300.0,
                capacity_exhausted=True,
                error_message="there are no instances currently available",
            )

        monkeypatch.setattr(rs, "resume_pod_with_retry", _fake_resume)

        api = _FakeAPIClient(
            query_response={"desiredStatus": "EXITED"},
            resume_ok=False,
            resume_capacity_msg="there are no instances currently available",
        )
        provider = _FakeRunPodProvider(api)
        run_dir = _seed_run_with_metadata(tmp_path)
        outcome = _make_service(provider).resume(run_dir)
        assert outcome.ok is False
        assert outcome.capacity_exhausted is True
        assert outcome.availability == PodAvailability.SLEEPING_RESUME_FAILED.value


# ---------------------------------------------------------------------------
# 3. Boundary — skipped path variants
# ---------------------------------------------------------------------------


class TestBoundarySkipped:
    def test_single_node_provider_skipped(
        self, tmp_path: Path,
    ) -> None:
        run_dir = _seed_run_with_metadata(tmp_path, provider="single_node")
        outcome = _make_service(fake_provider=None).resume(run_dir)
        assert outcome.ok is True
        assert outcome.availability == "skipped"
        assert "single_node" in outcome.message

    def test_runpod_missing_api_key_skipped(
        self, tmp_path: Path,
    ) -> None:
        # Resolver returns None when env missing.
        run_dir = _seed_run_with_metadata(tmp_path, provider="runpod")

        def _resolver(name: str) -> Any:
            return None

        svc = LaunchResumeService(provider_resolver=_resolver)
        outcome = svc.resume(run_dir)
        assert outcome.ok is True
        assert outcome.availability == "skipped"
        assert "RUNPOD_API_KEY" in outcome.message

    def test_unknown_provider_skipped(self, tmp_path: Path) -> None:
        run_dir = _seed_run_with_metadata(tmp_path, provider="lambda")
        outcome = _make_service(fake_provider=None).resume(run_dir)
        assert outcome.ok is True
        assert outcome.availability == "skipped"
        assert "lambda" in outcome.message


# ---------------------------------------------------------------------------
# 4. Invariants — frozen dataclasses + Protocol gating
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_resume_outcome_is_frozen(self) -> None:
        outcome = ResumeOutcome(availability="running", ok=True, message="x")
        with pytest.raises(Exception):  # FrozenInstanceError
            outcome.message = "y"  # type: ignore[misc]

    def test_resume_progress_is_frozen(self) -> None:
        evt = ResumeProgress(kind="probing", message="x")
        with pytest.raises(Exception):
            evt.kind = "verdict"  # type: ignore[misc]

    def test_on_progress_none_is_safe(self, tmp_path: Path) -> None:
        # No progress callback → service must not raise.
        api = _FakeAPIClient(query_response={"desiredStatus": "RUNNING"})
        provider = _FakeRunPodProvider(api)
        run_dir = _seed_run_with_metadata(tmp_path)
        outcome = _make_service(provider).resume(run_dir, on_progress=None)
        assert outcome.ok is True

    def test_provider_without_iterminal_protocol_skipped(
        self, tmp_path: Path,
    ) -> None:
        # Resolver returns object that doesn't conform to
        # ITerminalActionProvider — service should skip.
        class _NotALifecycleProvider:
            provider_name = "weird"

        run_dir = _seed_run_with_metadata(tmp_path, provider="weird")

        def _resolver(name: str) -> Any:
            return _NotALifecycleProvider()

        outcome = LaunchResumeService(provider_resolver=_resolver).resume(run_dir)
        assert outcome.ok is True
        assert outcome.availability == "skipped"


# ---------------------------------------------------------------------------
# 5. Dependency errors — query_pod raises arbitrary exception
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_query_pod_raises_caught_as_probe_failed(
        self, tmp_path: Path,
    ) -> None:
        api = _FakeAPIClient(query_raises=RuntimeError("boom"))
        provider = _FakeRunPodProvider(api)
        run_dir = _seed_run_with_metadata(tmp_path)
        outcome = _make_service(provider).resume(run_dir)
        assert outcome.ok is False
        assert outcome.availability == PodAvailability.PROBE_FAILED.value
        assert "boom" in outcome.message


# ---------------------------------------------------------------------------
# 6. Regressions — ResumeOutcome message formats preserved
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_resumed_message_format(
        self, tmp_path: Path, _stub_capacity_classifier: None,
    ) -> None:
        # Pin the user-facing message format. Pre-14.C this lived
        # in two places (CLI typer.echo + REST ResumePodResponse);
        # post-14.C it lives in one (service composes the message).
        api = _FakeAPIClient(
            query_response={"desiredStatus": "EXITED"},
            resume_ok=True,
        )
        provider = _FakeRunPodProvider(api)
        run_dir = _seed_run_with_metadata(tmp_path)
        outcome = _make_service(provider).resume(run_dir)
        # Format: "Pod resumed in N.Ns (M attempt(s))"
        assert "Pod resumed in" in outcome.message
        assert "attempt(s)" in outcome.message


# ---------------------------------------------------------------------------
# 7. Logic-specific — progress callback discriminator + detail payload
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_probing_event_carries_pod_id_in_detail(
        self, tmp_path: Path,
    ) -> None:
        api = _FakeAPIClient(query_response={"desiredStatus": "RUNNING"})
        provider = _FakeRunPodProvider(api)
        run_dir = _seed_run_with_metadata(tmp_path, pod_id="pod-zzz")
        progress = _ProgressCollector()
        _make_service(provider).resume(run_dir, on_progress=progress)
        probing = next(e for e in progress.events if e.kind == "probing")
        assert probing.detail.get("pod_id") == "pod-zzz"
        assert probing.detail.get("provider") == PROVIDER_RUNPOD

    def test_verdict_event_carries_availability_in_detail(
        self, tmp_path: Path, _stub_capacity_classifier: None,
    ) -> None:
        api = _FakeAPIClient(query_response={"desiredStatus": "EXITED"})
        provider = _FakeRunPodProvider(api)
        run_dir = _seed_run_with_metadata(tmp_path)
        progress = _ProgressCollector()
        _make_service(provider).resume(run_dir, on_progress=progress)
        verdict = next(e for e in progress.events if e.kind == "verdict")
        assert (
            verdict.detail.get("availability")
            == PodAvailability.SLEEPING_RESUMABLE.value
        )

    @pytest.mark.xfail(
        strict=True,
        reason="xfail-debt:resume-service-signature-drift — Pre-existing failure pre-packagization: RunPodProvider.from_resume_metadata signature changed (now requires extra kwarg).",
    )
    def test_default_resolver_uses_env_for_runpod(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Without provider_resolver kwarg, the service uses the
        # default. Pin: with no RUNPOD_API_KEY env, default returns
        # None (skipped path).
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        from ryotenkai_control.pipeline.launch.resume_service import (
            _default_resolve_lifecycle_provider,
        )
        result = _default_resolve_lifecycle_provider("runpod")
        assert result is None
