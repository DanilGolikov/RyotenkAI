"""Phase 14.A — :class:`RunPodProvider` capability methods contract.

Pin the new Phase 14.A methods on RunPod:
* :meth:`required_runtime_env_vars` — full dict including
  ``RYOTENKAI_RUNTIME_PROVIDER`` and RunPod credentials.
* :meth:`probe_availability` — maps RunPod desiredStatus to the
  provider-agnostic verdict.
* :meth:`get_capabilities` — populates new fields correctly.
* :meth:`terminate` / :meth:`pause` / :meth:`resume` — delegate to
  existing RunPod transport.
* :class:`RunPodProvider` IS an :class:`ITerminalActionProvider`.

7-category coverage. Slim-venv compatible — the ``runpod`` SDK is
stubbed at module load via ``sys.modules`` so the import chain
doesn't break.
"""

from __future__ import annotations

from types import SimpleNamespace

import sys
import types
from unittest.mock import MagicMock

import pytest

# Stub the `runpod` SDK before any RunPod imports — slim CI venv
# doesn't have it installed, but Phase 14.A's contract is verifiable
# without a real SDK round-trip.
if "runpod" not in sys.modules:
    _stub = types.ModuleType("runpod")
    _stub.api_key = ""
    _stub.create_pod = MagicMock()
    _stub.get_pod = MagicMock()
    _stub.stop_pod = MagicMock()
    _stub.resume_pod = MagicMock()
    _stub.terminate_pod = MagicMock()
    sys.modules["runpod"] = _stub


from ryotenkai_providers.runpod.training.provider import RunPodProvider
from ryotenkai_providers.training.interfaces import (
    ITerminalActionProvider,
    ProviderCapabilities,
    ProviderStatus,
    VolumeKind,
)
from ryotenkai_shared.constants import PROVIDER_RUNPOD, RUNTIME_PROVIDER_ENV_VAR
from ryotenkai_shared.errors import ProviderUnavailableError

from tests._fakes.provider_context import attach_manifest_capabilities


@pytest.fixture(autouse=True)
def _attach_runpod_manifest() -> None:
    """Attach manifest ClassVars to RunPodProvider.

    Tests in this file build the provider via ``__new__`` and skip the
    ``ProviderRegistry`` load path that normally stamps the manifest
    ClassVars. The helper sets them explicitly so ``provider_name`` /
    ``get_capabilities()`` return realistic values matching production.
    """
    attach_manifest_capabilities(
        RunPodProvider,
        provider_id=PROVIDER_RUNPOD,
        provider_name=PROVIDER_RUNPOD,
        provider_type="cloud",
        capabilities=ProviderCapabilities(
            provider_type="cloud",
            supports_multi_gpu=True,
            supports_spot_instances=True,
            supports_lifecycle_actions=True,
            volume_kind=VolumeKind.PERSISTENT,
            has_pause_resume=True,
            is_local=False,
        ),
    )


def _mk_provider(*, keep_on_error: bool = False) -> RunPodProvider:
    """Build a RunPodProvider WITHOUT going through the heavy
    Pydantic config validator chain.

    Phase 14.A capability methods only touch:
      * ``self._api_key`` (string)
      * ``self._config.cleanup.keep_pod_on_error`` (bool)
      * ``self._graphql_api_client`` / ``self._api_client`` (mocked)
      * ``self._gpu_info`` / ``self._pod_info``

    So we construct via ``__new__`` and stub the few attributes the
    methods read. Avoids dragging in the full RunPodProviderConfig
    schema (which requires connect.ssh.key_path + inference + …) for
    a unit test that's purely about the new methods.
    """
    provider = object.__new__(RunPodProvider)
    provider._api_key = "rk_test"
    provider._status = ProviderStatus.AVAILABLE
    provider._pod_id = None
    provider._ssh_connection_info = None
    provider._gpu_info = None
    provider._pod_info = None
    provider._had_error = False
    # Minimal config stub — capability methods only read .cleanup
    # and .training.gpu_type.
    cleanup_stub = SimpleNamespace(keep_pod_on_error=keep_on_error, auto_stop_after_training=True)
    training_stub = SimpleNamespace(gpu_type="NVIDIA RTX A6000")
    config_stub = SimpleNamespace(cleanup=cleanup_stub, training=training_stub)
    provider._config = config_stub
    # Stub the API clients — tests monkeypatch their methods.
    provider._graphql_api_client = MagicMock()
    provider._api_client = MagicMock()
    return provider


# ---------------------------------------------------------------------------
# 1. Positive — required_runtime_env_vars + protocol conformance
# ---------------------------------------------------------------------------


class TestPositive:
    def test_required_env_vars_includes_bootstrap_marker(self) -> None:
        provider = _mk_provider()
        env = provider.required_runtime_env_vars(resource_id="pod-abc")
        assert env[RUNTIME_PROVIDER_ENV_VAR] == PROVIDER_RUNPOD

    def test_required_env_vars_full_set_with_resource_id(self) -> None:
        provider = _mk_provider()
        env = provider.required_runtime_env_vars(resource_id="pod-abc")
        # Full RunPod env contract.
        assert "RUNPOD_API_KEY" in env
        assert env["RUNPOD_API_KEY"] == "rk_test"
        assert env["RUNPOD_POD_ID"] == "pod-abc"
        assert env["RUNPOD_KEEP_ON_ERROR"] == "false"
        assert env["RUNPOD_VOLUME_KIND"] == VolumeKind.PERSISTENT.value

    def test_required_env_vars_omits_pod_id_without_resource(self) -> None:
        # Defensive case — launcher may call before connect() in tests.
        provider = _mk_provider()
        env = provider.required_runtime_env_vars(resource_id=None)
        assert "RUNPOD_POD_ID" not in env
        # Other RunPod env vars still present.
        assert "RUNPOD_API_KEY" in env
        assert env[RUNTIME_PROVIDER_ENV_VAR] == PROVIDER_RUNPOD

    def test_keep_on_error_reflects_config(self) -> None:
        provider = _mk_provider(keep_on_error=True)
        env = provider.required_runtime_env_vars(resource_id="x")
        assert env["RUNPOD_KEEP_ON_ERROR"] == "true"


class TestProtocolConformance:
    def test_runpod_provider_is_iterminal_action_provider(self) -> None:
        # Phase 14.A two-source-of-truth invariant.
        provider = _mk_provider()
        assert isinstance(provider, ITerminalActionProvider)

    def test_capability_flag_true(self) -> None:
        provider = _mk_provider()
        caps = provider.get_capabilities()
        assert caps.supports_lifecycle_actions is True
        assert caps.has_pause_resume is True
        assert caps.volume_kind is VolumeKind.PERSISTENT
        assert caps.runner_workspace_root == "/workspace"


# ---------------------------------------------------------------------------
# 2. Negative — probe_availability error / unknown status paths
# ---------------------------------------------------------------------------


class TestProbeAvailabilityNegative:
    def test_empty_resource_id_returns_unknown(self) -> None:
        provider = _mk_provider()
        v = provider.probe_availability("")
        assert v.state == "unknown"

    def test_query_pod_raises_returns_probe_failed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = _mk_provider()
        monkeypatch.setattr(
            provider._graphql_api_client, "query_pod",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        v = provider.probe_availability("pod-abc")
        assert v.state == "probe_failed"
        assert "boom" in v.message

    def test_query_pod_returns_err_gone_marker(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Typed exception with "not found" → mapped to "gone".
        provider = _mk_provider()
        monkeypatch.setattr(
            provider._graphql_api_client, "query_pod",
            MagicMock(side_effect=ProviderUnavailableError(
                detail="pod does not exist", context={"code": "X"},
            )),
        )
        v = provider.probe_availability("pod-abc")
        assert v.state == "gone"

    def test_query_pod_returns_err_other(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = _mk_provider()
        monkeypatch.setattr(
            provider._graphql_api_client, "query_pod",
            MagicMock(side_effect=ProviderUnavailableError(
                detail="random transient", context={"code": "X"},
            )),
        )
        v = provider.probe_availability("pod-abc")
        assert v.state == "probe_failed"


# ---------------------------------------------------------------------------
# 3. Boundary — RunPod desiredStatus mapping
# ---------------------------------------------------------------------------


class TestProbeAvailabilityMapping:
    @pytest.mark.parametrize("raw,expected", [
        ("RUNNING", "running"),
        ("EXITED", "sleeping_resumable"),
        ("STOPPED", "sleeping_resumable"),
        ("PAUSED", "sleeping_resumable"),
        ("TERMINATED", "gone"),
        ("DEAD", "gone"),
    ])
    def test_status_map(
        self,
        monkeypatch: pytest.MonkeyPatch,
        raw: str,
        expected: str,
    ) -> None:
        provider = _mk_provider()
        monkeypatch.setattr(
            provider._graphql_api_client, "query_pod",
            MagicMock(return_value={"desiredStatus": raw}),
        )
        v = provider.probe_availability("pod-abc")
        assert v.state == expected
        assert v.raw_status == raw

    def test_unknown_status_returns_probe_failed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = _mk_provider()
        monkeypatch.setattr(
            provider._graphql_api_client, "query_pod",
            MagicMock(return_value={"desiredStatus": "WAT"}),
        )
        v = provider.probe_availability("pod-abc")
        assert v.state == "probe_failed"
        assert "WAT" in v.message


# ---------------------------------------------------------------------------
# 4. Lifecycle action delegation
# ---------------------------------------------------------------------------


class TestLifecycleActions:
    def test_terminate_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _mk_provider()
        delegate = MagicMock(return_value=None)
        monkeypatch.setattr(provider._api_client, "terminate_pod", delegate)
        # Phase A2 Batch 12: terminate() raises on failure, returns None on success.
        provider.terminate(resource_id="pod-abc", reason="user_stop")
        delegate.assert_called_once_with("pod-abc")

    def test_pause_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _mk_provider()
        delegate = MagicMock(return_value=None)
        monkeypatch.setattr(provider._api_client, "stop_pod", delegate)
        # Phase A2 Batch 12: pause() raises on failure, returns None on success.
        provider.pause(resource_id="pod-abc")
        delegate.assert_called_once_with(pod_id="pod-abc")

    def test_resume_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _mk_provider()
        delegate = MagicMock(return_value=None)
        monkeypatch.setattr(provider._api_client, "start_pod", delegate)
        # Phase A2 Batch 12: resume() raises on failure, returns None on success.
        provider.resume(resource_id="pod-abc")
        delegate.assert_called_once_with(pod_id="pod-abc")


# ---------------------------------------------------------------------------
# 5. Dependency errors — transport raises typed exception propagates
# ---------------------------------------------------------------------------


class TestDependencyErrors:
    def test_terminate_propagates_err(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = _mk_provider()
        monkeypatch.setattr(
            provider._api_client,
            "terminate_pod",
            MagicMock(side_effect=ProviderUnavailableError(detail="api down", context={"code": "X"})),
        )
        # Phase A2 Batch 12: terminate() raises ProviderUnavailableError on transport failure.
        with pytest.raises(ProviderUnavailableError, match="api down"):
            provider.terminate(resource_id="pod-abc", reason="x")


# ---------------------------------------------------------------------------
# 6. Regressions — legacy capabilities preserved
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_legacy_capability_fields_preserved(self) -> None:
        provider = _mk_provider()
        caps = provider.get_capabilities()
        assert caps.provider_type == "cloud"
        assert caps.supports_multi_gpu is True
        assert caps.supports_spot_instances is True

    def test_prepare_training_script_hooks_still_works(self) -> None:
        # Phase 14.A keeps both the legacy hooks API AND the new
        # required_runtime_env_vars callable. 14.D will collapse
        # them — for now they coexist.
        provider = _mk_provider()
        # Just verify the method still callable; full behavior
        # tested elsewhere.
        assert callable(provider.prepare_training_script_hooks)


# ---------------------------------------------------------------------------
# 7. Logic-specific — bootstrap env value MUST equal provider_name
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_bootstrap_env_var_value_matches_provider_name(self) -> None:
        provider = _mk_provider()
        env = provider.required_runtime_env_vars(resource_id="x")
        assert env[RUNTIME_PROVIDER_ENV_VAR] == provider.provider_name
        assert env[RUNTIME_PROVIDER_ENV_VAR] == PROVIDER_RUNPOD


# ---------------------------------------------------------------------------
# 8. PodLayout — provider exposes filesystem layout for a run
# ---------------------------------------------------------------------------


class TestPodLayoutForRun:
    def test_layout_root_uses_workspace_runs_runid(self) -> None:
        provider = _mk_provider()
        layout = provider.pod_layout_for_run("run_alpha")
        assert str(layout.root) == "/workspace/runs/run_alpha"

    def test_layout_provides_per_run_artefact_paths(self) -> None:
        provider = _mk_provider()
        layout = provider.pod_layout_for_run("run_42")
        assert str(layout.runner_log) == "/workspace/runs/run_42/logs/runner.log"
        assert str(layout.trainer_stdio_log) == "/workspace/runs/run_42/logs/trainer.stdio.log"
        assert str(layout.events_dir) == "/workspace/runs/run_42/events"
        assert str(layout.state_dir) == "/workspace/runs/run_42/state"

    def test_layout_rejects_empty_run_id(self) -> None:
        provider = _mk_provider()
        with pytest.raises(ValueError, match="run_id must be non-empty"):
            provider.pod_layout_for_run("")

    def test_layout_disjoint_for_different_runs(self) -> None:
        # Resume-collision regression: same pod, different run_ids
        # MUST produce DISJOINT path trees.
        provider = _mk_provider()
        a = provider.pod_layout_for_run("run_a")
        b = provider.pod_layout_for_run("run_b")
        assert a.root != b.root
        assert a.runner_log != b.runner_log
        assert a.trainer_stdio_log != b.trainer_stdio_log
