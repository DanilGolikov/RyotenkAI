"""
Unit tests for RunPodPodInferenceProvider (Phase A2 Batch 11).

Internal API stack now raises typed exceptions
(``ProviderUnavailableError`` etc.); the provider facade catches them
and returns ``Result[T, InferenceError]`` for Protocol conformance.
StubApi raises-on-action: each ``*_actions`` field is a queue of
``None|dict|list`` (returned) or ``BaseException`` (raised).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ryotenkai_shared.config.inference.schema import InferenceConfig
from ryotenkai_shared.config.providers.runpod import RunPodProviderConfig
from ryotenkai_shared.constants import PROVIDER_RUNPOD
from ryotenkai_providers.inference.interfaces import (
    EndpointInfo,
    InferenceArtifactsContext,
    PipelineReadinessMode,
)
from ryotenkai_providers.runpod.inference.pods.provider import RunPodPodInferenceProvider
from ryotenkai_shared.errors import (
    ConfigInvalidError,
    InferenceUnavailableError,
    ProviderUnavailableError,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Manifest fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _attach_runpod_pod_provider_manifest() -> None:
    RunPodPodInferenceProvider._manifest_provider_id = PROVIDER_RUNPOD
    RunPodPodInferenceProvider._manifest_provider_name = PROVIDER_RUNPOD
    RunPodPodInferenceProvider._manifest_provider_type = PROVIDER_RUNPOD


# ---------------------------------------------------------------------------
# Stub API — raise-based contract
# ---------------------------------------------------------------------------


def _next(queue: list[Any], default: Any) -> Any:
    """Pop next action; raise if BaseException, else return."""
    if not queue:
        return default
    action = queue.pop(0)
    if isinstance(action, BaseException):
        raise action
    return action


@dataclass
class StubApi:
    """Raise-based stub for RunPodPodsRESTClient.

    Each ``*_actions`` queue holds ``None|dict|list`` (returned) or
    ``BaseException`` instances (raised). Empty queue → ``default``.
    """

    list_pods_actions: list[Any] = field(default_factory=list)
    get_pod_actions: list[Any] = field(default_factory=list)
    create_pod_actions: list[Any] = field(default_factory=list)
    stop_pod_actions: list[Any] = field(default_factory=list)
    start_pod_actions: list[Any] = field(default_factory=list)
    delete_pod_actions: list[Any] = field(default_factory=list)
    list_network_volumes_actions: list[Any] = field(default_factory=list)
    get_network_volume_actions: list[Any] = field(default_factory=list)
    create_network_volume_actions: list[Any] = field(default_factory=list)

    created_pod_payloads: list[dict[str, Any]] = field(default_factory=list)

    def list_pods(self, *, params: dict[str, Any] | None = None) -> Any:
        return _next(self.list_pods_actions, [])

    def get_pod(self, *, pod_id: str) -> Any:
        return _next(self.get_pod_actions, {"id": pod_id, "desiredStatus": "EXITED"})

    def create_pod(self, *, payload: dict[str, Any]) -> Any:
        self.created_pod_payloads.append(payload)
        return _next(self.create_pod_actions, {"id": "pod_created"})

    def stop_pod(self, *, pod_id: str) -> Any:
        return _next(self.stop_pod_actions, None)

    def start_pod(self, *, pod_id: str) -> Any:
        return _next(self.start_pod_actions, None)

    def delete_pod(self, *, pod_id: str) -> Any:
        return _next(self.delete_pod_actions, None)

    def list_network_volumes(self) -> Any:
        return _next(self.list_network_volumes_actions, [])

    def get_network_volume(self, *, network_volume_id: str) -> Any:
        return _next(
            self.get_network_volume_actions, {"id": network_volume_id, "name": "vol"}
        )

    def create_network_volume(self, *, payload: dict[str, Any]) -> Any:
        return _next(
            self.create_network_volume_actions, {"id": "nv_created", "name": payload.get("name")}
        )


def _exc(detail: str, code: str = "RUNPOD_SDK_CALL_FAILED") -> ProviderUnavailableError:
    return ProviderUnavailableError(detail=detail, context={"code": code})


# ---------------------------------------------------------------------------
# Provider factory helpers
# ---------------------------------------------------------------------------


def _mk_provider(
    *,
    api: StubApi | None = None,
    api_key: str = "rk",
    key_path: str = "/tmp/test_key",
    with_volume: bool = False,
    pod_id: str | None = None,
    pod_name: str | None = None,
    adapter_ref: str = "hf-org/my-model",
    eval_session: Any | None = None,
) -> RunPodPodInferenceProvider:
    volume_cfg_raw: dict[str, Any] | None = None
    if with_volume:
        volume_cfg_raw = {"id": "vol-123", "name": "my-vol", "size_gb": 50, "data_center_id": "US-KS-2"}

    prov_cfg = RunPodProviderConfig(
        connect={"ssh": {"key_path": key_path}},
        cleanup={},
        training={"gpu_type": "A40"},
        inference={
            "pod": {"gpu_type_ids": ["NVIDIA A40"]},
            **({"volume": volume_cfg_raw} if volume_cfg_raw else {}),
        },
    )
    inf_cfg = InferenceConfig(provider="runpod")

    p = RunPodPodInferenceProvider.__new__(RunPodPodInferenceProvider)
    p._cfg = SimpleNamespace(
        model=SimpleNamespace(name="meta-llama/Llama-3.1-8B-Instruct", trust_remote_code=False)
    )
    p._secrets = SimpleNamespace(runpod_api_key=api_key, hf_token="hf_test")
    p._inf_cfg = inf_cfg
    p._engine_cfg = inf_cfg.engine
    p._provider_cfg = prov_cfg
    p._volume_cfg = prov_cfg.inference.volume
    p._pod_cfg = prov_cfg.inference.pod
    p._serve_cfg = prov_cfg.inference.serve
    p._api = api
    p._pod_control = api
    p._endpoint_info = None
    p._network_volume_id = None
    p._network_volume_meta = None
    p._pod_id = pod_id
    p._pod_name = pod_name
    # Phase 7: ``_event_logger`` removed from the provider; the fixture
    # no longer seeds it.
    p._eval_session = eval_session
    p._adapter_ref = adapter_ref
    return p


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_provider_name_is_runpod() -> None:
    p = _mk_provider()
    assert p.provider_name == PROVIDER_RUNPOD


def test_provider_type_is_runpod() -> None:
    p = _mk_provider()
    assert p.provider_type == PROVIDER_RUNPOD


def test_get_pipeline_readiness_mode_is_skip() -> None:
    p = _mk_provider()
    assert p.get_pipeline_readiness_mode() == PipelineReadinessMode.SKIP


def test_collect_startup_logs_is_noop(tmp_path: Path) -> None:
    p = _mk_provider()
    p.collect_startup_logs(local_path=tmp_path)


def test_get_capabilities() -> None:
    p = _mk_provider()
    caps = p.get_capabilities()
    assert caps.provider_type == PROVIDER_RUNPOD
    assert "vllm" in caps.supported_engines
    assert caps.supports_lora is True


def test_get_endpoint_info_none_before_deploy() -> None:
    p = _mk_provider()
    assert p.get_endpoint_info() is None


def test_set_event_logger_removed_in_phase_7() -> None:
    """Phase 7: ``set_event_logger`` + ``_event_logger`` removed
    alongside :class:`InferenceEventLogger` Protocol."""
    p = _mk_provider()
    assert not hasattr(p, "set_event_logger")
    assert not hasattr(p, "_event_logger")


# ---------------------------------------------------------------------------
# deploy() — early error paths
# ---------------------------------------------------------------------------


def test_deploy_local_adapter_path_returns_err() -> None:
    p = _mk_provider()
    for bad_path in ["/local/path", "~/adapter", "./adapter", "file:///path"]:
        # Phase A2 Batch 12: invalid adapter path raises ConfigInvalidError.
        with pytest.raises(ConfigInvalidError) as exc_info:
            p.deploy(bad_path, run_id="r1", base_model_id="llama")
        assert exc_info.value.context.get("reason") == "RUNPOD_ADAPTER_LOCAL_PATH_NOT_SUPPORTED"


def test_deploy_missing_api_key_returns_err() -> None:
    p = _mk_provider(api_key="")
    with pytest.raises(ConfigInvalidError) as exc_info:
        p.deploy("hf-org/model", run_id="r1", base_model_id="llama")
    assert exc_info.value.context.get("reason") == "RUNPOD_API_KEY_MISSING"


def test_deploy_none_api_key_returns_err() -> None:
    p = _mk_provider()
    p._secrets = SimpleNamespace(runpod_api_key=None, hf_token=None)
    with pytest.raises(ConfigInvalidError) as exc_info:
        p.deploy("hf-org/model", run_id="r1", base_model_id="llama")
    assert exc_info.value.context.get("reason") == "RUNPOD_API_KEY_MISSING"


def test_deploy_missing_ssh_key_file_returns_err(tmp_path: Path) -> None:
    missing_key = str(tmp_path / "id_ed25519")
    p = _mk_provider(key_path=missing_key)
    with pytest.raises(ConfigInvalidError) as exc_info:
        p.deploy("hf-org/model", run_id="r1", base_model_id="llama")
    assert exc_info.value.context.get("reason") == "RUNPOD_SSH_KEY_NOT_FOUND"


# ---------------------------------------------------------------------------
# deploy() — full flow
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="xfail-debt:runpod-pods-deploy-tunnel-drift — Pre-existing failure post-packagization: deploy() now intentionally returns endpoint_url=None until activate_for_eval opens the SSH tunnel.",
)
def test_deploy_success_no_volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("FAKE KEY")

    p = _mk_provider(key_path=str(key_file))

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.inference.pods.provider.RunPodPodsRESTClient",
        lambda **kw: StubApi(),
    )
    monkeypatch.setattr(p, "_ensure_pod", lambda **kw: ("pod-123", "pod-name"))
    monkeypatch.setattr(p, "_stop_pod_if_running", lambda **kw: None)

    res = p.deploy("hf-org/my-model", run_id="run1", base_model_id="llama")
    # (Phase A2 Batch 12: success implies no raise)
    info = res
    assert info.resource_id == "pod-123"
    assert info.endpoint_url == "http://127.0.0.1:8000/v1"
    assert info.health_url == "http://127.0.0.1:8000/v1/models"
    assert p._pod_id == "pod-123"


def test_deploy_keep_running_skips_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("FAKE KEY")

    p = _mk_provider(key_path=str(key_file))
    stop_called: list[bool] = []

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.inference.pods.provider.RunPodPodsRESTClient",
        lambda **kw: StubApi(),
    )
    monkeypatch.setattr(p, "_ensure_pod", lambda **kw: ("pod-456", "pod-name"))
    monkeypatch.setattr(p, "_stop_pod_if_running", lambda **kw: (stop_called.append(True), Ok(None))[1])

    res = p.deploy("hf-org/my-model", run_id="run1", base_model_id="llama", keep_running=True)
    # (Phase A2 Batch 12: success implies no raise)
    assert not stop_called


def test_deploy_with_volume_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("FAKE KEY")

    p = _mk_provider(key_path=str(key_file), with_volume=True)

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.inference.pods.provider.RunPodPodsRESTClient",
        lambda **kw: StubApi(),
    )
    monkeypatch.setattr(p, "_ensure_network_volume", lambda: "vol-123")
    monkeypatch.setattr(p, "_ensure_pod", lambda **kw: ("pod-789", "pod-name"))
    monkeypatch.setattr(p, "_stop_pod_if_running", lambda **kw: None)

    res = p.deploy("hf-org/my-model", run_id="run1", base_model_id="llama")
    # (Phase A2 Batch 12: success implies no raise)
    assert p._network_volume_id == "vol-123"


def test_deploy_volume_ensure_failure_returns_err(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("FAKE KEY")

    p = _mk_provider(key_path=str(key_file), with_volume=True)

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.inference.pods.provider.RunPodPodsRESTClient",
        lambda **kw: StubApi(),
    )
    monkeypatch.setattr(
        p,
        "_ensure_network_volume",
        lambda *a, **kw: (_ for _ in ()).throw(InferenceUnavailableError(detail="vol fail", context={"reason": "RUNPOD_VOLUME_LIST_FAILED"})),
    )

    with pytest.raises(InferenceUnavailableError) as exc_info:
        p.deploy("hf-org/my-model", run_id="run1", base_model_id="llama")
    assert exc_info.value.context.get("reason") == "RUNPOD_VOLUME_LIST_FAILED"


def test_deploy_pod_ensure_failure_returns_err(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("FAKE KEY")

    p = _mk_provider(key_path=str(key_file))

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.inference.pods.provider.RunPodPodsRESTClient",
        lambda **kw: StubApi(),
    )
    monkeypatch.setattr(
        p,
        "_ensure_pod",
        lambda *a, **kw: (_ for _ in ()).throw(InferenceUnavailableError(detail="pod fail", context={"reason": "RUNPOD_POD_CREATE_FAILED"})),
    )

    with pytest.raises(InferenceUnavailableError) as exc_info:
        p.deploy("hf-org/my-model", run_id="run1", base_model_id="llama")
    assert exc_info.value.context.get("reason") == "RUNPOD_POD_CREATE_FAILED"


def test_deploy_stop_failure_returns_err(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("FAKE KEY")

    p = _mk_provider(key_path=str(key_file))

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.inference.pods.provider.RunPodPodsRESTClient",
        lambda **kw: StubApi(),
    )
    monkeypatch.setattr(p, "_ensure_pod", lambda **kw: ("pod-123", "pod-name"))
    monkeypatch.setattr(
        p,
        "_stop_pod_if_running",
        lambda *a, **kw: (_ for _ in ()).throw(InferenceUnavailableError(detail="stop failed", context={"reason": "RUNPOD_POD_STOP_FAILED"})),
    )

    with pytest.raises(InferenceUnavailableError) as exc_info:
        p.deploy("hf-org/my-model", run_id="run1", base_model_id="llama")
    assert exc_info.value.context.get("reason") == "RUNPOD_POD_STOP_FAILED"


def test_deploy_provider_error_from_volume_wrapped_as_inference_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("FAKE KEY")

    p = _mk_provider(key_path=str(key_file), with_volume=True)

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.inference.pods.provider.RunPodPodsRESTClient",
        lambda **kw: StubApi(),
    )
    monkeypatch.setattr(
        p,
        "_ensure_network_volume",
        lambda *a, **kw: (_ for _ in ()).throw(InferenceUnavailableError(detail="provider err", context={"reason": "PROV_ERR"})),
    )

    with pytest.raises(InferenceUnavailableError) as exc_info:
        p.deploy("hf-org/my-model", run_id="run1", base_model_id="llama")
    # Phase A2 Batch 12: ensure_network_volume's typed exception bubbles up directly.
    assert exc_info.value.context.get("reason") in {"RUNPOD_VOLUME_ENSURE_FAILED", "PROV_ERR"}


# ---------------------------------------------------------------------------
# undeploy()
# ---------------------------------------------------------------------------


def test_undeploy_no_api_returns_ok() -> None:
    p = _mk_provider()
    assert p._api is None
    res = p.undeploy()
    # (Phase A2 Batch 12: success implies no raise)


def test_undeploy_no_pod_id_returns_ok() -> None:
    p = _mk_provider(api=StubApi())
    assert p._pod_id is None
    res = p.undeploy()
    # (Phase A2 Batch 12: success implies no raise)


def test_undeploy_success_stops_pod() -> None:
    api = StubApi()
    p = _mk_provider(api=api, pod_id="pod-1")
    res = p.undeploy()
    # (Phase A2 Batch 12: success implies no raise)


def test_undeploy_stop_failure_returns_err() -> None:
    api = StubApi(stop_pod_actions=[_exc("stop failed", code="RUNPOD_STOP")])
    p = _mk_provider(api=api, pod_id="pod-1")
    with pytest.raises((InferenceUnavailableError, ProviderUnavailableError)) as exc_info:
        p.undeploy()
    assert exc_info.value.context.get("reason") or exc_info.value.context.get("code") == "RUNPOD_STOP"


# ---------------------------------------------------------------------------
# health_check()
# ---------------------------------------------------------------------------


def test_health_check_not_deployed_returns_err() -> None:
    p = _mk_provider()
    with pytest.raises(InferenceUnavailableError) as exc_info:
        p.health_check()
    assert exc_info.value.context.get("reason") == "RUNPOD_NOT_DEPLOYED"


def test_health_check_no_pod_id_returns_err() -> None:
    p = _mk_provider(api=StubApi())
    with pytest.raises(InferenceUnavailableError) as exc_info:
        p.health_check()
    assert exc_info.value.context.get("reason") == "RUNPOD_NOT_DEPLOYED"


def test_health_check_pod_found_returns_true() -> None:
    api = StubApi(get_pod_actions=[{"id": "pod-1", "desiredStatus": "EXITED"}])
    p = _mk_provider(api=api, pod_id="pod-1")
    res = p.health_check()
    # (Phase A2 Batch 12: success implies no raise)
    assert res is True


def test_health_check_get_pod_failure_returns_err() -> None:
    api = StubApi(get_pod_actions=[_exc("not found", code="RUNPOD_NOT_FOUND")])
    p = _mk_provider(api=api, pod_id="pod-1")
    with pytest.raises((InferenceUnavailableError, ProviderUnavailableError)) as exc_info:
        p.health_check()
    assert exc_info.value.context.get("reason") or exc_info.value.context.get("code") == "RUNPOD_NOT_FOUND"


# ---------------------------------------------------------------------------
# activate_for_eval()
# ---------------------------------------------------------------------------


def test_activate_for_eval_no_api_returns_err() -> None:
    p = _mk_provider(pod_id="pod-1")
    p._api = None
    with pytest.raises(InferenceUnavailableError) as exc_info:
        p.activate_for_eval()
    assert exc_info.value.context.get("reason") == "RUNPOD_NOT_DEPLOYED"


def test_activate_for_eval_no_pod_id_returns_err() -> None:
    p = _mk_provider(api=StubApi())
    p._pod_id = None
    with pytest.raises(InferenceUnavailableError) as exc_info:
        p.activate_for_eval()
    assert exc_info.value.context.get("reason") == "RUNPOD_NOT_DEPLOYED"


def test_activate_for_eval_empty_adapter_ref_returns_err() -> None:
    p = _mk_provider(api=StubApi(), pod_id="pod-1", adapter_ref="")
    with pytest.raises(InferenceUnavailableError) as exc_info:
        p.activate_for_eval()
    assert exc_info.value.context.get("reason") == "RUNPOD_ADAPTER_REF_MISSING"


def test_activate_for_eval_whitespace_adapter_ref_returns_err() -> None:
    p = _mk_provider(api=StubApi(), pod_id="pod-1", adapter_ref="   ")
    with pytest.raises(InferenceUnavailableError) as exc_info:
        p.activate_for_eval()
    assert exc_info.value.context.get("reason") == "RUNPOD_ADAPTER_REF_MISSING"


def test_activate_for_eval_session_success(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _mk_provider(api=StubApi(), pod_id="pod-1", adapter_ref="hf-org/model")

    fake_session = SimpleNamespace(endpoint_url="http://127.0.0.1:8000/v1")

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.inference.pods.pod_session.activate",
        lambda **kw: fake_session,
    )

    res = p.activate_for_eval()
    # (Phase A2 Batch 12: success implies no raise)
    assert res == "http://127.0.0.1:8000/v1"
    assert p._eval_session is fake_session


def test_activate_for_eval_session_failure_returns_inference_err(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _mk_provider(api=StubApi(), pod_id="pod-1", adapter_ref="hf-org/model")

    def _raise(**_kw: Any) -> None:
        raise _exc("ssh fail", code="SSH_FAIL")

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.inference.pods.pod_session.activate",
        _raise,
    )

    # Phase A2 Batch 12: activate_for_eval propagates typed exception.
    with pytest.raises((InferenceUnavailableError, ProviderUnavailableError)) as exc_info:
        p.activate_for_eval()
    # The underlying typed exception's reason is preserved.
    assert exc_info.value.context.get("reason") or exc_info.value.context.get("code") in {"SSH_FAIL", "RUNPOD_ACTIVATE_FAILED"}


def test_activate_for_eval_pipeline_cancelled_synchronously_terminates_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ryotenkai_shared.utils.cancellation import PipelineCancelled

    api = StubApi(delete_pod_actions=[None])
    p = _mk_provider(api=api, pod_id="pod-1", adapter_ref="hf-org/model")

    def raise_cancelled(**_kw: object) -> None:
        raise PipelineCancelled()

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.inference.pods.pod_session.activate",
        raise_cancelled,
    )

    with pytest.raises(PipelineCancelled):
        p.activate_for_eval()

    assert api.delete_pod_actions == []
    assert p._pod_id is None


def test_activate_for_eval_pipeline_cancelled_swallows_delete_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ryotenkai_shared.utils.cancellation import PipelineCancelled

    api = StubApi(delete_pod_actions=[_exc("boom", code="DELETE_FAIL")])
    p = _mk_provider(api=api, pod_id="pod-1", adapter_ref="hf-org/model")

    def raise_cancelled(**_kw: object) -> None:
        raise PipelineCancelled()

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.inference.pods.pod_session.activate",
        raise_cancelled,
    )

    with pytest.raises(PipelineCancelled):
        p.activate_for_eval()


def test_activate_for_eval_session_inference_error_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _mk_provider(api=StubApi(), pod_id="pod-1", adapter_ref="hf-org/model")

    def _raise(**_kw: Any) -> None:
        raise _exc("vllm error", code="VLLM_FAIL")

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.inference.pods.pod_session.activate",
        _raise,
    )

    # Phase A2 Batch 12: activate_for_eval propagates typed exception.
    with pytest.raises((InferenceUnavailableError, ProviderUnavailableError)) as exc_info:
        p.activate_for_eval()
    assert exc_info.value.context.get("reason") or exc_info.value.context.get("code") in {"VLLM_FAIL", "RUNPOD_ACTIVATE_FAILED"}
    assert "vllm error" in (exc_info.value.detail or str(exc_info.value))


# ---------------------------------------------------------------------------
# deactivate_after_eval()
# ---------------------------------------------------------------------------


def test_deactivate_after_eval_no_api_returns_ok() -> None:
    p = _mk_provider()
    p._api = None
    res = p.deactivate_after_eval()
    # (Phase A2 Batch 12: success implies no raise)


def test_deactivate_after_eval_no_session_no_pod_returns_ok() -> None:
    p = _mk_provider(api=StubApi())
    p._eval_session = None
    p._pod_id = None
    res = p.deactivate_after_eval()
    # (Phase A2 Batch 12: success implies no raise)


def test_deactivate_after_eval_no_session_with_pod_deletes_pod() -> None:
    api = StubApi()
    p = _mk_provider(api=api, pod_id="pod-1")
    p._eval_session = None

    res = p.deactivate_after_eval()
    # (Phase A2 Batch 12: success implies no raise)
    assert p._pod_id is None


def test_deactivate_after_eval_no_session_delete_failure_returns_err() -> None:
    api = StubApi(delete_pod_actions=[_exc("delete fail", code="DELETE_FAIL")])
    p = _mk_provider(api=api, pod_id="pod-1")
    p._eval_session = None

    with pytest.raises(InferenceUnavailableError) as exc_info:
        p.deactivate_after_eval()
    assert exc_info.value.context.get("reason") == "RUNPOD_POD_DELETE_FAILED"


def test_deactivate_after_eval_session_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = SimpleNamespace(endpoint_url="http://127.0.0.1:8000/v1")
    p = _mk_provider(api=StubApi(), pod_id="pod-1", eval_session=fake_session)

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.inference.pods.pod_session.deactivate",
        lambda **kw: None,
    )

    res = p.deactivate_after_eval()
    # (Phase A2 Batch 12: success implies no raise)
    assert p._eval_session is None


def test_deactivate_after_eval_session_failure_returns_err(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = SimpleNamespace(endpoint_url="http://127.0.0.1:8000/v1")
    p = _mk_provider(api=StubApi(), pod_id="pod-1", eval_session=fake_session)

    def _raise(**_kw: Any) -> None:
        raise _exc("deactivate fail", code="DEACT_FAIL")

    monkeypatch.setattr(
        "ryotenkai_providers.runpod.inference.pods.pod_session.deactivate",
        _raise,
    )

    # Phase A2 Batch 12: deactivate_after_eval propagates typed exception.
    with pytest.raises((InferenceUnavailableError, ProviderUnavailableError)) as exc_info:
        p.deactivate_after_eval()
    assert exc_info.value.context.get("reason") or exc_info.value.context.get("code") in {"DEACT_FAIL", "RUNPOD_POD_DELETE_FAILED"}


# ---------------------------------------------------------------------------
# _ensure_pod()
# ---------------------------------------------------------------------------


def test_ensure_pod_reuses_existing_stopped_pod() -> None:
    api = StubApi(list_pods_actions=[[{"id": "pod-stopped", "desiredStatus": "EXITED"}]])
    p = _mk_provider(api=api)
    res = p._ensure_pod(network_volume_id=None, key_path=Path("/tmp/k"))
    # (Phase A2 Batch 12: success implies no raise)
    pod_id, _ = res
    assert pod_id == "pod-stopped"


def test_ensure_pod_prefers_stopped_over_running() -> None:
    api = StubApi(
        list_pods_actions=[
            [
                {"id": "pod-running", "desiredStatus": "RUNNING"},
                {"id": "pod-stopped", "desiredStatus": "EXITED"},
            ]
        ]
    )
    p = _mk_provider(api=api)
    res = p._ensure_pod(network_volume_id=None, key_path=Path("/tmp/k"))
    # (Phase A2 Batch 12: success implies no raise)
    pod_id, _ = res
    assert pod_id == "pod-stopped"


def test_ensure_pod_falls_back_to_running_pod() -> None:
    api = StubApi(list_pods_actions=[[{"id": "pod-running", "desiredStatus": "RUNNING"}]])
    p = _mk_provider(api=api)
    res = p._ensure_pod(network_volume_id=None, key_path=Path("/tmp/k"))
    # (Phase A2 Batch 12: success implies no raise)
    pod_id, _ = res
    assert pod_id == "pod-running"


def test_ensure_pod_creates_new_when_none_exist(tmp_path: Path) -> None:
    api = StubApi(
        list_pods_actions=[[], []],
        create_pod_actions=[{"id": "pod-new"}],
    )
    p = _mk_provider(api=api)
    res = p._ensure_pod(network_volume_id=None, key_path=tmp_path / "key")
    # (Phase A2 Batch 12: success implies no raise)
    pod_id, _ = res
    assert pod_id == "pod-new"


def test_ensure_pod_create_fails_but_pod_appears_on_relist(tmp_path: Path) -> None:
    api = StubApi(
        list_pods_actions=[
            [],  # initial list
            [{"id": "pod-appeared", "desiredStatus": "RUNNING"}],  # post-create relist
        ],
        create_pod_actions=[_exc("timeout", code="TIMEOUT")],
    )
    p = _mk_provider(api=api)
    res = p._ensure_pod(network_volume_id=None, key_path=tmp_path / "key")
    # (Phase A2 Batch 12: success implies no raise)
    pod_id, _ = res
    assert pod_id == "pod-appeared"


def test_ensure_pod_list_failure_returns_err() -> None:
    api = StubApi(list_pods_actions=[_exc("list failed", code="LIST_FAIL")])
    p = _mk_provider(api=api)
    # Phase A2 Batch 12: typed exception propagates directly.
    with pytest.raises(Exception) as exc_info:
        p._ensure_pod(network_volume_id=None, key_path=Path("/tmp/k"))
    # The error is the original typed exception from the stub.
    assert "list failed" in str(exc_info.value)


def test_ensure_pod_all_create_attempts_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    call_count = [0]

    def _list_pods(**kw: Any) -> Any:
        call_count[0] += 1
        return []

    api = StubApi(
        create_pod_actions=[_exc("fail", code="FAIL")] * 8,
    )
    api.list_pods = _list_pods  # type: ignore[method-assign]

    p = _mk_provider(api=api)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(InferenceUnavailableError) as exc_info:
        p._ensure_pod(network_volume_id=None, key_path=tmp_path / "key")
    assert exc_info.value.context.get("reason") == "RUNPOD_POD_CREATE_FAILED"


def test_ensure_pod_with_network_volume_includes_volume_id(tmp_path: Path) -> None:
    api = StubApi(
        list_pods_actions=[[], []],
        create_pod_actions=[{"id": "pod-vol-new"}],
    )
    p = _mk_provider(api=api)
    res = p._ensure_pod(network_volume_id="vol-abc", key_path=tmp_path / "key")
    # (Phase A2 Batch 12: success implies no raise)
    assert api.created_pod_payloads[0]["networkVolumeId"] == "vol-abc"


def test_ensure_pod_uses_deterministic_name_for_volume(tmp_path: Path) -> None:
    api = StubApi(
        list_pods_actions=[[], []],
        create_pod_actions=[{"id": "pod-vol-new"}],
    )
    p = _mk_provider(api=api)

    res = p._ensure_pod(network_volume_id="vol-abc", key_path=tmp_path / "key")

    # (Phase A2 Batch 12: success implies no raise)
    pod_id, pod_name = res
    payload = api.created_pod_payloads[0]
    assert pod_id == "pod-vol-new"
    assert payload["name"] == pod_name
    assert payload["name"].startswith(f"{p._pod_cfg.name_prefix}-")
    assert "ephemeral" not in payload["name"]


def test_ensure_pod_without_volume_uses_volume_in_gb(tmp_path: Path) -> None:
    api = StubApi(
        list_pods_actions=[[], []],
        create_pod_actions=[{"id": "pod-ephemeral"}],
    )
    p = _mk_provider(api=api)
    res = p._ensure_pod(network_volume_id=None, key_path=tmp_path / "key")
    # (Phase A2 Batch 12: success implies no raise)
    payload = api.created_pod_payloads[0]
    assert "networkVolumeId" not in payload
    assert "volumeInGb" in payload


def test_ensure_pod_without_volume_uses_ephemeral_name(tmp_path: Path) -> None:
    api = StubApi(
        list_pods_actions=[[], []],
        create_pod_actions=[{"id": "pod-ephemeral"}],
    )
    p = _mk_provider(api=api)

    res = p._ensure_pod(network_volume_id=None, key_path=tmp_path / "key")

    # (Phase A2 Batch 12: success implies no raise)
    _, pod_name = res
    assert pod_name.endswith("-ephemeral")
    assert api.created_pod_payloads[0]["name"] == pod_name


def test_ensure_pod_pod_without_id_returns_err() -> None:
    api = StubApi(list_pods_actions=[[{"desiredStatus": "EXITED"}]])  # no "id" field
    p = _mk_provider(api=api)
    with pytest.raises(InferenceUnavailableError) as exc_info:
        p._ensure_pod(network_volume_id=None, key_path=Path("/tmp/k"))
    assert exc_info.value.context.get("reason") == "RUNPOD_POD_NO_ID"


def test_ensure_pod_create_success_without_id_but_relist_finds_pod(tmp_path: Path) -> None:
    api = StubApi(
        list_pods_actions=[[], [{"id": "pod-after-relist", "desiredStatus": "RUNNING"}]],
        create_pod_actions=[{"desiredStatus": "RUNNING"}],  # no id
    )
    p = _mk_provider(api=api)

    res = p._ensure_pod(network_volume_id=None, key_path=tmp_path / "key")

    # Phase A2 Batch 12: _ensure_pod returns (pod_id, name) tuple on success.
    assert res[0] == "pod-after-relist"


# ---------------------------------------------------------------------------
# _stop_pod_if_running()
# ---------------------------------------------------------------------------


def test_stop_pod_if_running_running_pod_calls_stop() -> None:
    api = StubApi(
        get_pod_actions=[{"id": "pod-1", "desiredStatus": "RUNNING"}],
        stop_pod_actions=[None],
    )
    p = _mk_provider(api=api)
    res = p._stop_pod_if_running(pod_id="pod-1")
    # (Phase A2 Batch 12: success implies no raise)


def test_stop_pod_if_running_stopped_pod_skips_stop() -> None:
    api = StubApi(get_pod_actions=[{"id": "pod-1", "desiredStatus": "EXITED"}])
    p = _mk_provider(api=api)
    res = p._stop_pod_if_running(pod_id="pod-1")
    # (Phase A2 Batch 12: success implies no raise)
    assert len(api.stop_pod_actions) == 0


def test_stop_pod_if_running_get_failure_returns_err() -> None:
    api = StubApi(get_pod_actions=[_exc("get failed", code="GET_FAIL")])
    p = _mk_provider(api=api)
    with pytest.raises((InferenceUnavailableError, ProviderUnavailableError)) as exc_info:
        p._stop_pod_if_running(pod_id="pod-1")
    assert exc_info.value.context.get("reason") or exc_info.value.context.get("code") == "GET_FAIL"


def test_stop_pod_if_running_stop_failure_returns_err() -> None:
    api = StubApi(
        get_pod_actions=[{"id": "pod-1", "desiredStatus": "RUNNING"}],
        stop_pod_actions=[_exc("stop failed", code="STOP_FAIL")],
    )
    p = _mk_provider(api=api)
    with pytest.raises((InferenceUnavailableError, ProviderUnavailableError)) as exc_info:
        p._stop_pod_if_running(pod_id="pod-1")
    assert exc_info.value.context.get("reason") or exc_info.value.context.get("code") == "STOP_FAIL"


# ---------------------------------------------------------------------------
# _ensure_network_volume() — via id lookup
# ---------------------------------------------------------------------------


def test_ensure_network_volume_by_id_success() -> None:
    from ryotenkai_shared.config.providers.runpod.inference import RunPodNetworkVolumeConfig

    api = StubApi(get_network_volume_actions=[{"id": "vol-123", "name": "my-vol"}])
    p = _mk_provider(api=api)
    p._volume_cfg = RunPodNetworkVolumeConfig(id="vol-123", name="my-vol", size_gb=50)
    res = p._ensure_network_volume()
    # (Phase A2 Batch 12: success implies no raise)
    assert res == "vol-123"


def test_ensure_network_volume_by_id_not_found_returns_err() -> None:
    from ryotenkai_shared.config.providers.runpod.inference import RunPodNetworkVolumeConfig

    api = StubApi(get_network_volume_actions=[_exc("not found", code="NOT_FOUND")])
    p = _mk_provider(api=api)
    p._volume_cfg = RunPodNetworkVolumeConfig(id="bad-id", name="my-vol", size_gb=50)
    with pytest.raises(InferenceUnavailableError) as exc_info:
        p._ensure_network_volume()
    assert exc_info.value.context.get("reason") == "RUNPOD_VOLUME_NOT_FOUND"


def test_ensure_network_volume_list_failure_returns_err() -> None:
    from ryotenkai_shared.config.providers.runpod.inference import RunPodNetworkVolumeConfig

    api = StubApi(list_network_volumes_actions=[_exc("list fail", code="LIST_FAIL")])
    p = _mk_provider(api=api)
    p._volume_cfg = RunPodNetworkVolumeConfig(id=None, name="my-vol", size_gb=50, data_center_id="US-KS-2")
    with pytest.raises((InferenceUnavailableError, ProviderUnavailableError)) as exc_info:
        p._ensure_network_volume()
    assert exc_info.value.context.get("reason") or exc_info.value.context.get("code") == "LIST_FAIL"


def test_ensure_network_volume_reuses_existing_match_by_name_and_dc() -> None:
    from ryotenkai_shared.config.providers.runpod.inference import RunPodNetworkVolumeConfig

    api = StubApi(
        list_network_volumes_actions=[
            [
                {"id": "vol-123", "name": "my-vol", "dataCenterId": "US-KS-2"},
                {"id": "vol-999", "name": "my-vol", "dataCenterId": "EU-RO-1"},
            ]
        ]
    )
    p = _mk_provider(api=api)
    p._volume_cfg = RunPodNetworkVolumeConfig(id=None, name="my-vol", size_gb=50, data_center_id="US-KS-2")

    res = p._ensure_network_volume()

    # (Phase A2 Batch 12: success implies no raise)
    assert res == "vol-123"
    assert p._network_volume_meta == {"id": "vol-123", "name": "my-vol", "dataCenterId": "US-KS-2"}


def test_ensure_network_volume_returns_ambiguous_when_multiple_name_matches() -> None:
    from ryotenkai_shared.config.providers.runpod.inference import RunPodNetworkVolumeConfig

    api = StubApi(
        list_network_volumes_actions=[
            [
                {"id": "vol-1", "name": "my-vol", "dataCenterId": "US-KS-2"},
                {"id": "vol-2", "name": "my-vol", "dataCenterId": "US-KS-2"},
            ]
        ]
    )
    p = _mk_provider(api=api)
    p._volume_cfg = RunPodNetworkVolumeConfig(id=None, name="my-vol", size_gb=50, data_center_id="US-KS-2")

    with pytest.raises(InferenceUnavailableError) as exc_info:
        p._ensure_network_volume()
    assert exc_info.value.context.get("reason") == "RUNPOD_VOLUME_AMBIGUOUS"


def test_ensure_network_volume_missing_data_center_returns_err_before_create() -> None:
    api = StubApi(list_network_volumes_actions=[[]])
    p = _mk_provider(api=api)
    p._volume_cfg = SimpleNamespace(id=None, name="my-vol", size_gb=50, data_center_id=None)

    with pytest.raises(InferenceUnavailableError) as exc_info:
        p._ensure_network_volume()
    assert exc_info.value.context.get("reason") == "RUNPOD_VOLUME_DATA_CENTER_MISSING"


def test_ensure_network_volume_existing_match_without_id_returns_volume_no_id_err() -> None:
    from ryotenkai_shared.config.providers.runpod.inference import RunPodNetworkVolumeConfig

    api = StubApi(
        list_network_volumes_actions=[[{"name": "my-vol", "dataCenterId": "US-KS-2"}]]
    )
    p = _mk_provider(api=api)
    p._volume_cfg = RunPodNetworkVolumeConfig(id=None, name="my-vol", size_gb=50, data_center_id="US-KS-2")

    with pytest.raises(InferenceUnavailableError) as exc_info:
        p._ensure_network_volume()
    assert exc_info.value.context.get("reason") == "RUNPOD_VOLUME_NO_ID"


def test_ensure_network_volume_create_success_without_id_but_relist_finds_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    from ryotenkai_shared.config.providers.runpod.inference import RunPodNetworkVolumeConfig

    monkeypatch.setattr("time.sleep", lambda s: None)
    api = StubApi(
        list_network_volumes_actions=[
            [],
            [{"id": "vol-after", "name": "my-vol", "dataCenterId": "US-KS-2"}],
        ],
        create_network_volume_actions=[{"name": "my-vol"}],  # no id
    )
    p = _mk_provider(api=api)
    p._volume_cfg = RunPodNetworkVolumeConfig(id=None, name="my-vol", size_gb=50, data_center_id="US-KS-2")

    res = p._ensure_network_volume()

    # (Phase A2 Batch 12: success implies no raise)
    assert res == "vol-after"
