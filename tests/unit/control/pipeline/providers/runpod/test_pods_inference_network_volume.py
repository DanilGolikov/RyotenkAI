from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import ValidationError

from ryotenkai_shared.config.providers.runpod.inference import RunPodNetworkVolumeConfig
from ryotenkai_providers.runpod.inference.pods.provider import RunPodPodInferenceProvider
from ryotenkai_shared.errors import InferenceUnavailableError, ProviderUnavailableError

pytestmark = pytest.mark.unit


@dataclass
class StubPodsApi:
    """Raise-based stub for the inference REST client.

    Each queued item is ``list|dict|None`` (returned) or ``BaseException``
    (raised). Empty queues fall through to a sensible default.
    """

    list_actions: list[Any] = field(default_factory=list)
    create_actions: list[Any] = field(default_factory=list)
    created_payloads: list[dict[str, Any]] = field(default_factory=list)

    def get_network_volume(self, *, network_volume_id: str) -> dict[str, Any]:
        _ = network_volume_id
        raise ProviderUnavailableError(
            detail="not implemented in stub",
            context={"code": "STUB_NOT_IMPLEMENTED"},
        )

    def list_network_volumes(self) -> list[dict[str, Any]]:
        if not self.list_actions:
            return []
        action = self.list_actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action

    def create_network_volume(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        self.created_payloads.append(payload)
        if not self.create_actions:
            return {
                "id": "nv_created",
                "name": payload.get("name"),
                "size": payload.get("size"),
                "dataCenterId": payload.get("dataCenterId"),
            }
        action = self.create_actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action


def _mk_provider(*, api: StubPodsApi, volume_cfg: RunPodNetworkVolumeConfig) -> RunPodPodInferenceProvider:
    p = RunPodPodInferenceProvider.__new__(RunPodPodInferenceProvider)
    p._api = api
    p._volume_cfg = volume_cfg
    p._network_volume_meta = None
    return p


def test_network_volume_config_requires_datacenter_when_id_missing() -> None:
    with pytest.raises(ValidationError, match=r"RunPod network volume config requires either"):
        _ = RunPodNetworkVolumeConfig(id=None, name="vol", size_gb=50)


def test_network_volume_autocreate_uses_config_datacenter() -> None:
    api = StubPodsApi(list_actions=[[]])
    cfg = RunPodNetworkVolumeConfig(id=None, name="vol", data_center_id="EU-RO-1", size_gb=50)
    p = _mk_provider(api=api, volume_cfg=cfg)

    # Phase A2 Batch 12: _ensure_network_volume returns id string on success, raises on failure.
    vol_id = p._ensure_network_volume()
    assert isinstance(vol_id, str) and vol_id
    assert api.created_payloads[0]["dataCenterId"] == "EU-RO-1"


def test_network_volume_create_failure_but_volume_appears_after_relist() -> None:
    api = StubPodsApi(
        list_actions=[
            [],
            [{"id": "nv_found", "name": "vol", "size": 50, "dataCenterId": "US-KS-2"}],
        ],
        create_actions=[
            ProviderUnavailableError(
                detail="RunPod REST HTTP 500 for POST /networkvolumes: ...",
                context={"code": "RUNPOD_REST_HTTP_ERROR"},
            )
        ],
    )
    cfg = RunPodNetworkVolumeConfig(id=None, name="vol", data_center_id="US-KS-2", size_gb=50)
    p = _mk_provider(api=api, volume_cfg=cfg)

    vol_id = p._ensure_network_volume()
    assert vol_id == "nv_found"
    assert isinstance(p._network_volume_meta, dict)
    assert p._network_volume_meta.get("id") == "nv_found"


def test_network_volume_multiple_matches_by_name_requires_id() -> None:
    api = StubPodsApi(
        list_actions=[
            [
                {"id": "nv_1", "name": "vol", "dataCenterId": "EU-RO-1"},
                {"id": "nv_2", "name": "vol", "dataCenterId": "EU-RO-1"},
            ]
        ]
    )
    cfg = RunPodNetworkVolumeConfig(id=None, name="vol", data_center_id="EU-RO-1", size_gb=50)
    p = _mk_provider(api=api, volume_cfg=cfg)

    with pytest.raises(InferenceUnavailableError) as exc_info:
        p._ensure_network_volume()
    msg = str(exc_info.value.detail or exc_info.value)
    assert "multiple network volumes" in msg
    assert "providers.runpod.inference.volume.id" in msg


def test_network_volume_multiple_matches_can_disambiguate_by_config_datacenter() -> None:
    api = StubPodsApi(
        list_actions=[
            [
                {"id": "nv_eu", "name": "vol", "dataCenterId": "EU-RO-1"},
                {"id": "nv_us", "name": "vol", "dataCenterId": "US-KS-2"},
            ]
        ]
    )
    cfg = RunPodNetworkVolumeConfig(id=None, name="vol", data_center_id="EU-RO-1", size_gb=50)
    p = _mk_provider(api=api, volume_cfg=cfg)

    vol_id = p._ensure_network_volume()
    assert vol_id == "nv_eu"
