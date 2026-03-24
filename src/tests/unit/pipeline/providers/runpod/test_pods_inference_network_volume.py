from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import ValidationError

from src.config.providers.runpod.inference import RunPodNetworkVolumeConfig
from src.providers.runpod.inference.pods.provider import RunPodPodInferenceProvider
from src.utils.result import Err, Ok, ProviderError, Result

pytestmark = pytest.mark.unit


@dataclass
class StubPodsApi:
    list_results: list[Result[list[dict[str, Any]], ProviderError]] = field(default_factory=list)
    create_results: list[Result[dict[str, Any], ProviderError]] = field(default_factory=list)
    created_payloads: list[dict[str, Any]] = field(default_factory=list)

    def get_network_volume(self, *, network_volume_id: str) -> Result[dict[str, Any], ProviderError]:
        _ = network_volume_id
        return Err(ProviderError(message="not implemented in stub", code="STUB_NOT_IMPLEMENTED"))

    def list_network_volumes(self) -> Result[list[dict[str, Any]], ProviderError]:
        if self.list_results:
            return self.list_results.pop(0)
        return Ok([])  # type: ignore[call-arg]

    def create_network_volume(self, *, payload: dict[str, Any]) -> Result[dict[str, Any], ProviderError]:
        self.created_payloads.append(payload)
        if self.create_results:
            return self.create_results.pop(0)
        return Ok(  # type: ignore[call-arg]
            {
                "id": "nv_created",
                "name": payload.get("name"),
                "size": payload.get("size"),
                "dataCenterId": payload.get("dataCenterId"),
            }
        )


def _mk_provider(*, api: StubPodsApi, volume_cfg: RunPodNetworkVolumeConfig) -> RunPodPodInferenceProvider:
    # We intentionally bypass __init__ here to unit-test the internal volume logic
    # without building a full PipelineConfig.
    p = RunPodPodInferenceProvider.__new__(RunPodPodInferenceProvider)
    p._api = api
    p._volume_cfg = volume_cfg
    p._network_volume_meta = None
    return p


def test_network_volume_config_requires_datacenter_when_id_missing() -> None:
    with pytest.raises(ValidationError, match=r"RunPod network volume config requires either"):
        _ = RunPodNetworkVolumeConfig(id=None, name="vol", size_gb=50)


def test_network_volume_autocreate_uses_config_datacenter() -> None:
    api = StubPodsApi(list_results=[Ok([])])  # type: ignore[call-arg]
    cfg = RunPodNetworkVolumeConfig(id=None, name="vol", data_center_id="EU-RO-1", size_gb=50)
    p = _mk_provider(api=api, volume_cfg=cfg)

    res = p._ensure_network_volume()
    assert res.is_success()
    assert api.created_payloads[0]["dataCenterId"] == "EU-RO-1"


def test_network_volume_create_failure_but_volume_appears_after_relist() -> None:
    api = StubPodsApi(
        list_results=[
            Ok([]),  # type: ignore[call-arg]
            Ok(  # type: ignore[call-arg]
                [{"id": "nv_found", "name": "vol", "size": 50, "dataCenterId": "US-KS-2"}]
            ),
        ],
        create_results=[Err("RunPod REST HTTP 500 for POST /networkvolumes: ...")],
    )
    cfg = RunPodNetworkVolumeConfig(id=None, name="vol", data_center_id="US-KS-2", size_gb=50)
    p = _mk_provider(api=api, volume_cfg=cfg)

    res = p._ensure_network_volume()
    assert res.is_success()
    assert res.unwrap() == "nv_found"
    assert isinstance(p._network_volume_meta, dict)
    assert p._network_volume_meta.get("id") == "nv_found"


def test_network_volume_multiple_matches_by_name_requires_id() -> None:
    api = StubPodsApi(
        list_results=[
            Ok(  # type: ignore[call-arg]
                [
                    {"id": "nv_1", "name": "vol", "dataCenterId": "EU-RO-1"},
                    {"id": "nv_2", "name": "vol", "dataCenterId": "EU-RO-1"},
                ]
            )
        ]
    )
    cfg = RunPodNetworkVolumeConfig(id=None, name="vol", data_center_id="EU-RO-1", size_gb=50)
    p = _mk_provider(api=api, volume_cfg=cfg)

    res = p._ensure_network_volume()
    assert res.is_failure()
    msg = str(res.unwrap_err())
    assert "multiple network volumes" in msg
    assert "providers.runpod.inference.volume.id" in msg


def test_network_volume_multiple_matches_can_disambiguate_by_config_datacenter() -> None:
    api = StubPodsApi(
        list_results=[
            Ok(  # type: ignore[call-arg]
                [
                    {"id": "nv_eu", "name": "vol", "dataCenterId": "EU-RO-1"},
                    {"id": "nv_us", "name": "vol", "dataCenterId": "US-KS-2"},
                ]
            )
        ]
    )
    cfg = RunPodNetworkVolumeConfig(id=None, name="vol", data_center_id="EU-RO-1", size_gb=50)
    p = _mk_provider(api=api, volume_cfg=cfg)

    res = p._ensure_network_volume()
    assert res.is_success()
    assert res.unwrap() == "nv_eu"

