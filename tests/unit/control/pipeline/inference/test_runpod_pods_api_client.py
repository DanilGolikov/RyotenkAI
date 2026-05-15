"""Unit tests for RunPodPodsRESTClient (raise-based, Phase A2 Batch 11).

Coverage:
- _request_json: success, HTTP error, request exception, empty body, non-JSON body
- list_pods: success, empty, unexpected type, error
- get_pod: success, unexpected type, error
- create_pod: success, error, unexpected type
- start_pod / stop_pod / delete_pod: success, error
- list_network_volumes / get_network_volume / create_network_volume: REST success/error
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ryotenkai_providers.runpod.inference.pods.api_client import RunPodPodsRESTClient
from ryotenkai_shared.errors import ProviderUnavailableError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fake HTTP response
# ---------------------------------------------------------------------------


@dataclass
class _FakeResp:
    status_code: int = 200
    _text: str = ""
    _json_data: Any = None
    _raise_on_json: bool = False

    @property
    def text(self) -> str:
        return self._text

    def json(self) -> Any:
        if self._raise_on_json:
            raise ValueError("invalid json")
        return self._json_data


def _ok_resp(data: Any) -> _FakeResp:
    return _FakeResp(status_code=200, _text=str(data), _json_data=data)


def _err_resp(status: int = 500, body: Any = "server error") -> _FakeResp:
    return _FakeResp(status_code=status, _text=str(body), _json_data=body)


@dataclass
class _FakeSDK:
    """Raise-based fake for ``RunPodSDKClient``."""

    list_value: Any = None
    list_exc: BaseException | None = None
    get_value: Any = None
    get_exc: BaseException | None = None
    create_value: Any = None
    create_exc: BaseException | None = None
    start_exc: BaseException | None = None
    stop_exc: BaseException | None = None
    delete_exc: BaseException | None = None
    captured: dict[str, Any] = field(default_factory=dict)

    def list_pods(self, *, params: dict[str, Any] | None = None):
        self.captured["list_params"] = params
        if self.list_exc is not None:
            raise self.list_exc
        return self.list_value

    def get_pod(self, *, pod_id: str):
        self.captured["get_pod_id"] = pod_id
        if self.get_exc is not None:
            raise self.get_exc
        return self.get_value

    def create_pod_from_payload(self, *, payload: dict[str, Any]):
        self.captured["create_payload"] = payload
        if self.create_exc is not None:
            raise self.create_exc
        return self.create_value

    def start_pod(self, *, pod_id: str):
        self.captured["start_pod_id"] = pod_id
        if self.start_exc is not None:
            raise self.start_exc

    def stop_pod(self, *, pod_id: str):
        self.captured["stop_pod_id"] = pod_id
        if self.stop_exc is not None:
            raise self.stop_exc

    def delete_pod(self, *, pod_id: str):
        self.captured["delete_pod_id"] = pod_id
        if self.delete_exc is not None:
            raise self.delete_exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client() -> RunPodPodsRESTClient:
    return RunPodPodsRESTClient(api_key="test-key", api_base_url="https://rest.runpod.io/v1")


def _err(detail: str, code: str = "RUNPOD_SDK_CALL_FAILED") -> ProviderUnavailableError:
    return ProviderUnavailableError(detail=detail, context={"code": code})


def test_client_init_normalizes_base_url_and_auth_header() -> None:
    c = RunPodPodsRESTClient(api_key="secret", api_base_url="https://rest.runpod.io/v1/")

    assert c.api_base == "https://rest.runpod.io/v1"
    assert c.headers["Authorization"] == "Bearer secret"
    assert c.headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# _request_json
# ---------------------------------------------------------------------------


def test_request_json_success_returns_parsed_body(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _ok_resp({"key": "val"}))
    out = c._request_json("GET", "/test")
    assert out == {"key": "val"}


def test_request_json_http_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _err_resp(500))
    with pytest.raises(ProviderUnavailableError) as ei:
        c._request_json("GET", "/test")
    assert ei.value.context["code"] == "RUNPOD_REST_HTTP_ERROR"
    assert ei.value.context["status_code"] == 500


def test_request_json_request_exception_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()

    def _raise(**kw: Any) -> None:
        raise ConnectionError("network down")

    monkeypatch.setattr(c.session, "request", _raise)
    with pytest.raises(ProviderUnavailableError) as ei:
        c._request_json("GET", "/test")
    assert ei.value.context["code"] == "RUNPOD_REST_REQUEST_FAILED"


def test_request_json_empty_body_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _FakeResp(status_code=200, _text="", _json_data=None))
    out = c._request_json("DELETE", "/pods/p1")
    assert out is None


def test_request_json_non_json_body_returns_raw_text(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(
        c.session,
        "request",
        lambda **kw: _FakeResp(status_code=200, _text="plain text", _raise_on_json=True),
    )
    out = c._request_json("GET", "/test")
    assert out == "plain text"


def test_request_json_http_error_includes_method_and_url(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _err_resp(403, "forbidden"))
    with pytest.raises(ProviderUnavailableError) as ei:
        c._request_json("POST", "/pods")
    assert ei.value.context["method"] == "POST"


def test_request_json_passes_headers_params_payload_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    captured: dict[str, Any] = {}

    def fake_request(**kw: Any) -> _FakeResp:
        captured.update(kw)
        return _ok_resp({"ok": True})

    monkeypatch.setattr(c.session, "request", fake_request)

    c._request_json(
        "POST",
        "/networkvolumes",
        params={"page": 2},
        payload={"name": "vol"},
        timeout_seconds=17,
    )

    assert captured["method"] == "POST"
    assert captured["url"] == "https://rest.runpod.io/v1/networkvolumes"
    assert captured["headers"] == c.headers
    assert captured["params"] == {"page": 2}
    assert captured["json"] == {"name": "vol"}
    assert captured["timeout"] == 17


# ---------------------------------------------------------------------------
# list_pods
# ---------------------------------------------------------------------------


def test_list_pods_returns_list(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    pods = [{"id": "p1", "desiredStatus": "RUNNING"}, {"id": "p2", "desiredStatus": "EXITED"}]
    monkeypatch.setattr(c, "_sdk", _FakeSDK(list_value=pods))
    assert c.list_pods() == pods


def test_list_pods_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c, "_sdk", _FakeSDK(list_value=[]))
    assert c.list_pods() == []


def test_list_pods_propagates_sdk_error(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c, "_sdk", _FakeSDK(list_exc=_err("unauthorized")))
    with pytest.raises(ProviderUnavailableError) as ei:
        c.list_pods()
    assert ei.value.context["code"] == "RUNPOD_SDK_CALL_FAILED"


def test_list_pods_passes_params(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    sdk = _FakeSDK(list_value=[])
    monkeypatch.setattr(c, "_sdk", sdk)
    c.list_pods(params={"computeType": "GPU", "name": "test-pod"})
    assert sdk.captured["list_params"] == {"computeType": "GPU", "name": "test-pod"}


def test_create_pod_passes_payload_to_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    sdk = _FakeSDK(create_value={"id": "p1"})
    monkeypatch.setattr(c, "_sdk", sdk)

    out = c.create_pod(payload={"name": "pod", "imageName": "img"})

    assert out["id"] == "p1"
    assert sdk.captured["create_payload"] == {"name": "pod", "imageName": "img"}


# ---------------------------------------------------------------------------
# get_pod
# ---------------------------------------------------------------------------


def test_get_pod_success(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    pod = {"id": "p1", "desiredStatus": "RUNNING"}
    monkeypatch.setattr(c, "_sdk", _FakeSDK(get_value=pod))
    assert c.get_pod(pod_id="p1") == pod


def test_get_pod_propagates_sdk_error(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c, "_sdk", _FakeSDK(get_exc=_err("not found")))
    with pytest.raises(ProviderUnavailableError):
        c.get_pod(pod_id="missing")


# ---------------------------------------------------------------------------
# create_pod
# ---------------------------------------------------------------------------


def test_create_pod_success_returns_pod_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    pod = {"id": "new_pod", "desiredStatus": "RUNNING", "imageName": "img"}
    monkeypatch.setattr(c, "_sdk", _FakeSDK(create_value=pod))
    assert c.create_pod(payload={"name": "test", "imageName": "img"})["id"] == "new_pod"


def test_create_pod_api_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c, "_sdk", _FakeSDK(create_exc=_err("bad request")))
    with pytest.raises(ProviderUnavailableError):
        c.create_pod(payload={})


# ---------------------------------------------------------------------------
# start_pod / stop_pod / delete_pod
# ---------------------------------------------------------------------------


def test_start_pod_success(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    sdk = _FakeSDK()
    monkeypatch.setattr(c, "_sdk", sdk)
    c.start_pod(pod_id="p1")
    assert sdk.captured["start_pod_id"] == "p1"


def test_start_pod_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c, "_sdk", _FakeSDK(start_exc=_err("boom")))
    with pytest.raises(ProviderUnavailableError):
        c.start_pod(pod_id="p1")


def test_stop_pod_success(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    sdk = _FakeSDK()
    monkeypatch.setattr(c, "_sdk", sdk)
    c.stop_pod(pod_id="p1")
    assert sdk.captured["stop_pod_id"] == "p1"


def test_stop_pod_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c, "_sdk", _FakeSDK(stop_exc=_err("boom")))
    with pytest.raises(ProviderUnavailableError):
        c.stop_pod(pod_id="p1")


def test_delete_pod_success(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    sdk = _FakeSDK()
    monkeypatch.setattr(c, "_sdk", sdk)
    c.delete_pod(pod_id="p1")
    assert sdk.captured["delete_pod_id"] == "p1"


def test_delete_pod_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c, "_sdk", _FakeSDK(delete_exc=_err("server error")))
    with pytest.raises(ProviderUnavailableError):
        c.delete_pod(pod_id="p1")


# ---------------------------------------------------------------------------
# list_network_volumes
# ---------------------------------------------------------------------------


def test_list_network_volumes_success(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    volumes = [{"id": "v1", "name": "my-vol"}, {"id": "v2", "name": "other-vol"}]
    monkeypatch.setattr(c.session, "request", lambda **kw: _ok_resp(volumes))
    assert c.list_network_volumes() == volumes


def test_list_network_volumes_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _ok_resp([]))
    assert c.list_network_volumes() == []


def test_list_network_volumes_unexpected_dict_type_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _ok_resp({"volumes": []}))
    with pytest.raises(ProviderUnavailableError) as ei:
        c.list_network_volumes()
    assert ei.value.context["code"] == "RUNPOD_UNEXPECTED_RESPONSE"


def test_list_network_volumes_http_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _err_resp(403, "forbidden"))
    with pytest.raises(ProviderUnavailableError):
        c.list_network_volumes()


def test_list_network_volumes_calls_expected_rest_path(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    captured: dict[str, Any] = {}

    def fake_request(**kw: Any) -> _FakeResp:
        captured.update(kw)
        return _ok_resp([])

    monkeypatch.setattr(c.session, "request", fake_request)

    c.list_network_volumes()

    assert captured["method"] == "GET"
    assert captured["url"].endswith("/networkvolumes")


# ---------------------------------------------------------------------------
# get_network_volume
# ---------------------------------------------------------------------------


def test_get_network_volume_success(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    vol = {"id": "v1", "name": "my-vol", "size": 50}
    monkeypatch.setattr(c.session, "request", lambda **kw: _ok_resp(vol))
    assert c.get_network_volume(network_volume_id="v1")["id"] == "v1"


def test_get_network_volume_unexpected_list_type_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _ok_resp([{"id": "v1"}]))
    with pytest.raises(ProviderUnavailableError) as ei:
        c.get_network_volume(network_volume_id="v1")
    assert ei.value.context["code"] == "RUNPOD_UNEXPECTED_RESPONSE"


def test_get_network_volume_http_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _err_resp(404, "not found"))
    with pytest.raises(ProviderUnavailableError):
        c.get_network_volume(network_volume_id="missing")


def test_get_network_volume_calls_expected_rest_path(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    captured: dict[str, Any] = {}

    def fake_request(**kw: Any) -> _FakeResp:
        captured.update(kw)
        return _ok_resp({"id": "v1"})

    monkeypatch.setattr(c.session, "request", fake_request)

    c.get_network_volume(network_volume_id="v1")

    assert captured["method"] == "GET"
    assert captured["url"].endswith("/networkvolumes/v1")


# ---------------------------------------------------------------------------
# create_network_volume
# ---------------------------------------------------------------------------


def test_create_network_volume_success(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    vol = {"id": "v_new", "name": "my-vol", "size": 50, "dataCenterId": "US-KS-2"}
    monkeypatch.setattr(c.session, "request", lambda **kw: _ok_resp(vol))
    out = c.create_network_volume(payload={"name": "my-vol", "size": 50, "dataCenterId": "US-KS-2"})
    assert out["id"] == "v_new"


def test_create_network_volume_http_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _err_resp(500))
    with pytest.raises(ProviderUnavailableError):
        c.create_network_volume(payload={})


def test_create_network_volume_unexpected_list_type_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _ok_resp([{"id": "v"}]))
    with pytest.raises(ProviderUnavailableError) as ei:
        c.create_network_volume(payload={})
    assert ei.value.context["code"] == "RUNPOD_UNEXPECTED_RESPONSE"


def test_create_network_volume_request_exception_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()

    def _raise(**kw: Any) -> None:
        raise ConnectionError("net down")

    monkeypatch.setattr(c.session, "request", _raise)
    with pytest.raises(ProviderUnavailableError) as ei:
        c.create_network_volume(payload={})
    assert ei.value.context["code"] == "RUNPOD_REST_REQUEST_FAILED"


def test_create_network_volume_calls_expected_rest_path_and_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    captured: dict[str, Any] = {}

    def fake_request(**kw: Any) -> _FakeResp:
        captured.update(kw)
        return _ok_resp({"id": "v1"})

    monkeypatch.setattr(c.session, "request", fake_request)

    payload = {"name": "my-vol", "size": 50, "dataCenterId": "US-KS-2"}
    c.create_network_volume(payload=payload)

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/networkvolumes")
    assert captured["json"] == payload
