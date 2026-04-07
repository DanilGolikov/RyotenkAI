"""
Unit tests for RunPodPodsRESTClient (src/providers/runpod/inference/pods/api_client.py).

Coverage:
- _request_json: success, HTTP error, request exception, empty body, non-JSON body
- list_pods: success, empty, unexpected type, HTTP error
- get_pod: success, unexpected type, HTTP error
- create_pod: success, API error, unexpected type
- start_pod: success, error
- stop_pod: success, error
- delete_pod: success, not found, request exception
- list_network_volumes: success, unexpected type, HTTP error
- get_network_volume: success, unexpected type, HTTP error
- create_network_volume: success, error, unexpected type
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.providers.runpod.inference.pods.api_client import RunPodPodsRESTClient
from src.utils.result import Err, Ok
from src.utils.result import ProviderError

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
    list_result: Any = None
    get_result: Any = None
    create_result: Any = None
    start_result: Any = None
    stop_result: Any = None
    delete_result: Any = None

    def list_pods(self, *, params: dict[str, Any] | None = None):
        return self.list_result

    def get_pod(self, *, pod_id: str):
        return self.get_result

    def create_pod_from_payload(self, *, payload: dict[str, Any]):
        return self.create_result

    def start_pod(self, *, pod_id: str):
        return self.start_result

    def stop_pod(self, *, pod_id: str):
        return self.stop_result

    def delete_pod(self, *, pod_id: str):
        return self.delete_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client() -> RunPodPodsRESTClient:
    return RunPodPodsRESTClient(api_key="test-key", api_base_url="https://rest.runpod.io/v1")


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
    res = c._request_json("GET", "/test")
    assert res.is_success()
    assert res.unwrap() == {"key": "val"}


def test_request_json_http_error_returns_err(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _err_resp(500))
    res = c._request_json("GET", "/test")
    assert res.is_failure()
    err = res.unwrap_err()
    assert err.code == "RUNPOD_REST_HTTP_ERROR"
    assert err.details["status_code"] == 500


def test_request_json_request_exception_returns_err(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()

    def _raise(**kw: Any) -> None:
        raise ConnectionError("network down")

    monkeypatch.setattr(c.session, "request", _raise)
    res = c._request_json("GET", "/test")
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_REST_REQUEST_FAILED"


def test_request_json_empty_body_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _FakeResp(status_code=200, _text="", _json_data=None))
    res = c._request_json("DELETE", "/pods/p1")
    assert res.is_success()
    assert res.unwrap() is None


def test_request_json_non_json_body_returns_raw_text(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(
        c.session,
        "request",
        lambda **kw: _FakeResp(status_code=200, _text="plain text", _raise_on_json=True),
    )
    res = c._request_json("GET", "/test")
    assert res.is_success()
    assert res.unwrap() == "plain text"


def test_request_json_http_error_includes_method_and_url(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _err_resp(403, "forbidden"))
    res = c._request_json("POST", "/pods")
    assert res.is_failure()
    err = res.unwrap_err()
    assert err.details["method"] == "POST"


def test_request_json_passes_headers_params_payload_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    captured: dict[str, Any] = {}

    def fake_request(**kw: Any) -> _FakeResp:
        captured.update(kw)
        return _ok_resp({"ok": True})

    monkeypatch.setattr(c.session, "request", fake_request)

    res = c._request_json(
        "POST",
        "/networkvolumes",
        params={"page": 2},
        payload={"name": "vol"},
        timeout_seconds=17,
    )

    assert res.is_success()
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
    monkeypatch.setattr(c, "_sdk", _FakeSDK(list_result=Ok(pods)))
    res = c.list_pods()
    assert res.is_success()
    assert res.unwrap() == pods


def test_list_pods_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c, "_sdk", _FakeSDK(list_result=Ok([])))
    res = c.list_pods()
    assert res.is_success()
    assert res.unwrap() == []


def test_list_pods_unexpected_dict_type_returns_err(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(
        c,
        "_sdk",
        _FakeSDK(
            list_result=Err(
                ProviderError(message="Unexpected runpod SDK get_pods response type: dict", code="RUNPOD_SDK_UNEXPECTED_RESPONSE")
            )
        ),
    )
    res = c.list_pods()
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_SDK_UNEXPECTED_RESPONSE"


def test_list_pods_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(
        c,
        "_sdk",
        _FakeSDK(list_result=Err(ProviderError(message="unauthorized", code="RUNPOD_SDK_CALL_FAILED"))),
    )
    res = c.list_pods()
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_SDK_CALL_FAILED"


def test_list_pods_passes_params(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    captured: dict[str, Any] = {}

    class _CaptureSDK(_FakeSDK):
        def list_pods(self, *, params: dict[str, Any] | None = None):
            captured["params"] = params
            return Ok([])

    monkeypatch.setattr(c, "_sdk", _CaptureSDK())
    c.list_pods(params={"computeType": "GPU", "name": "test-pod"})
    assert captured.get("params") == {"computeType": "GPU", "name": "test-pod"}


def test_create_pod_passes_payload_to_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    captured: dict[str, Any] = {}

    class _CaptureSDK(_FakeSDK):
        def create_pod_from_payload(self, *, payload: dict[str, Any]):
            captured["payload"] = payload
            return Ok({"id": "p1"})

    monkeypatch.setattr(c, "_sdk", _CaptureSDK())

    res = c.create_pod(payload={"name": "pod", "imageName": "img"})

    assert res.is_success()
    assert captured["payload"] == {"name": "pod", "imageName": "img"}


# ---------------------------------------------------------------------------
# get_pod
# ---------------------------------------------------------------------------


def test_get_pod_success(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    pod = {"id": "p1", "desiredStatus": "RUNNING"}
    monkeypatch.setattr(c, "_sdk", _FakeSDK(get_result=Ok(pod)))
    res = c.get_pod(pod_id="p1")
    assert res.is_success()
    assert res.unwrap()["id"] == "p1"


def test_get_pod_unexpected_list_type(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(
        c,
        "_sdk",
        _FakeSDK(
            get_result=Err(
                ProviderError(message="Unexpected runpod SDK get_pod response type: list", code="RUNPOD_SDK_UNEXPECTED_RESPONSE")
            )
        ),
    )
    res = c.get_pod(pod_id="p1")
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_SDK_UNEXPECTED_RESPONSE"


def test_get_pod_http_error_404(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(
        c,
        "_sdk",
        _FakeSDK(get_result=Err(ProviderError(message="not found", code="RUNPOD_SDK_CALL_FAILED"))),
    )
    res = c.get_pod(pod_id="missing")
    assert res.is_failure()


def test_get_pod_request_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(
        c,
        "_sdk",
        _FakeSDK(get_result=Err(ProviderError(message="timeout", code="RUNPOD_SDK_CALL_FAILED"))),
    )
    res = c.get_pod(pod_id="p1")
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_SDK_CALL_FAILED"


# ---------------------------------------------------------------------------
# create_pod
# ---------------------------------------------------------------------------


def test_create_pod_success_returns_pod_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    pod = {"id": "new_pod", "desiredStatus": "RUNNING", "imageName": "img"}
    monkeypatch.setattr(c, "_sdk", _FakeSDK(create_result=Ok(pod)))
    res = c.create_pod(payload={"name": "test", "imageName": "img"})
    assert res.is_success()
    assert res.unwrap()["id"] == "new_pod"


def test_create_pod_api_error_returns_err(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(
        c,
        "_sdk",
        _FakeSDK(create_result=Err(ProviderError(message="bad request", code="RUNPOD_SDK_CALL_FAILED"))),
    )
    res = c.create_pod(payload={})
    assert res.is_failure()


def test_create_pod_unexpected_list_type(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(
        c,
        "_sdk",
        _FakeSDK(
            create_result=Err(
                ProviderError(message="Unexpected runpod SDK create_pod response type: list", code="RUNPOD_SDK_UNEXPECTED_RESPONSE")
            )
        ),
    )
    res = c.create_pod(payload={})
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_SDK_UNEXPECTED_RESPONSE"


def test_create_pod_request_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(
        c,
        "_sdk",
        _FakeSDK(create_result=Err(ProviderError(message="connection reset", code="RUNPOD_SDK_CALL_FAILED"))),
    )
    res = c.create_pod(payload={})
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_SDK_CALL_FAILED"


# ---------------------------------------------------------------------------
# start_pod
# ---------------------------------------------------------------------------


def test_start_pod_success(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c, "_sdk", _FakeSDK(start_result=Ok(None)))
    res = c.start_pod(pod_id="p1")
    assert res.is_success()
    assert res.unwrap() is None


def test_start_pod_error_returns_err(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(
        c,
        "_sdk",
        _FakeSDK(start_result=Err(ProviderError(message="boom", code="RUNPOD_SDK_CALL_FAILED"))),
    )
    res = c.start_pod(pod_id="p1")
    assert res.is_failure()


def test_start_pod_passes_pod_id_to_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    captured: dict[str, Any] = {}

    class _CaptureSDK(_FakeSDK):
        def start_pod(self, *, pod_id: str):
            captured["pod_id"] = pod_id
            return Ok(None)

    monkeypatch.setattr(c, "_sdk", _CaptureSDK())

    res = c.start_pod(pod_id="p9")

    assert res.is_success()
    assert captured["pod_id"] == "p9"


# ---------------------------------------------------------------------------
# stop_pod
# ---------------------------------------------------------------------------


def test_stop_pod_success(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c, "_sdk", _FakeSDK(stop_result=Ok(None)))
    res = c.stop_pod(pod_id="p1")
    assert res.is_success()
    assert res.unwrap() is None


def test_stop_pod_error_returns_err(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(
        c,
        "_sdk",
        _FakeSDK(stop_result=Err(ProviderError(message="boom", code="RUNPOD_SDK_CALL_FAILED"))),
    )
    res = c.stop_pod(pod_id="p1")
    assert res.is_failure()


def test_stop_pod_request_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(
        c,
        "_sdk",
        _FakeSDK(stop_result=Err(ProviderError(message="net down", code="RUNPOD_SDK_CALL_FAILED"))),
    )
    res = c.stop_pod(pod_id="p1")
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_SDK_CALL_FAILED"


# ---------------------------------------------------------------------------
# delete_pod
# ---------------------------------------------------------------------------


def test_delete_pod_success(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c, "_sdk", _FakeSDK(delete_result=Ok(None)))
    res = c.delete_pod(pod_id="p1")
    assert res.is_success()
    assert res.unwrap() is None


def test_delete_pod_not_found_returns_err(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(
        c,
        "_sdk",
        _FakeSDK(delete_result=Err(ProviderError(message="not found", code="RUNPOD_SDK_CALL_FAILED"))),
    )
    res = c.delete_pod(pod_id="missing")
    assert res.is_failure()


def test_delete_pod_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(
        c,
        "_sdk",
        _FakeSDK(delete_result=Err(ProviderError(message="server error", code="RUNPOD_SDK_CALL_FAILED"))),
    )
    res = c.delete_pod(pod_id="p1")
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_SDK_CALL_FAILED"


def test_delete_pod_request_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(
        c,
        "_sdk",
        _FakeSDK(delete_result=Err(ProviderError(message="timeout", code="RUNPOD_SDK_CALL_FAILED"))),
    )
    res = c.delete_pod(pod_id="p1")
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_SDK_CALL_FAILED"


# ---------------------------------------------------------------------------
# list_network_volumes
# ---------------------------------------------------------------------------


def test_list_network_volumes_success(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    volumes = [{"id": "v1", "name": "my-vol"}, {"id": "v2", "name": "other-vol"}]
    monkeypatch.setattr(c.session, "request", lambda **kw: _ok_resp(volumes))
    res = c.list_network_volumes()
    assert res.is_success()
    assert res.unwrap() == volumes


def test_list_network_volumes_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _ok_resp([]))
    res = c.list_network_volumes()
    assert res.is_success()
    assert res.unwrap() == []


def test_list_network_volumes_unexpected_dict_type(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _ok_resp({"volumes": []}))
    res = c.list_network_volumes()
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_UNEXPECTED_RESPONSE"


def test_list_network_volumes_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _err_resp(403, "forbidden"))
    res = c.list_network_volumes()
    assert res.is_failure()


def test_list_network_volumes_calls_expected_rest_path(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    captured: dict[str, Any] = {}

    def fake_request(**kw: Any) -> _FakeResp:
        captured.update(kw)
        return _ok_resp([])

    monkeypatch.setattr(c.session, "request", fake_request)

    res = c.list_network_volumes()

    assert res.is_success()
    assert captured["method"] == "GET"
    assert captured["url"].endswith("/networkvolumes")


# ---------------------------------------------------------------------------
# get_network_volume
# ---------------------------------------------------------------------------


def test_get_network_volume_success(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    vol = {"id": "v1", "name": "my-vol", "size": 50}
    monkeypatch.setattr(c.session, "request", lambda **kw: _ok_resp(vol))
    res = c.get_network_volume(network_volume_id="v1")
    assert res.is_success()
    assert res.unwrap()["id"] == "v1"


def test_get_network_volume_unexpected_list_type(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _ok_resp([{"id": "v1"}]))
    res = c.get_network_volume(network_volume_id="v1")
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_UNEXPECTED_RESPONSE"


def test_get_network_volume_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _err_resp(404, "not found"))
    res = c.get_network_volume(network_volume_id="missing")
    assert res.is_failure()


def test_get_network_volume_calls_expected_rest_path(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    captured: dict[str, Any] = {}

    def fake_request(**kw: Any) -> _FakeResp:
        captured.update(kw)
        return _ok_resp({"id": "v1"})

    monkeypatch.setattr(c.session, "request", fake_request)

    res = c.get_network_volume(network_volume_id="v1")

    assert res.is_success()
    assert captured["method"] == "GET"
    assert captured["url"].endswith("/networkvolumes/v1")


# ---------------------------------------------------------------------------
# create_network_volume
# ---------------------------------------------------------------------------


def test_create_network_volume_success(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    vol = {"id": "v_new", "name": "my-vol", "size": 50, "dataCenterId": "US-KS-2"}
    monkeypatch.setattr(c.session, "request", lambda **kw: _ok_resp(vol))
    res = c.create_network_volume(payload={"name": "my-vol", "size": 50, "dataCenterId": "US-KS-2"})
    assert res.is_success()
    assert res.unwrap()["id"] == "v_new"


def test_create_network_volume_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _err_resp(500))
    res = c.create_network_volume(payload={})
    assert res.is_failure()


def test_create_network_volume_unexpected_list_type(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    monkeypatch.setattr(c.session, "request", lambda **kw: _ok_resp([{"id": "v"}]))
    res = c.create_network_volume(payload={})
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_UNEXPECTED_RESPONSE"


def test_create_network_volume_request_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()

    def _raise(**kw: Any) -> None:
        raise ConnectionError("net down")

    monkeypatch.setattr(c.session, "request", _raise)
    res = c.create_network_volume(payload={})
    assert res.is_failure()
    assert res.unwrap_err().code == "RUNPOD_REST_REQUEST_FAILED"


def test_create_network_volume_calls_expected_rest_path_and_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client()
    captured: dict[str, Any] = {}

    def fake_request(**kw: Any) -> _FakeResp:
        captured.update(kw)
        return _ok_resp({"id": "v1"})

    monkeypatch.setattr(c.session, "request", fake_request)

    payload = {"name": "my-vol", "size": 50, "dataCenterId": "US-KS-2"}
    res = c.create_network_volume(payload=payload)

    assert res.is_success()
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/networkvolumes")
    assert captured["json"] == payload
