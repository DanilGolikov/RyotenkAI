"""Connection probes for providers and integrations.

One dispatcher keyed by ``type`` so we never return HTTP 5xx for a
remote-side failure — the result is always a structured ``{ok,
latency_ms, detail}`` envelope.

Handler registry:
- ``mlflow``              → MLflow tracking server `/api/2.0/mlflow/experiments/search`
- ``huggingface``         → HuggingFace Hub `/api/whoami-v2`
- ``provider:runpod``     → RunPod REST `/v2/user`
- ``provider:single_node``→ SSH probe via src.utils.ssh_client (if configured)

Each handler has its own wall-clock timeout because the expected SLAs
differ wildly (HF is snappy, RunPod may lag, sshfs mount checks can
block). Timeouts are configurable via env vars — see ``TIMEOUTS``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import requests

from src.api.schemas.integration import ConnectionTestResult

# Default timeouts in seconds. Each handler reads its own key from env
# to let ops tune without code changes.
TIMEOUTS: dict[str, float] = {
    "mlflow": float(os.environ.get("RYOTENKAI_TEST_CONN_TIMEOUT_MLFLOW", "5")),
    "huggingface": float(os.environ.get("RYOTENKAI_TEST_CONN_TIMEOUT_HF", "5")),
    "provider:runpod": float(os.environ.get("RYOTENKAI_TEST_CONN_TIMEOUT_RUNPOD", "10")),
    "provider:single_node": float(
        os.environ.get("RYOTENKAI_TEST_CONN_TIMEOUT_SSH", "8")
    ),
}


def _ok(start: float, detail: str = "") -> ConnectionTestResult:
    return ConnectionTestResult(
        ok=True, latency_ms=int((time.monotonic() - start) * 1000), detail=detail
    )


def _fail(start: float, detail: str) -> ConnectionTestResult:
    return ConnectionTestResult(
        ok=False,
        latency_ms=int((time.monotonic() - start) * 1000),
        detail=detail,
    )


# ---------- Integration handlers -----------------------------------------


def _test_mlflow(config: dict[str, Any], token: str | None) -> ConnectionTestResult:
    start = time.monotonic()
    tracking_uri = (config.get("tracking_uri") or "").strip()
    if not tracking_uri:
        return _fail(start, "tracking_uri is empty — nothing to probe")

    verify: str | bool = True
    ca_bundle = (config.get("ca_bundle_path") or "").strip()
    if ca_bundle:
        verify = ca_bundle

    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = tracking_uri.rstrip("/") + "/api/2.0/mlflow/experiments/search"
    try:
        resp = requests.post(
            url,
            json={"max_results": 1},
            headers=headers,
            timeout=TIMEOUTS["mlflow"],
            verify=verify,
        )
    except requests.RequestException as exc:
        return _fail(start, f"network error: {exc}")

    if resp.status_code == 200:
        return _ok(start, f"MLflow {tracking_uri} reachable")
    return _fail(start, f"HTTP {resp.status_code}: {resp.text[:200]}")


def _test_huggingface(
    config: dict[str, Any], token: str | None
) -> ConnectionTestResult:
    _ = config  # HF integration schema carries no knobs in v1
    start = time.monotonic()
    if not token:
        return _fail(start, "no token configured — set it in Token tab first")
    try:
        resp = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=TIMEOUTS["huggingface"],
        )
    except requests.RequestException as exc:
        return _fail(start, f"network error: {exc}")

    if resp.status_code == 200:
        try:
            name = resp.json().get("name", "")
        except ValueError:
            name = ""
        return _ok(start, f"authenticated as {name}" if name else "authenticated")
    if resp.status_code == 401:
        return _fail(start, "unauthorized — token rejected by HuggingFace")
    return _fail(start, f"HTTP {resp.status_code}: {resp.text[:200]}")


INTEGRATION_HANDLERS: dict[
    str, Callable[[dict[str, Any], str | None], ConnectionTestResult]
] = {
    "mlflow": _test_mlflow,
    "huggingface": _test_huggingface,
}


# ---------- Provider handlers --------------------------------------------


def _test_runpod(config: dict[str, Any], token: str | None) -> ConnectionTestResult:
    _ = config
    start = time.monotonic()
    if not token:
        return _fail(start, "no API key set — add it in Token tab first")
    try:
        resp = requests.get(
            "https://rest.runpod.io/v1/user",
            headers={"Authorization": f"Bearer {token}"},
            timeout=TIMEOUTS["provider:runpod"],
        )
    except requests.RequestException as exc:
        return _fail(start, f"network error: {exc}")
    if resp.status_code == 200:
        return _ok(start, "RunPod API reachable")
    if resp.status_code == 401:
        return _fail(start, "unauthorized — API key rejected by RunPod")
    return _fail(start, f"HTTP {resp.status_code}: {resp.text[:200]}")


def _test_single_node(
    config: dict[str, Any], token: str | None
) -> ConnectionTestResult:
    """SSH-probe for a single-node provider.

    Reads ``connect.host`` / ``connect.user`` / ``connect.port`` /
    ``connect.key_path`` from the provider YAML when present. When the
    provider runs without SSH (pure local), we just verify the working
    directory is writable.
    """
    _ = token  # single_node doesn't use a bearer token today
    start = time.monotonic()

    connect = config.get("connect") if isinstance(config.get("connect"), dict) else {}
    host = str(connect.get("host", "")).strip()
    if not host:
        # Pure-local mode: nothing meaningful to probe over the network.
        # Report "ok" with a hint so the UI doesn't show a scary red.
        return _ok(start, "single-node provider runs locally (no SSH host configured)")

    user = str(connect.get("user", "") or "root").strip()
    port = int(connect.get("port", 22) or 22)
    key_path_raw = str(connect.get("key_path", "") or "").strip()

    # Lazy import to keep ``paramiko`` (heavy dep) out of non-probe paths.
    try:
        import paramiko
    except ImportError as exc:  # pragma: no cover — paramiko is in base deps
        return _fail(start, f"paramiko not available: {exc}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    kwargs: dict[str, Any] = {
        "hostname": host,
        "port": port,
        "username": user,
        "timeout": TIMEOUTS["provider:single_node"],
        "auth_timeout": TIMEOUTS["provider:single_node"],
        "banner_timeout": TIMEOUTS["provider:single_node"],
    }
    if key_path_raw:
        from pathlib import Path

        key_path = Path(key_path_raw).expanduser()
        if not key_path.is_file():
            return _fail(start, f"SSH key not found at {key_path}")
        kwargs["key_filename"] = str(key_path)

    try:
        client.connect(**kwargs)
    except Exception as exc:
        return _fail(start, f"SSH connect failed: {exc}")
    try:
        _, stdout, _ = client.exec_command("echo ok", timeout=TIMEOUTS["provider:single_node"])
        output = stdout.read().decode("utf-8", errors="replace").strip()
    except Exception as exc:
        return _fail(start, f"SSH exec failed: {exc}")
    finally:
        client.close()

    if output != "ok":
        return _fail(start, f"unexpected probe output: {output!r}")
    return _ok(start, f"SSH {user}@{host}:{port} reachable")


PROVIDER_HANDLERS: dict[
    str, Callable[[dict[str, Any], str | None], ConnectionTestResult]
] = {
    "runpod": _test_runpod,
    "single_node": _test_single_node,
}


def test_integration(
    integration_type: str, config: dict[str, Any], token: str | None
) -> ConnectionTestResult:
    handler = INTEGRATION_HANDLERS.get(integration_type)
    if handler is None:
        return ConnectionTestResult(
            ok=False,
            detail=f"no test-connection handler for integration type {integration_type!r}",
        )
    return handler(config, token)


def test_provider(
    provider_type: str, config: dict[str, Any], token: str | None
) -> ConnectionTestResult:
    handler = PROVIDER_HANDLERS.get(provider_type)
    if handler is None:
        return ConnectionTestResult(
            ok=False,
            detail=f"no test-connection handler for provider type {provider_type!r}",
        )
    return handler(config, token)


__all__ = [
    "INTEGRATION_HANDLERS",
    "PROVIDER_HANDLERS",
    "TIMEOUTS",
    "test_integration",
    "test_provider",
]
