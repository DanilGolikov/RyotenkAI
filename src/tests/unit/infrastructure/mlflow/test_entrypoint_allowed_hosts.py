"""
Regression test for docker/mlflow/entrypoint.mlflow.sh — allowed-hosts behaviour.

Rules:
- Without MLFLOW_SERVER_ALLOWED_HOSTS → no --allowed-hosts flag (MLflow accepts all)
- With MLFLOW_SERVER_ALLOWED_HOSTS → --allowed-hosts = localhost invariant + external hosts
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ENTRYPOINT = Path(__file__).resolve().parents[5] / "docker" / "mlflow" / "entrypoint.mlflow.sh"

LOCAL_HOSTS_EXPECTED = {"localhost", "localhost:5000", "127.0.0.1", "127.0.0.1:5000"}


def _run_entrypoint_dry(
    env_allowed_hosts: str | None = None,
    host_port: str | None = "5002",
) -> str:
    """
    Run the entrypoint with 'echo' replacing 'exec' to capture
    the command that WOULD be executed, without starting mlflow.
    """
    env = {
        "BACKEND_STORE_URI": "postgresql://test:test@localhost/test",
        "ARTIFACTS_DESTINATION": "s3://test",
        "PATH": "/usr/bin:/bin",
    }
    if env_allowed_hosts is not None:
        env["MLFLOW_SERVER_ALLOWED_HOSTS"] = env_allowed_hosts
    if host_port is not None:
        env["MLFLOW_HOST_PORT"] = host_port

    script = ENTRYPOINT.read_text().replace("exec ", "echo ")

    result = subprocess.run(
        ["/bin/sh", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=5,
    )
    assert result.returncode == 0, f"entrypoint failed: {result.stderr}"
    return result.stdout.strip()


def _extract_allowed_hosts(cmd_output: str) -> set[str]:
    parts = cmd_output.split()
    for i, part in enumerate(parts):
        if part == "--allowed-hosts" and i + 1 < len(parts):
            return set(parts[i + 1].split(","))
    return set()


def _has_allowed_hosts_flag(cmd_output: str) -> bool:
    return "--allowed-hosts" in cmd_output


@pytest.mark.skipif(not ENTRYPOINT.exists(), reason="entrypoint.mlflow.sh not found")
class TestEntrypointAllowedHosts:

    def test_no_allowed_hosts_when_env_unset(self) -> None:
        """Without MLFLOW_SERVER_ALLOWED_HOSTS, --allowed-hosts must NOT appear."""
        cmd = _run_entrypoint_dry(None)
        assert not _has_allowed_hosts_flag(cmd), f"Unexpected --allowed-hosts in: {cmd}"

    def test_no_allowed_hosts_when_env_empty(self) -> None:
        """Empty MLFLOW_SERVER_ALLOWED_HOSTS → no --allowed-hosts flag."""
        cmd = _run_entrypoint_dry("")
        assert not _has_allowed_hosts_flag(cmd), f"Unexpected --allowed-hosts in: {cmd}"

    def test_localhost_invariant_when_external_hosts_set(self) -> None:
        """With external hosts, localhost entries are always prepended."""
        cmd = _run_entrypoint_dry("my-node.example.ts.net")
        hosts = _extract_allowed_hosts(cmd)
        assert LOCAL_HOSTS_EXPECTED.issubset(hosts), f"Missing local hosts in: {hosts}"
        assert "my-node.example.ts.net" in hosts

    def test_host_mapped_port_included(self) -> None:
        """Host-mapped port (e.g. 5002) must appear so Host: localhost:5002 is accepted."""
        cmd = _run_entrypoint_dry("ext.example.com", host_port="5002")
        hosts = _extract_allowed_hosts(cmd)
        assert "localhost:5002" in hosts, f"Missing host-mapped port in: {hosts}"
        assert "127.0.0.1:5002" in hosts, f"Missing host-mapped port in: {hosts}"

    def test_multiple_external_hosts_preserved(self) -> None:
        cmd = _run_entrypoint_dry("a.example.com,b.example.com:8443")
        hosts = _extract_allowed_hosts(cmd)
        assert "a.example.com" in hosts
        assert "b.example.com:8443" in hosts
        assert LOCAL_HOSTS_EXPECTED.issubset(hosts)
