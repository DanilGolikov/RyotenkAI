"""Normalized domain models and shared utilities for RunPod integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SSH_PORT_CONTAINER = 22


@dataclass(frozen=True, slots=True)
class SshEndpoint:
    """Automation-grade SSH endpoint (exposed TCP, not gateway)."""

    host: str
    port: int


@dataclass(frozen=True, slots=True)
class PodSnapshot:
    """Backend-agnostic, typed representation of a RunPod pod's observable state."""

    pod_id: str
    status: str | None
    uptime_seconds: int
    ssh_endpoint: SshEndpoint | None
    port_count: int

    @property
    def is_ready(self) -> bool:
        return self.status == "RUNNING" and self.ssh_endpoint is not None

    @property
    def is_terminal(self) -> bool:
        return self.status in ("FAILED", "TERMINATED", "EXITED")

    @classmethod
    def from_graphql(cls, pod_data: dict[str, Any]) -> PodSnapshot:
        """Parse a GraphQL pod query response into a typed snapshot."""
        pod_id = str(pod_data.get("id") or "")
        status = pod_data.get("desiredStatus")
        runtime = pod_data.get("runtime") or {}
        uptime = int(runtime.get("uptimeInSeconds") or 0)
        ports = runtime.get("ports") or []
        if not isinstance(ports, list):
            ports = []

        return cls(
            pod_id=pod_id,
            status=status,
            uptime_seconds=uptime,
            ssh_endpoint=_extract_ssh_endpoint(ports),
            port_count=len(ports),
        )

    @classmethod
    def from_runpodctl(cls, payload: dict[str, Any]) -> PodSnapshot:
        """Parse a runpodctl JSON response into a typed snapshot.

        Handles nested wrappers (``{"pod": {...}}``, ``{"data": {"pod": {...}}}``)
        and the non-standard port mapping formats that runpodctl may return.
        """
        source = payload
        if isinstance(payload.get("pod"), dict):
            source = payload["pod"]
        elif isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("pod"), dict):
            source = payload["data"]["pod"]

        pod_id = str(source.get("id") or "")
        status = source.get("desiredStatus") or source.get("status")

        runtime = source.get("runtime")
        if isinstance(runtime, dict) and isinstance(runtime.get("ports"), list):
            uptime = int(runtime.get("uptimeInSeconds") or 0)
            ports_list: list[dict[str, Any]] = runtime["ports"]
            return cls(
                pod_id=pod_id,
                status=status,
                uptime_seconds=uptime,
                ssh_endpoint=_extract_ssh_endpoint(ports_list),
                port_count=len(ports_list),
            )

        public_ip = str(source.get("publicIp") or "").strip()
        raw_ports = source.get("portMappings") or source.get("ports") or []
        normalized = _normalize_runpodctl_ports(raw_ports, public_ip)
        uptime = int(source.get("uptimeInSeconds") or 0)
        return cls(
            pod_id=pod_id,
            status=status,
            uptime_seconds=uptime,
            ssh_endpoint=_extract_ssh_endpoint(normalized),
            port_count=len(normalized),
        )


def _extract_ssh_endpoint(ports: list[dict[str, Any]]) -> SshEndpoint | None:
    """Find the first automation-grade SSH endpoint (exposed TCP on container port 22)."""
    for port in ports:
        if not isinstance(port, dict):
            continue
        if port.get("privatePort") != _SSH_PORT_CONTAINER:
            continue

        is_ip_public = port.get("isIpPublic")
        if isinstance(is_ip_public, bool) and not is_ip_public:
            continue

        host = port.get("ip")
        public_port = port.get("publicPort")
        if isinstance(host, str) and host and isinstance(public_port, int) and public_port > 0:
            return SshEndpoint(host=host, port=public_port)

    return None


def _normalize_runpodctl_ports(raw: Any, public_ip: str) -> list[dict[str, Any]]:
    """Normalize runpodctl port formats into the canonical GraphQL-like shape."""
    ports: list[dict[str, Any]] = []

    if isinstance(raw, dict):
        for key, value in raw.items():
            private_port: int | None = None
            if isinstance(key, int):
                private_port = key
            elif isinstance(key, str):
                digits = "".join(ch for ch in key if ch.isdigit())
                if digits:
                    private_port = int(digits)
            if private_port is None:
                continue

            public_port: int | None = None
            if isinstance(value, dict):
                candidate = value.get("hostPort") or value.get("publicPort") or value.get("port")
                try:
                    public_port = int(candidate)
                except Exception:
                    public_port = None
            else:
                try:
                    public_port = int(value)
                except Exception:
                    public_port = None

            if public_port is not None:
                ports.append(
                    {
                        "ip": public_ip,
                        "privatePort": private_port,
                        "publicPort": public_port,
                        "isIpPublic": bool(public_ip),
                    }
                )

    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            cport = (
                item.get("containerPort") or item.get("internalPort") or item.get("privatePort") or item.get("port")
            )
            hport = item.get("hostPort") or item.get("externalPort") or item.get("publicPort")
            try:
                ports.append(
                    {
                        "ip": str(item.get("ip") or public_ip or "").strip(),
                        "privatePort": int(cport),
                        "publicPort": int(hport),
                        "isIpPublic": True,
                    }
                )
            except Exception:
                continue

    return ports


@dataclass(frozen=True, slots=True)
class PodResourceInfo:
    """Metadata returned by the RunPod API at pod creation time."""

    pod_id: str
    machine_id: str | None
    gpu_count: int | None
    cost_per_hr: float | None
    gpu_type: str | None

    @classmethod
    def from_create_response(cls, data: dict[str, Any]) -> PodResourceInfo:
        """Parse the dict returned by ``RunPodAPIClient.create_pod``."""
        cost_raw = data.get("cost_per_hr")
        try:
            cost = float(cost_raw) if cost_raw is not None else None
        except (TypeError, ValueError):
            cost = None

        gpu_count_raw = data.get("gpu_count")
        try:
            gpu_count = int(gpu_count_raw) if gpu_count_raw is not None else None
        except (TypeError, ValueError):
            gpu_count = None

        return cls(
            pod_id=str(data.get("pod_id") or ""),
            machine_id=data.get("machine"),
            gpu_count=gpu_count,
            cost_per_hr=cost,
            gpu_type=data.get("gpu_type"),
        )


def read_ssh_public_key(key_path: str) -> str | None:
    """Read the sibling ``.pub`` file for a given SSH private key path.

    Returns the public key string, or ``None`` if not found / unreadable.
    """
    pub_path = Path(str(key_path) + ".pub").expanduser()
    try:
        if pub_path.exists():
            value = pub_path.read_text(encoding="utf-8").strip()
            return value or None
    except OSError:
        return None
    return None


__all__ = ["PodResourceInfo", "PodSnapshot", "SshEndpoint", "read_ssh_public_key"]
