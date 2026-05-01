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
        """Parse a RunPod pod-query response into a typed snapshot.

        Despite the historical name, this factory handles both shapes of
        pod payload that the RunPod backend returns:

        * ``runtime.ports`` — canonical list of port dicts emitted by the
          Python SDK's GraphQL-backed ``runpod.get_pod()``.
        * ``portMappings`` + ``publicIp`` — flat shortcut emitted by the
          REST endpoint and (defensively) by some SDK versions where
          ``runtime.ports`` is empty during early pod startup.

        Both are read from the same input dict; ``runtime.ports`` takes
        precedence when present and populated. See ``_normalize_ports``
        for the per-shape parsing rules.
        """
        pod_id = str(pod_data.get("id") or "")
        status = pod_data.get("desiredStatus")
        runtime = pod_data.get("runtime") or {}
        uptime = int(runtime.get("uptimeInSeconds") or 0)

        ssh_endpoint, port_count = _normalize_ports(pod_data)

        return cls(
            pod_id=pod_id,
            status=status,
            uptime_seconds=uptime,
            ssh_endpoint=ssh_endpoint,
            port_count=port_count,
        )


def _normalize_ports(pod_data: dict[str, Any]) -> tuple[SshEndpoint | None, int]:
    """Extract ``(ssh_endpoint, port_count)`` from any RunPod pod payload shape.

    Order of precedence:

    1. ``runtime.ports`` — list of ``{ip, privatePort, publicPort, isIpPublic}``.
       Canonical for ``runpod.get_pod()`` responses.
    2. ``portMappings`` + ``publicIp`` — REST/flat fallback. Multiple
       observed sub-shapes for ``portMappings``:

       * ``{"22": 23828}``
       * ``{"22/tcp": "23828"}``
       * ``{"22": {"hostPort": 23828}}``
       * ``[{"containerPort": 22, "hostPort": 23828}, ...]``

    Returns ``(None, port_count)`` if no automation-grade SSH endpoint can
    be reconstructed; ``port_count`` is best-effort across shapes (length
    of the underlying collection).
    """
    runtime = pod_data.get("runtime") or {}
    ports = runtime.get("ports")
    if isinstance(ports, list) and ports:
        return _extract_ssh_from_runtime_ports(ports), len(ports)

    mappings = pod_data.get("portMappings")
    public_ip = str(pod_data.get("publicIp") or "").strip()
    ssh_port = _extract_ssh_port_from_mappings(mappings)
    port_count = _count_mappings(mappings)
    if public_ip and ssh_port:
        return SshEndpoint(host=public_ip, port=ssh_port), port_count
    return None, port_count


def _extract_ssh_from_runtime_ports(ports: list[Any]) -> SshEndpoint | None:
    """Find the first automation-grade SSH endpoint in a ``runtime.ports`` list.

    Automation-grade = exposed TCP on container port 22 with a public IP.
    """
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


def _extract_ssh_port_from_mappings(mappings: Any) -> int | None:
    """Find the SSH host-port in a ``portMappings`` payload.

    ``portMappings`` can be a dict (key=container port, value=host port,
    possibly nested in an object) or a list (entries with explicit
    ``containerPort`` / ``hostPort`` keys). Both shapes are observed in
    the wild — handle them defensively rather than asserting on one.
    """
    if isinstance(mappings, dict):
        for key in ("22", 22, "22/tcp", "tcp/22"):
            if key not in mappings:
                continue
            raw = mappings.get(key)
            port = _coerce_port(raw)
            if port is not None:
                return port
        return None

    if isinstance(mappings, list):
        for entry in mappings:
            if not isinstance(entry, dict):
                continue
            cport = entry.get("containerPort") or entry.get("internalPort") or entry.get("port")
            hport = entry.get("hostPort") or entry.get("externalPort") or entry.get("publicPort")
            if cport is None or hport is None:
                continue
            try:
                if int(cport) != _SSH_PORT_CONTAINER:
                    continue
                port = int(hport)
                if port > 0:
                    return port
            except (TypeError, ValueError):
                continue
        return None

    return None


def _coerce_port(raw: Any) -> int | None:
    """Coerce a port value from any of the observed shapes to a positive int.

    Handles bare ints, numeric strings, and the nested
    ``{"hostPort": N}`` / ``{"publicPort": N}`` / ``{"port": N}`` dict.
    """
    if isinstance(raw, dict):
        raw = raw.get("hostPort") or raw.get("publicPort") or raw.get("port")
    if isinstance(raw, bool):  # bool is subclass of int — exclude explicitly
        return None
    if isinstance(raw, int) and raw > 0:
        return raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.isdigit():
            value = int(stripped)
            return value if value > 0 else None
    return None


def _count_mappings(mappings: Any) -> int:
    """Length of the underlying ``portMappings`` collection (0 if unknown)."""
    if isinstance(mappings, dict | list):
        return len(mappings)
    return 0


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
