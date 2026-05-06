"""``LaunchSpec`` → docker shell command formatting.

Engines build a structured :class:`ryotenkai_engines.LaunchSpec` describing
*what* to run (image, args, env, port, volumes). Providers know *how* to
run things in their own runtime (Docker, Kubernetes, systemd, …). This
module contains the docker-specific formatter used by SSH-based providers.

A future Kubernetes provider would replace this layer with a translator
to ``ContainerSpec`` / ``PodSpec`` — engines and configs are unchanged.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ryotenkai_engines.interfaces import LaunchSpec


def format_docker_run(
    spec: LaunchSpec,
    *,
    host_bind: str = "0.0.0.0",
    detach: bool = True,
    gpus_all: bool = True,
) -> str:
    """Format a :class:`LaunchSpec` as a single ``docker run …`` shell command.

    Args:
        spec: Structured launch description from the engine.
        host_bind: Host-side bind address for the published port. Defaults to
            ``"0.0.0.0"``; pass ``"127.0.0.1"`` to restrict.
        detach: Pass ``--detach`` so docker returns immediately. The provider
            polls the healthcheck command separately.
        gpus_all: Pass ``--gpus all``. Set to ``False`` only for CPU-only
            engines once any are added — none today.

    Returns:
        A shell-safe ``docker run …`` command string. Each argument that may
        contain whitespace or shell metacharacters is quoted via :func:`shlex.quote`;
        engine args remain readable when they're plain identifiers.
    """
    parts: list[str] = ["docker run"]
    if detach:
        parts.append("--detach")
    parts.append(f"--name {shlex.quote(spec.container_name)}")
    if gpus_all:
        parts.append("--gpus all")
    parts.append(f"-p {host_bind}:{spec.port}:{spec.port}")
    for host_path, container_path in spec.volumes:
        parts.append(f"-v {shlex.quote(host_path)}:{shlex.quote(container_path)}")
    for key, value in spec.env.items():
        parts.append(f"-e {shlex.quote(f'{key}={value}')}")
    parts.append(shlex.quote(spec.image))
    parts.extend(shlex.quote(arg) for arg in spec.args)
    return " ".join(parts)


__all__ = ["format_docker_run"]
