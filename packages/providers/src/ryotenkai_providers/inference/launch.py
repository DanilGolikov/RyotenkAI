"""``LaunchSpec`` / ``PrepareStep`` → docker shell command formatting.

Engines build structured specs describing *what* to run; providers know
*how* to run things in their own runtime (Docker, Kubernetes, systemd, …).
This module is the docker-specific formatter — two siblings:

  * :func:`format_docker_run` — long-running inference server (``LaunchSpec``)
  * :func:`format_prepare_step` — one-shot preparation container (``PrepareStep``)

The two specs differ enough that polymorphism would obscure the
intent (port publishing vs none; default entrypoint vs override).
A future Kubernetes provider replaces this module with a ``ContainerSpec``
/ ``PodSpec`` translator — engines and configs are unchanged.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ryotenkai_engines.interfaces import LaunchSpec, PrepareStep


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


def format_prepare_step(
    step: PrepareStep,
    *,
    image: str,
    container_name: str,
    extra_env: Mapping[str, str] | None = None,
    gpus_all: bool = True,
) -> str:
    """Format a :class:`PrepareStep` as a single ``docker run …`` command.

    Sibling of :func:`format_docker_run` for ephemeral one-shot containers.
    Differences vs :func:`format_docker_run`:

      * No ``-p host:port:port`` — prepare steps don't expose ports.
      * ``--detach`` is always set so the provider can poll logs and
        collect exit code; provider does ``docker rm -f`` after.
      * ``--entrypoint`` is set when the engine overrides it (e.g.
        vLLM merge step uses ``python3`` to run ``merge_lora.py`` inside
        the same image whose default ENTRYPOINT is the inference server).
      * ``extra_env`` is overlay-merged on top of ``step.env``; on key
        collision provider wins. This is the secrets-injection boundary
        (``HF_TOKEN`` flows through here, not from the engine).

    Args:
        step: Engine-described preparation step.
        image: Resolved image (``step.image`` or the engine's serve image
            when ``step.image is None`` — provider does the fall-through).
        container_name: Provider-chosen container name (e.g.
            ``f"helix-prepare-{run_id}-{step.name}"``).
        extra_env: Provider-injected env (HF_TOKEN, etc.). Merged on top
            of ``step.env``; provider keys override engine keys.
        gpus_all: Pass ``--gpus all``. Set ``False`` only for CPU-only prep.

    Returns:
        A shell-safe ``docker run --detach …`` command. Every shell-meta
        value (image, container name, paths, env values, args) is quoted
        via :func:`shlex.quote`.
    """
    parts: list[str] = ["docker run", "--detach"]
    parts.append(f"--name {shlex.quote(container_name)}")
    if gpus_all:
        parts.append("--gpus all")
    for host_path, container_path in step.volumes:
        parts.append(f"-v {shlex.quote(host_path)}:{shlex.quote(container_path)}")
    # Engine env first, then provider extra_env — extra_env wins on collision.
    merged_env: dict[str, str] = {**step.env, **(dict(extra_env) if extra_env else {})}
    for key, value in merged_env.items():
        parts.append(f"-e {shlex.quote(f'{key}={value}')}")
    if step.entrypoint:
        # Docker --entrypoint takes a single token; if the engine's tuple has
        # more than one element, the rest is prepended to args. (Today's vLLM
        # merge uses ``("python3",)`` only; tuple is forward-compat.)
        head, *rest = step.entrypoint
        parts.append(f"--entrypoint {shlex.quote(head)}")
        parts.append(shlex.quote(image))
        parts.extend(shlex.quote(a) for a in rest)
        parts.extend(shlex.quote(a) for a in step.args)
    else:
        parts.append(shlex.quote(image))
        parts.extend(shlex.quote(a) for a in step.args)
    return " ".join(parts)


__all__ = ["format_docker_run", "format_prepare_step"]
