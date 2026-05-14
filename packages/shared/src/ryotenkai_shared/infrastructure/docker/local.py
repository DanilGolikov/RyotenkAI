"""Production :class:`IDockerClient` impl — delegates to the legacy
module-level functions in :mod:`ryotenkai_shared.utils.docker`.

This is a thin adapter — no logic changes vs. the legacy call sites.
The reason for keeping it thin: the legacy functions encode subtle
behaviour (container-name validation, ``--tail`` formatting, exit-code
parsing) that's tested via a dedicated test module; reimplementing it
in the class would duplicate the surface and create a divergence risk.

Stateless and safe to share across threads / providers — the SSH
client is passed per call, not stored.

Phase A2 Batch 4 (2026-05-14): both the underlying module-level
functions and these adapter methods now raise typed exceptions
(:class:`ProviderUnavailableError`, :class:`ConfigInvalidError`)
instead of returning :class:`Result`. The class is otherwise unchanged.
"""

from __future__ import annotations

from ryotenkai_shared.infrastructure.docker.protocol import _ExecClient
from ryotenkai_shared.utils.docker import (
    docker_container_exit_code as _docker_container_exit_code,
)
from ryotenkai_shared.utils.docker import (
    docker_image_exists as _docker_image_exists,
)
from ryotenkai_shared.utils.docker import (
    docker_is_container_running as _docker_is_container_running,
)
from ryotenkai_shared.utils.docker import (
    docker_logs as _docker_logs,
)
from ryotenkai_shared.utils.docker import (
    docker_rm_force as _docker_rm_force,
)
from ryotenkai_shared.utils.docker import (
    ensure_docker_image as _ensure_docker_image,
)


class LocalDockerClient:
    """Subprocess/SSH-backed :class:`IDockerClient`.

    Every call delegates to the legacy function of the same shape in
    :mod:`ryotenkai_shared.utils.docker`. The legacy functions remain
    public for callers that haven't migrated yet.
    """

    def image_exists(self, ssh: _ExecClient, image: str) -> bool:
        return _docker_image_exists(ssh, image)

    def ensure_image(
        self,
        *,
        ssh: _ExecClient,
        image: str,
        pull_timeout_seconds: int = 1200,
        verify_after_pull: bool = True,
    ) -> None:
        _ensure_docker_image(
            ssh=ssh,
            image=image,
            pull_timeout_seconds=pull_timeout_seconds,
            verify_after_pull=verify_after_pull,
        )

    def rm_force(
        self,
        ssh: _ExecClient,
        *,
        container_name: str,
        timeout_seconds: int = 60,
    ) -> None:
        _docker_rm_force(
            ssh, container_name=container_name, timeout_seconds=timeout_seconds
        )

    def is_container_running(
        self,
        ssh: _ExecClient,
        *,
        name_filter: str,
        timeout_seconds: int = 5,
    ) -> bool:
        return _docker_is_container_running(
            ssh, name_filter=name_filter, timeout_seconds=timeout_seconds
        )

    def logs(
        self,
        ssh: _ExecClient,
        *,
        container_name: str,
        tail: int | None = None,
        timeout_seconds: int = 30,
    ) -> str:
        return _docker_logs(
            ssh,
            container_name=container_name,
            tail=tail,
            timeout_seconds=timeout_seconds,
        )

    def container_exit_code(
        self,
        ssh: _ExecClient,
        *,
        container_name: str,
        timeout_seconds: int = 5,
    ) -> int:
        return _docker_container_exit_code(
            ssh,
            container_name=container_name,
            timeout_seconds=timeout_seconds,
        )


__all__ = ["LocalDockerClient"]
