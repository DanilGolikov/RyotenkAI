"""Phase 4A — Provider-agnostic ``IDockerClient`` Protocol.

Extracted additively in 2026-05-12. The legacy module-level functions
in :mod:`ryotenkai_shared.utils.docker` (``docker_logs``,
``docker_rm_force``, ``docker_is_container_running``,
``docker_container_exit_code``, ``ensure_docker_image``,
``docker_image_exists``) remain as backwards-compat shims; the
concrete :class:`LocalDockerClient` (in :mod:`.local`) delegates to
them so production behaviour is unchanged.

Why a Protocol? — the SUT (e.g.
:class:`ryotenkai_providers.single_node.inference.provider.SingleNodeInferenceProvider`)
historically reached into a sibling module and tests patched
``patch.object(_mod, "docker_logs", ...)``. That coupled tests to
module-level globals and made the SUT untestable without
monkey-patching imports. With ``IDockerClient`` the SUT takes an
optional ``docker: IDockerClient | None = None`` kwarg, defaults to a
shared :class:`LocalDockerClient`, and tests inject a
:class:`tests._fakes.docker.FakeDockerClient` with deterministic
in-memory container state.

Phase A2 Batch 4 (2026-05-14): Methods that returned
``Result[T, ProviderError]`` now return ``T`` directly and raise typed
exceptions on failure (``ProviderUnavailableError`` for transient
docker daemon / pull / inspect failures; ``ConfigInvalidError`` for
caller-side validation failures like a malformed container name). The
two bool-returning methods (``image_exists``, ``is_container_running``)
keep their plain-``bool`` contract — a missing image / stopped
container is not a backend error.

Connection model: the Protocol takes an "exec client" (anything with
the ``exec_command(command, *, timeout, silent) -> (ok, stdout,
stderr)`` shape — typed via :class:`_ExecClient` below) and assumes
that client is already connected. Closing / reconnecting is the
caller's responsibility.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class _ExecClient(Protocol):
    """Structural type for the SSH exec surface ``IDockerClient`` needs.

    Kept narrow on purpose: a Docker call only needs to run one shell
    command on the remote host. The full :class:`ISSHClient` Protocol
    (in :mod:`ryotenkai_shared.infrastructure.ssh`) is a richer
    async-only surface; the legacy ``SSHClient`` is sync. Both satisfy
    this minimal shape, so the Protocol stays compatible with the
    production ``SSHClient`` without bringing async into the docker
    call graph.
    """

    def exec_command(
        self,
        command: str,
        background: bool = False,
        timeout: int = 30,
        silent: bool = False,
    ) -> tuple[bool, str, str]: ...


@runtime_checkable
class IDockerClient(Protocol):
    """Container + image operations over an SSH-reachable Docker host.

    Concrete production impl: :class:`LocalDockerClient`. Test impl:
    :class:`tests._fakes.docker.FakeDockerClient`.

    Failure-mode contract (Phase A2 Batch 4):

    * ``ensure_image``, ``rm_force``, ``logs``, ``container_exit_code`` —
      raise :class:`ProviderUnavailableError` (PROVIDER_UNAVAILABLE, 503)
      on docker-daemon / SSH-exec failure; ``rm_force`` / ``logs`` /
      ``container_exit_code`` raise :class:`ConfigInvalidError` for
      invalid container-name input.
    * ``image_exists`` / ``is_container_running`` — return plain
      ``bool`` (no exceptions for "absent" since that's a normal,
      expected outcome, not an error).
    """

    def image_exists(self, ssh: _ExecClient, image: str) -> bool:
        """Return whether ``image`` is present in the remote Docker daemon."""
        ...

    def ensure_image(
        self,
        *,
        ssh: _ExecClient,
        image: str,
        pull_timeout_seconds: int = 1200,
        verify_after_pull: bool = True,
    ) -> None:
        """Pull ``image`` if missing (or always if tag is ``:latest``).

        Idempotent: returns silently when the image is already present
        and not ``:latest``. ``verify_after_pull=False`` skips the
        post-pull inspect retries (useful in tests).

        Raises:
            ProviderUnavailableError: pull failed or post-pull verify
                failed across all retries.
        """
        ...

    def rm_force(
        self,
        ssh: _ExecClient,
        *,
        container_name: str,
        timeout_seconds: int = 60,
    ) -> None:
        """Force-remove ``container_name`` if it exists. Idempotent.

        Raises:
            ConfigInvalidError: invalid container name.
            ProviderUnavailableError: ``docker rm -f`` exec failed.
        """
        ...

    def is_container_running(
        self,
        ssh: _ExecClient,
        *,
        name_filter: str,
        timeout_seconds: int = 5,
    ) -> bool:
        """Return whether a container matching ``name_filter`` is in state ``running``."""
        ...

    def logs(
        self,
        ssh: _ExecClient,
        *,
        container_name: str,
        tail: int | None = None,
        timeout_seconds: int = 30,
    ) -> str:
        """Fetch container logs (best-effort).

        Raises:
            ConfigInvalidError: invalid container name.
            ProviderUnavailableError: ``docker logs`` exec failed.
        """
        ...

    def container_exit_code(
        self,
        ssh: _ExecClient,
        *,
        container_name: str,
        timeout_seconds: int = 5,
    ) -> int:
        """Get container exit code via ``docker inspect``.

        Raises:
            ConfigInvalidError: invalid container name.
            ProviderUnavailableError: ``docker inspect`` exec failed or
                stdout was not an integer.
        """
        ...


__all__ = ["IDockerClient", "_ExecClient"]
