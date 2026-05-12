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

The interface mirrors the legacy function surface 1:1 — same kwargs,
same ``Result[..., ProviderError]`` returns — to keep the migration
mechanical. The Protocol takes an SSH client per call (rather than
binding it at construction) because the production provider creates
the SSH session lazily and rebinds across retries; binding inside the
docker client would force a circular dependency on the SSH
lifecycle.

Connection model: the Protocol takes an "exec client" (anything with
the ``exec_command(command, *, timeout, silent) -> (ok, stdout,
stderr)`` shape — typed via :class:`_ExecClient` below) and assumes
that client is already connected. Closing / reconnecting is the
caller's responsibility.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ryotenkai_shared.utils.result import ProviderError, Result


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

    All methods return :class:`Result` rather than raising — failures
    flow as ``Err(ProviderError)`` so the provider can decide between
    fail-fast and best-effort cleanup paths without try/except clutter.
    The two non-``Result`` methods (``image_exists``,
    ``is_container_running``) return plain ``bool`` — a missing image
    or stopped container is not a backend error, it's just "no".
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
    ) -> Result[None, ProviderError]:
        """Pull ``image`` if missing (or always if tag is ``:latest``).

        Idempotent: returns ``Ok(None)`` when the image is already
        present and not ``:latest``. ``verify_after_pull=False`` skips
        the post-pull inspect retries (useful in tests).
        """
        ...

    def rm_force(
        self,
        ssh: _ExecClient,
        *,
        container_name: str,
        timeout_seconds: int = 60,
    ) -> Result[None, ProviderError]:
        """Force-remove ``container_name`` if it exists. Idempotent."""
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
    ) -> Result[str, ProviderError]:
        """Fetch container logs (best-effort)."""
        ...

    def container_exit_code(
        self,
        ssh: _ExecClient,
        *,
        container_name: str,
        timeout_seconds: int = 5,
    ) -> Result[int, ProviderError]:
        """Get container exit code via ``docker inspect``."""
        ...


__all__ = ["IDockerClient", "_ExecClient"]
