from __future__ import annotations

import re
import time
from typing import Protocol

from ryotenkai_shared.errors import ConfigInvalidError, ProviderUnavailableError
from ryotenkai_shared.utils.constants import LOG_OUTPUT_LONG_CHARS, LOG_OUTPUT_SHORT_CHARS
from ryotenkai_shared.utils.logger import get_logger

_DOCKER_INSPECT_TIMEOUT = 20  # seconds — conservative for slow disks
_DOCKER_LOGS_PULL_CHARS = 300  # docker logs fetch truncation
_CONTAINER_NAME_KEY = "container_name"

logger = get_logger("docker")


class _ExecClient(Protocol):
    def exec_command(
        self,
        command: str,
        background: bool = False,
        timeout: int = 30,
        silent: bool = False,
    ) -> tuple[bool, str, str]: ...


_SAFE_CONTAINER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")


def _validate_container_name(name: str) -> None:
    """Validate a container name; raise :class:`ConfigInvalidError` on failure.

    Phase A2 Batch 4 migration: previously returned ``Result[None, ProviderError]``
    with code ``"DOCKER_INVALID_CONTAINER_NAME"``; now raises a typed
    :class:`ConfigInvalidError` since this is a caller-side validation
    failure (4xx semantics — the caller passed a bad name).
    """
    if not name or not isinstance(name, str):
        raise ConfigInvalidError(
            detail="Container name must be a non-empty string",
            context={"reason": "DOCKER_INVALID_CONTAINER_NAME", _CONTAINER_NAME_KEY: name},
        )
    if not _SAFE_CONTAINER_NAME_RE.match(name):
        raise ConfigInvalidError(
            detail=f"Unsafe Docker container name: {name!r}",
            context={"reason": "DOCKER_INVALID_CONTAINER_NAME", _CONTAINER_NAME_KEY: name},
        )


def docker_image_exists(ssh: _ExecClient, image: str) -> bool:
    """
    Check if Docker image exists locally on the remote host.

    NOTE: Avoid single quotes in commands — SSHClient wraps remote command in single quotes.
    """
    # NOTE: inspect can be slow on busy/slow disks; keep timeout conservative but not tiny.
    ok, _stdout, _stderr = ssh.exec_command(
        f"docker image inspect {image} >/dev/null 2>&1",
        timeout=_DOCKER_INSPECT_TIMEOUT,
        silent=True,
    )
    return bool(ok)


def _is_latest_tag(image: str) -> bool:
    """
    Best-effort detect whether the image reference points to the "latest" tag.

    Rules:
    - If a digest is used (contains '@'), this is NOT treated as latest.
    - If no explicit tag is present, Docker defaults to :latest -> treated as latest.
    - If an explicit tag is present and equals 'latest' -> treated as latest.
    """
    if not isinstance(image, str) or not image:
        return False
    if "@" in image:
        return False

    last_slash = image.rfind("/")
    last_colon = image.rfind(":")

    # Tag separator is ':' after the last '/' (port separator is before the last '/').
    has_tag = last_colon > last_slash
    if not has_tag:
        return True  # implicit :latest

    tag = image[last_colon + 1 :]
    return tag == "latest"


def ensure_docker_image(
    *,
    ssh: _ExecClient,
    image: str,
    pull_timeout_seconds: int = 1200,
    verify_after_pull: bool = True,
) -> None:
    """
    Ensure Docker image is available on the remote host.

    Fixed behavior (docker-only, no user-configured pull policy):
    - If the image tag is 'latest' (or tag is omitted -> implicit latest): always pull to refresh.
    - Otherwise: pull only if the image is missing locally.

    Raises:
        ProviderUnavailableError: pull command exited non-zero, or
            post-pull verify failed across all retries. ``context``
            carries ``reason`` (legacy code: ``DOCKER_PULL_FAILED`` /
            ``DOCKER_IMAGE_NOT_AVAILABLE``) and ``image``.
    """
    latest = _is_latest_tag(image)
    logger.debug(f"[DOCKER] ensure image='{image}' (latest={latest})")
    exists = docker_image_exists(ssh, image)
    logger.debug(f"[DOCKER] image exists locally: {exists}")

    if exists and not latest:
        logger.info(f"[DOCKER] Image already present, skipping pull: {image}")
        return

    if latest:
        logger.info(f"[DOCKER] Image tag is 'latest' -> pulling to refresh: {image}")
    else:
        logger.info(f"[DOCKER] Image missing -> pulling: {image}")

    ok, stdout, stderr = ssh.exec_command(
        f"docker pull {image}",
        timeout=int(pull_timeout_seconds),
        silent=False,
    )
    if not ok:
        details = (stderr or stdout or "").strip()[:LOG_OUTPUT_LONG_CHARS]
        logger.error(f"[DOCKER] Failed to pull image: {image}. Details: {details}")
        raise ProviderUnavailableError(
            detail=f"Failed to pull image '{image}': {details}",
            context={"reason": "DOCKER_PULL_FAILED", "image": image},
        )

    if not verify_after_pull:
        logger.info(f"[DOCKER] Pull completed (no post-verify): {image}")
        return

    # Docker registry may take a moment to finalize.
    # Also, docker daemon may still be registering layers; retry a few times to avoid false negatives.
    for attempt in range(1, 6):
        time.sleep(2)
        if docker_image_exists(ssh, image):
            logger.info(f"[DOCKER] Image ready: {image}")
            return
        logger.debug(f"[DOCKER] Post-pull inspect retry {attempt}/5 failed: {image}")

    logger.error(f"[DOCKER] Image not available after pull: {image}")
    raise ProviderUnavailableError(
        detail=(
            f"Image '{image}' was pulled but is not available in Docker registry. "
            "This may indicate a network timeout or Docker daemon issue."
        ),
        context={"reason": "DOCKER_IMAGE_NOT_AVAILABLE", "image": image},
    )


def docker_rm_force(ssh: _ExecClient, *, container_name: str, timeout_seconds: int = 60) -> None:
    """Force remove a container if it exists (idempotent).

    Raises:
        ConfigInvalidError: ``container_name`` failed validation.
        ProviderUnavailableError: ``docker rm -f`` exited non-zero
            (``context["reason"] == "DOCKER_RM_FAILED"``).
    """
    _validate_container_name(container_name)

    ok, _stdout, stderr = ssh.exec_command(
        f"docker rm -f {container_name} >/dev/null 2>&1 || true",
        timeout=int(timeout_seconds),
        silent=True,
    )
    if not ok:
        details = (stderr or "").strip()[:LOG_OUTPUT_SHORT_CHARS] or "docker rm failed"
        raise ProviderUnavailableError(
            detail=details,
            context={"reason": "DOCKER_RM_FAILED", _CONTAINER_NAME_KEY: container_name},
        )


def docker_is_container_running(
    ssh: _ExecClient,
    *,
    name_filter: str,
    timeout_seconds: int = 5,
) -> bool:
    """
    Check if there is a running container matching name filter.

    Uses docker native filters (no grep) to avoid quoting issues.
    """
    # name filter is not necessarily a valid container name (can be substring),
    # but we still keep it conservative to avoid shell injection.
    if not isinstance(name_filter, str) or not name_filter or any(c.isspace() for c in name_filter):
        return False

    ok, stdout, _stderr = ssh.exec_command(
        f"docker ps -q -f name={name_filter} -f status=running",
        timeout=int(timeout_seconds),
        silent=True,
    )
    return bool(ok and stdout.strip())


def docker_logs(
    ssh: _ExecClient,
    *,
    container_name: str,
    tail: int | None = None,
    timeout_seconds: int = 30,
) -> str:
    """Get container logs (best-effort).

    Raises:
        ConfigInvalidError: invalid ``container_name``.
        ProviderUnavailableError: ``docker logs`` exited non-zero
            (``context["reason"] == "DOCKER_LOGS_FAILED"``).
    """
    _validate_container_name(container_name)

    tail_part = f" --tail {int(tail)}" if isinstance(tail, int) and tail > 0 else ""
    ok, stdout, stderr = ssh.exec_command(
        f"docker logs {container_name}{tail_part} 2>&1",
        timeout=int(timeout_seconds),
        silent=True,
    )
    if not ok:
        details = (stderr or stdout or "").strip()[:_DOCKER_LOGS_PULL_CHARS] or "docker logs failed"
        raise ProviderUnavailableError(
            detail=details,
            context={"reason": "DOCKER_LOGS_FAILED", _CONTAINER_NAME_KEY: container_name},
        )
    return stdout


def docker_container_exit_code(
    ssh: _ExecClient, *, container_name: str, timeout_seconds: int = 5
) -> int:
    """Get container exit code via `docker inspect`.

    Raises:
        ConfigInvalidError: invalid ``container_name``.
        ProviderUnavailableError: ``docker inspect`` exited non-zero, or
            stdout could not be parsed as an integer. ``context["reason"]``
            is ``"DOCKER_INSPECT_FAILED"`` or ``"DOCKER_INSPECT_INVALID_OUTPUT"``.
    """
    _validate_container_name(container_name)

    # IMPORTANT: do NOT use single quotes here (SSHClient wraps command in single quotes).
    cmd = f'docker inspect {container_name} --format "{{{{.State.ExitCode}}}}"'
    ok, stdout, stderr = ssh.exec_command(cmd, timeout=int(timeout_seconds), silent=True)
    if not ok:
        details = (stderr or "").strip()[:LOG_OUTPUT_SHORT_CHARS] or "docker inspect failed"
        raise ProviderUnavailableError(
            detail=details,
            context={"reason": "DOCKER_INSPECT_FAILED", _CONTAINER_NAME_KEY: container_name},
        )

    try:
        return int(stdout.strip())
    except ValueError as exc:
        raise ProviderUnavailableError(
            detail=f"Invalid exit code output: {stdout.strip()!r}",
            context={
                "reason": "DOCKER_INSPECT_INVALID_OUTPUT",
                _CONTAINER_NAME_KEY: container_name,
                "raw_output": stdout.strip(),
            },
            cause=exc,
        ) from exc


__all__ = [
    "docker_container_exit_code",
    "docker_image_exists",
    "docker_is_container_running",
    "docker_logs",
    "docker_rm_force",
    "ensure_docker_image",
]
