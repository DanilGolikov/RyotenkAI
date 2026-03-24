from __future__ import annotations

import re
import time
from typing import Protocol

from src.utils.constants import LOG_OUTPUT_LONG_CHARS, LOG_OUTPUT_SHORT_CHARS
from src.utils.logger import get_logger
from src.utils.result import Err, Ok, ProviderError, Result

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


def _validate_container_name(name: str) -> Result[None, ProviderError]:
    if not name or not isinstance(name, str):
        return Err(
            ProviderError(message="Container name must be a non-empty string", code="DOCKER_INVALID_CONTAINER_NAME")
        )
    if not _SAFE_CONTAINER_NAME_RE.match(name):
        return Err(
            ProviderError(message=f"Unsafe Docker container name: {name!r}", code="DOCKER_INVALID_CONTAINER_NAME")
        )
    return Ok(None)


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
) -> Result[None, ProviderError]:
    """
    Ensure Docker image is available on the remote host.

    Fixed behavior (docker-only, no user-configured pull policy):
    - If the image tag is 'latest' (or tag is omitted -> implicit latest): always pull to refresh.
    - Otherwise: pull only if the image is missing locally.
    """
    latest = _is_latest_tag(image)
    logger.debug(f"[DOCKER] ensure image='{image}' (latest={latest})")
    exists = docker_image_exists(ssh, image)
    logger.debug(f"[DOCKER] image exists locally: {exists}")

    if exists and not latest:
        logger.info(f"[DOCKER] Image already present, skipping pull: {image}")
        return Ok(None)

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
        return Err(
            ProviderError(
                message=f"Failed to pull image '{image}': {details}",
                code="DOCKER_PULL_FAILED",
                details={"image": image},
            )
        )

    if not verify_after_pull:
        logger.info(f"[DOCKER] Pull completed (no post-verify): {image}")
        return Ok(None)

    # Docker registry may take a moment to finalize.
    # Also, docker daemon may still be registering layers; retry a few times to avoid false negatives.
    for attempt in range(1, 6):
        time.sleep(2)
        if docker_image_exists(ssh, image):
            logger.info(f"[DOCKER] Image ready: {image}")
            return Ok(None)
        logger.debug(f"[DOCKER] Post-pull inspect retry {attempt}/5 failed: {image}")

    logger.error(f"[DOCKER] Image not available after pull: {image}")
    return Err(
        ProviderError(
            message=(
                f"Image '{image}' was pulled but is not available in Docker registry. "
                "This may indicate a network timeout or Docker daemon issue."
            ),
            code="DOCKER_IMAGE_NOT_AVAILABLE",
            details={"image": image},
        )
    )


def docker_rm_force(ssh: _ExecClient, *, container_name: str, timeout_seconds: int = 60) -> Result[None, ProviderError]:
    """Force remove a container if it exists (idempotent)."""
    valid = _validate_container_name(container_name)
    if valid.is_failure():
        return Err(valid.unwrap_err())  # type: ignore[union-attr]

    ok, _stdout, stderr = ssh.exec_command(
        f"docker rm -f {container_name} >/dev/null 2>&1 || true",
        timeout=int(timeout_seconds),
        silent=True,
    )
    if not ok:
        details = (stderr or "").strip()[:LOG_OUTPUT_SHORT_CHARS] or "docker rm failed"
        return Err(
            ProviderError(message=details, code="DOCKER_RM_FAILED", details={_CONTAINER_NAME_KEY: container_name})
        )
    return Ok(None)


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
) -> Result[str, ProviderError]:
    """Get container logs (best-effort)."""
    valid = _validate_container_name(container_name)
    if valid.is_failure():
        return Err(valid.unwrap_err())  # type: ignore[union-attr]

    tail_part = f" --tail {int(tail)}" if isinstance(tail, int) and tail > 0 else ""
    ok, stdout, stderr = ssh.exec_command(
        f"docker logs {container_name}{tail_part} 2>&1",
        timeout=int(timeout_seconds),
        silent=True,
    )
    if not ok:
        details = (stderr or stdout or "").strip()[:_DOCKER_LOGS_PULL_CHARS] or "docker logs failed"
        return Err(
            ProviderError(message=details, code="DOCKER_LOGS_FAILED", details={_CONTAINER_NAME_KEY: container_name})
        )
    return Ok(stdout)


def docker_container_exit_code(
    ssh: _ExecClient, *, container_name: str, timeout_seconds: int = 5
) -> Result[int, ProviderError]:
    """Get container exit code via `docker inspect`."""
    valid = _validate_container_name(container_name)
    if valid.is_failure():
        return Err(valid.unwrap_err())  # type: ignore[union-attr]

    # IMPORTANT: do NOT use single quotes here (SSHClient wraps command in single quotes).
    cmd = f'docker inspect {container_name} --format "{{{{.State.ExitCode}}}}"'
    ok, stdout, stderr = ssh.exec_command(cmd, timeout=int(timeout_seconds), silent=True)
    if not ok:
        details = (stderr or "").strip()[:LOG_OUTPUT_SHORT_CHARS] or "docker inspect failed"
        return Err(
            ProviderError(message=details, code="DOCKER_INSPECT_FAILED", details={_CONTAINER_NAME_KEY: container_name})
        )

    try:
        return Ok(int(stdout.strip()))
    except ValueError:
        return Err(
            ProviderError(
                message=f"Invalid exit code output: {stdout.strip()!r}",
                code="DOCKER_INSPECT_INVALID_OUTPUT",
                details={_CONTAINER_NAME_KEY: container_name, "raw_output": stdout.strip()},
            )
        )


__all__ = [
    "docker_container_exit_code",
    "docker_image_exists",
    "docker_is_container_running",
    "docker_logs",
    "docker_rm_force",
    "ensure_docker_image",
]
