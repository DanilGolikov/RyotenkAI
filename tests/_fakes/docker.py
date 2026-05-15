"""``FakeDockerClient`` — canonical fake for :class:`IDockerClient`.

In-memory container + image registry with a small state machine
(``running`` / ``exited`` / ``removed``) and a deterministic logs
queue per container.

Determinism: no time source — log content is exactly what tests
``append`` and stays put until cleared. No background tasks, no
network — every operation is pure data manipulation.

Phase A2 Batch 4 (2026-05-14): contract switched from
``Result[T, ProviderError]`` to plain ``T`` returns with raised
:class:`ProviderUnavailableError` / :class:`ConfigInvalidError` on
failure — mirrors the new production :class:`LocalDockerClient`.

Chaos surface (programming API):

* :meth:`register_container` — pre-create a container with a state
* :meth:`set_container_state` — flip state directly
* :meth:`set_exit_code` — set ``exit_code`` and transition to ``exited``
* :meth:`append_logs` — append log content for the named container
* :meth:`set_image_present` / :meth:`set_image_missing` — image registry
* :meth:`fail_next_n_calls` — count-down failure injection per method
* :meth:`set_pull_behaviour` — control what ``ensure_image`` does on
  the next pull (succeed-and-register / fail / succeed-but-stay-missing)
* :meth:`reset_chaos` — back to clean state

The fake records every call to ``_call_log`` so tests can assert on
ordering / arguments without resorting to ``MagicMock``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ryotenkai_shared.errors import ProviderUnavailableError
from ryotenkai_shared.infrastructure.docker.protocol import IDockerClient, _ExecClient


@dataclass
class _ContainerRecord:
    name: str
    state: str = "running"  # running / exited / removed
    exit_code: int | None = None
    logs: list[str] = field(default_factory=list)


class FakeDockerClient:
    """Deterministic in-memory :class:`IDockerClient`.

    Default behaviour: every method is "yes, sure" — image always
    exists, ``rm_force`` always succeeds, ``logs`` returns ``""``,
    ``is_container_running`` returns ``False`` for unknown containers.

    Tests opt in to specific behaviour by calling the chaos surface
    before the SUT runs. Example::

        fake = FakeDockerClient()
        fake.register_container("helix-prepare-r1-merge", state="exited", exit_code=0)
        fake.append_logs("helix-prepare-r1-merge", "MERGE_SUCCESS")
        provider = SingleNodeInferenceProvider(ctx, docker=fake)
        provider._run_prepare_plan(ssh=..., plan=..., ...)
        # Inspect the call log:
        assert ("rm_force", {"container_name": "helix-prepare-r1-merge", ...}) in fake.calls
    """

    def __init__(self) -> None:
        self._containers: dict[str, _ContainerRecord] = {}
        self._known_images: set[str] = set()
        # By default every image is "present" — tests that care flip
        # this off via set_image_missing.
        self._image_default_present: bool = True
        self._pull_behaviour: str = "register"  # register / fail / silently_missing
        self._fail_counts: dict[str, int] = {}
        self._call_log: list[tuple[str, dict[str, Any]]] = []

    # ------------------------------------------------------------------
    # Programming surface — container registry
    # ------------------------------------------------------------------

    def register_container(
        self,
        name: str,
        *,
        state: str = "running",
        exit_code: int | None = None,
        logs: list[str] | None = None,
    ) -> None:
        self._containers[name] = _ContainerRecord(
            name=name,
            state=state,
            exit_code=exit_code,
            logs=list(logs) if logs else [],
        )

    def set_container_state(self, name: str, state: str) -> None:
        rec = self._containers.setdefault(name, _ContainerRecord(name=name))
        rec.state = state

    def set_exit_code(self, name: str, code: int) -> None:
        rec = self._containers.setdefault(name, _ContainerRecord(name=name))
        rec.exit_code = code
        rec.state = "exited"

    def append_logs(self, name: str, content: str) -> None:
        rec = self._containers.setdefault(name, _ContainerRecord(name=name))
        rec.logs.append(content)

    def set_logs(self, name: str, content: str) -> None:
        """Replace the current log buffer with a single chunk."""
        rec = self._containers.setdefault(name, _ContainerRecord(name=name))
        rec.logs = [content]

    # ------------------------------------------------------------------
    # Programming surface — image registry
    # ------------------------------------------------------------------

    def set_image_present(self, image: str) -> None:
        self._known_images.add(image)

    def set_image_missing(self, image: str) -> None:
        self._known_images.discard(image)
        # Once an image is explicitly marked missing the default
        # "every image is present" answer no longer applies for *this*
        # image. The trick: also flip the global default so a
        # ``set_image_missing`` call without a follow-up
        # ``set_image_present`` actually reports the image as missing.
        self._image_default_present = False

    def set_pull_behaviour(self, behaviour: str) -> None:
        """Control ``ensure_image`` outcome on next call.

        - ``"register"`` (default): pull succeeds and image is added to
          the registry.
        - ``"fail"``: pull raises ``ProviderUnavailableError`` with
          ``context["reason"] == "DOCKER_PULL_FAILED"``.
        - ``"silently_missing"``: pull "succeeds" but image stays
          missing — used to exercise the post-pull verify-retry path
          in :class:`LocalDockerClient`. Raises
          ``ProviderUnavailableError`` with
          ``context["reason"] == "DOCKER_IMAGE_NOT_AVAILABLE"``.
        """
        if behaviour not in {"register", "fail", "silently_missing"}:
            raise ValueError(f"unknown pull behaviour: {behaviour!r}")
        self._pull_behaviour = behaviour

    # ------------------------------------------------------------------
    # Programming surface — chaos
    # ------------------------------------------------------------------

    def fail_next_n_calls(self, method: str, n: int) -> None:
        if method not in {
            "image_exists",
            "ensure_image",
            "rm_force",
            "is_container_running",
            "logs",
            "container_exit_code",
        }:
            raise ValueError(f"unknown method for fail-injection: {method!r}")
        if n < 0:
            raise ValueError("n must be non-negative")
        self._fail_counts[method] = n

    def reset_chaos(self) -> None:
        self._fail_counts.clear()
        self._pull_behaviour = "register"

    # ------------------------------------------------------------------
    # Snapshot — assertion helper
    # ------------------------------------------------------------------

    @property
    def calls(self) -> list[tuple[str, dict[str, Any]]]:
        return list(self._call_log)

    def calls_for(self, method: str) -> list[dict[str, Any]]:
        return [args for (m, args) in self._call_log if m == method]

    def snapshot(self) -> dict[str, Any]:
        return {
            "containers": {
                name: {
                    "state": rec.state,
                    "exit_code": rec.exit_code,
                    "log_chunks": len(rec.logs),
                }
                for name, rec in self._containers.items()
            },
            "known_images": sorted(self._known_images),
            "calls": list(self._call_log),
        }

    # ------------------------------------------------------------------
    # Internal — failure injection helper
    # ------------------------------------------------------------------

    def _maybe_raise(self, method: str, reason: str) -> None:
        n = self._fail_counts.get(method, 0)
        if n > 0:
            self._fail_counts[method] = n - 1
            raise ProviderUnavailableError(
                detail=f"fake_injected_failure for {method}",
                context={"reason": reason, "method": method},
            )

    # ------------------------------------------------------------------
    # IDockerClient surface
    # ------------------------------------------------------------------

    def image_exists(self, ssh: _ExecClient, image: str) -> bool:
        self._call_log.append(("image_exists", {"image": image}))
        # Injected failure for image_exists collapses to "not present"
        # because the legacy function returns ``bool`` not Result.
        if self._fail_counts.get("image_exists", 0) > 0:
            self._fail_counts["image_exists"] -= 1
            return False
        if image in self._known_images:
            return True
        return self._image_default_present

    def ensure_image(
        self,
        *,
        ssh: _ExecClient,
        image: str,
        pull_timeout_seconds: int = 1200,
        verify_after_pull: bool = True,
    ) -> None:
        self._call_log.append(
            (
                "ensure_image",
                {
                    "image": image,
                    "pull_timeout_seconds": pull_timeout_seconds,
                    "verify_after_pull": verify_after_pull,
                },
            )
        )
        self._maybe_raise("ensure_image", "DOCKER_PULL_FAILED")
        if self._pull_behaviour == "fail":
            raise ProviderUnavailableError(
                detail=f"fake pull failed for {image}",
                context={"reason": "DOCKER_PULL_FAILED", "image": image},
            )
        if self._pull_behaviour == "silently_missing":
            raise ProviderUnavailableError(
                detail=f"Image '{image}' was pulled but is not available.",
                context={"reason": "DOCKER_IMAGE_NOT_AVAILABLE", "image": image},
            )
        # Default: register and succeed.
        self._known_images.add(image)

    def rm_force(
        self,
        ssh: _ExecClient,
        *,
        container_name: str,
        timeout_seconds: int = 60,
    ) -> None:
        self._call_log.append(
            (
                "rm_force",
                {"container_name": container_name, "timeout_seconds": timeout_seconds},
            )
        )
        self._maybe_raise("rm_force", "DOCKER_RM_FAILED")
        rec = self._containers.get(container_name)
        if rec is not None:
            rec.state = "removed"

    def is_container_running(
        self,
        ssh: _ExecClient,
        *,
        name_filter: str,
        timeout_seconds: int = 5,
    ) -> bool:
        self._call_log.append(
            (
                "is_container_running",
                {"name_filter": name_filter, "timeout_seconds": timeout_seconds},
            )
        )
        # Failure injection: collapse to ``False`` (matches legacy
        # function's behaviour when ``docker ps`` returns non-zero).
        if self._fail_counts.get("is_container_running", 0) > 0:
            self._fail_counts["is_container_running"] -= 1
            return False
        rec = self._containers.get(name_filter)
        return rec is not None and rec.state == "running"

    def logs(
        self,
        ssh: _ExecClient,
        *,
        container_name: str,
        tail: int | None = None,
        timeout_seconds: int = 30,
    ) -> str:
        self._call_log.append(
            (
                "logs",
                {
                    "container_name": container_name,
                    "tail": tail,
                    "timeout_seconds": timeout_seconds,
                },
            )
        )
        self._maybe_raise("logs", "DOCKER_LOGS_FAILED")
        rec = self._containers.get(container_name)
        if rec is None:
            return ""
        chunks = rec.logs
        if tail is not None and tail > 0:
            chunks = chunks[-tail:]
        return "\n".join(chunks)

    def container_exit_code(
        self,
        ssh: _ExecClient,
        *,
        container_name: str,
        timeout_seconds: int = 5,
    ) -> int:
        self._call_log.append(
            (
                "container_exit_code",
                {"container_name": container_name, "timeout_seconds": timeout_seconds},
            )
        )
        self._maybe_raise("container_exit_code", "DOCKER_INSPECT_FAILED")
        rec = self._containers.get(container_name)
        if rec is None or rec.exit_code is None:
            # Legacy function returns the value docker reports; a
            # missing container or never-set exit code is most useful
            # in tests as "0" — assertions on exit_code use ``== N``.
            return 0
        return rec.exit_code


def assert_fake_implements_protocol() -> None:
    """Module-load sanity — fake must satisfy the runtime-checkable Protocol.

    Called from the compliance test; module-load assert keeps the
    Protocol↔Fake link visible to readers of this file.
    """
    if not isinstance(FakeDockerClient(), IDockerClient):
        raise TypeError("FakeDockerClient does not satisfy IDockerClient")


__all__ = ["FakeDockerClient"]
