"""Canonical fake for :class:`ITrackingClient` Protocol.

Use this in tests instead of ``unittest.mock.Mock(spec=ITrackingClient)``
— the sentinel :mod:`tests._lint.test_no_protocol_mocking` forbids that.

The fake holds all run state in-memory and records every call so tests can
make assertions about lifecycle ordering, tag/param sets, and termination
status. No external dependencies, no real MLflow calls.

Chaos surface
-------------
* :meth:`fail_next_n_calls` — count-down failure injection across the API
* :meth:`set_unavailable` — every call raises until cleared
* :meth:`reset_chaos` — clear all chaos state

Example::

    fake = FakeTrackingClient()
    handle = fake.start_run("exp", "run-1", tags={}, params={"lr": "0.01"})
    fake.set_tags(handle.run_id, {"role": "child"})
    fake.set_terminated(handle.run_id, RunStatus.FINISHED)

    assert len(fake.start_run_calls) == 1
    assert fake.start_run_calls[0].experiment == "exp"
    assert fake.terminated_calls == [(handle.run_id, RunStatus.FINISHED)]
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from ryotenkai_shared.infrastructure.mlflow.protocols import RunStatus
from ryotenkai_shared.infrastructure.mlflow.run_handle import RunHandle

if TYPE_CHECKING:
    pass


class TransientTrackingError(Exception):
    """Default exception class raised by :meth:`FakeTrackingClient.fail_next_n_calls`."""


class TrackingUnavailableError(Exception):
    """Raised when the fake is in ``set_unavailable(True)`` mode."""


@dataclass(frozen=True)
class StartRunCall:
    """Captured invocation of :meth:`FakeTrackingClient.start_run`.

    :param experiment: Experiment name passed by the caller.
    :param name: Run name passed by the caller.
    :param tags: Snapshot of the tags mapping (frozen at call time).
    :param params: Snapshot of the params mapping (frozen at call time).
    :param run_id: Synthetic run id assigned by the fake.
    """

    experiment: str
    name: str
    tags: dict[str, str]
    params: dict[str, str]
    run_id: str


@dataclass(frozen=True)
class StartNestedRunCall:
    """Captured invocation of :meth:`FakeTrackingClient.start_nested_run`."""

    parent_run_id: str
    name: str
    tags: dict[str, str]
    run_id: str


@dataclass(frozen=True)
class SetTagsCall:
    """Captured invocation of :meth:`FakeTrackingClient.set_tags`."""

    run_id: str
    tags: dict[str, str]


class FakeTrackingClient:
    """In-memory fake for :class:`ITrackingClient`.

    Holds run state in ``_runs`` (``run_id`` -> :class:`RunHandle`) and
    exposes call logs for assertions. Constructed without arguments by
    default; programmable failure modes are opt-in.

    :param ping_should_raise: If set, :meth:`ping` raises this exception
        instead of returning ``None``. Cleared by :meth:`reset_chaos`.
    :param tracking_uri: URI baked into emitted :class:`RunHandle`
        instances. Defaults to ``"fake://in-memory"``.
    """

    def __init__(
        self,
        *,
        ping_should_raise: Exception | None = None,
        tracking_uri: str = "fake://in-memory",
    ) -> None:
        self._ping_should_raise = ping_should_raise
        self._tracking_uri = tracking_uri
        self._runs: dict[str, RunHandle] = {}
        self._tags: dict[str, dict[str, str]] = {}
        self._params: dict[str, dict[str, str]] = {}
        # Call logs
        self.ping_calls: list[float] = []
        self.start_run_calls: list[StartRunCall] = []
        self.start_nested_run_calls: list[StartNestedRunCall] = []
        self.adopt_run_calls: list[str] = []
        self.terminated_calls: list[tuple[str, RunStatus]] = []
        self.set_tags_calls: list[SetTagsCall] = []
        # Chaos state
        self._fail_remaining: int = 0
        self._fail_kind: type[Exception] = TransientTrackingError
        self._unavailable: bool = False

    # ------------------------------------------------------------------
    # Chaos surface
    # ------------------------------------------------------------------

    def fail_next_n_calls(
        self,
        n: int,
        kind: type[Exception] = TransientTrackingError,
    ) -> None:
        """Program the next ``n`` mutating calls to raise ``kind``.

        :param n: Non-negative count of failures to inject.
        :param kind: Exception class to raise (default
            :class:`TransientTrackingError`).
        :raises ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError("fail_next_n_calls requires non-negative count")
        self._fail_remaining = n
        self._fail_kind = kind

    def set_unavailable(self, value: bool) -> None:
        """When ``True`` every call raises :class:`TrackingUnavailableError`."""
        self._unavailable = value

    def reset_chaos(self) -> None:
        """Clear all chaos state and ping override."""
        self._fail_remaining = 0
        self._unavailable = False
        self._ping_should_raise = None

    # ------------------------------------------------------------------
    # Inspection helpers (test convenience)
    # ------------------------------------------------------------------

    def get_tags(self, run_id: str) -> dict[str, str]:
        """Return a snapshot of tags currently attached to ``run_id``."""
        return dict(self._tags.get(run_id, {}))

    def get_params(self, run_id: str) -> dict[str, str]:
        """Return a snapshot of params currently attached to ``run_id``."""
        return dict(self._params.get(run_id, {}))

    def has_run(self, run_id: str) -> bool:
        """Return ``True`` if ``run_id`` was emitted by this fake."""
        return run_id in self._runs

    # ------------------------------------------------------------------
    # ITrackingClient surface
    # ------------------------------------------------------------------

    def ping(self, timeout_s: float) -> None:
        """Record the call and raise if ``ping_should_raise`` is set.

        :param timeout_s: Timeout in seconds (recorded for assertion).
        """
        self.ping_calls.append(timeout_s)
        if self._ping_should_raise is not None:
            raise self._ping_should_raise
        self._guard()

    def start_run(
        self,
        experiment: str,
        name: str,
        tags: Mapping[str, str],
        params: Mapping[str, str],
    ) -> RunHandle:
        """Create a new top-level run and return its :class:`RunHandle`.

        :param experiment: Target experiment name.
        :param name: Run name (recorded but not embedded in the handle).
        :param tags: Initial tag set (snapshotted into fake state).
        :param params: Initial param set (snapshotted into fake state).
        :returns: Frozen :class:`RunHandle` with synthetic run id.
        """
        self._guard()
        run_id = f"fake-{uuid4().hex[:12]}"
        handle = RunHandle(
            run_id=run_id,
            experiment_id=f"exp-{experiment}",
            parent_run_id=None,
            tracking_uri=self._tracking_uri,
            status=RunStatus.RUNNING,
        )
        tags_copy = dict(tags)
        params_copy = dict(params)
        self._runs[run_id] = handle
        self._tags[run_id] = tags_copy
        self._params[run_id] = params_copy
        self.start_run_calls.append(
            StartRunCall(
                experiment=experiment,
                name=name,
                tags=tags_copy,
                params=params_copy,
                run_id=run_id,
            )
        )
        return handle

    def start_nested_run(
        self,
        parent_run_id: str,
        name: str,
        tags: Mapping[str, str],
    ) -> RunHandle:
        """Create a nested child of ``parent_run_id``.

        :param parent_run_id: Parent run id (must already exist).
        :param name: Child run name.
        :param tags: Initial tag set for the child.
        :returns: Frozen :class:`RunHandle` for the child run.
        :raises KeyError: If ``parent_run_id`` does not exist.
        """
        self._guard()
        if parent_run_id not in self._runs:
            raise KeyError(f"unknown parent_run_id: {parent_run_id}")
        parent = self._runs[parent_run_id]
        run_id = f"fake-{uuid4().hex[:12]}"
        handle = RunHandle(
            run_id=run_id,
            experiment_id=parent.experiment_id,
            parent_run_id=parent_run_id,
            tracking_uri=self._tracking_uri,
            status=RunStatus.RUNNING,
        )
        tags_copy = dict(tags)
        self._runs[run_id] = handle
        self._tags[run_id] = tags_copy
        self._params[run_id] = {}
        self.start_nested_run_calls.append(
            StartNestedRunCall(
                parent_run_id=parent_run_id,
                name=name,
                tags=tags_copy,
                run_id=run_id,
            )
        )
        return handle

    def adopt_run(self, run_id: str) -> RunHandle:
        """Re-open an existing run by id and return its handle.

        :param run_id: Identifier of a run previously created by this fake.
        :returns: Frozen :class:`RunHandle`.
        :raises KeyError: If ``run_id`` is unknown.
        """
        self._guard()
        self.adopt_run_calls.append(run_id)
        if run_id not in self._runs:
            raise KeyError(f"unknown run_id: {run_id}")
        return self._runs[run_id]

    def set_terminated(self, run_id: str, status: RunStatus) -> None:
        """Mark ``run_id`` terminated with ``status``.

        Replaces the stored :class:`RunHandle` with a copy whose ``status``
        field matches ``status``. Idempotent if called twice with the same
        terminal status.

        :param run_id: Run id to terminate.
        :param status: Terminal :class:`RunStatus`.
        :raises KeyError: If ``run_id`` is unknown.
        """
        self._guard()
        if run_id not in self._runs:
            raise KeyError(f"unknown run_id: {run_id}")
        old = self._runs[run_id]
        self._runs[run_id] = RunHandle(
            run_id=old.run_id,
            experiment_id=old.experiment_id,
            parent_run_id=old.parent_run_id,
            tracking_uri=old.tracking_uri,
            status=status,
        )
        self.terminated_calls.append((run_id, status))

    def set_tags(self, run_id: str, tags: Mapping[str, str]) -> None:
        """Merge ``tags`` into the tag set for ``run_id``.

        :param run_id: Target run id.
        :param tags: Tags to merge (overrides existing keys).
        :raises KeyError: If ``run_id`` is unknown.
        """
        self._guard()
        if run_id not in self._runs:
            raise KeyError(f"unknown run_id: {run_id}")
        tags_copy = dict(tags)
        self._tags[run_id].update(tags_copy)
        self.set_tags_calls.append(SetTagsCall(run_id=run_id, tags=tags_copy))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _guard(self) -> None:
        """Apply chaos hooks at the top of each ITrackingClient call."""
        if self._unavailable:
            raise TrackingUnavailableError("fake_tracking_unavailable")
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise self._fail_kind("fake_injected_failure")


__all__ = [
    "FakeTrackingClient",
    "SetTagsCall",
    "StartNestedRunCall",
    "StartRunCall",
    "TrackingUnavailableError",
    "TransientTrackingError",
]
