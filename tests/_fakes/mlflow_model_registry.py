"""Canonical fake for :class:`IModelRegistry` Protocol.

Use this in tests instead of ``unittest.mock.Mock(spec=IModelRegistry)``
— the sentinel :mod:`tests._lint.test_no_protocol_mocking` forbids that.

The fake models MLflow 3.x alias-based promotion (``@champion``,
``@challenger``) without depending on real MLflow. It auto-increments
version numbers per model name and rejects ``set_alias``/``resolve_alias``
calls against unknown versions.

Example::

    registry = FakeModelRegistry()
    v1 = registry.register("runs:/abc/model", "sentiment")
    assert v1.version == "1"
    registry.set_alias("sentiment", "champion", v1.version)
    champ = registry.resolve_alias("sentiment", "champion")
    assert champ.run_id == "abc"  # parsed out of the URI
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class TransientRegistryError(Exception):
    """Default exception raised by :meth:`FakeModelRegistry.fail_next_n_calls`."""


class UnknownAliasError(KeyError):
    """Raised when :meth:`FakeModelRegistry.resolve_alias` cannot resolve."""


class UnknownVersionError(KeyError):
    """Raised when :meth:`FakeModelRegistry.set_alias` targets a missing version."""


_RUN_URI_RE = re.compile(r"^runs:/(?P<run_id>[^/]+)/.*$")


@dataclass(frozen=True)
class FakeModelVersion:
    """Concrete value object satisfying :class:`ModelVersion` Protocol.

    :param name: Registered model name.
    :param version: Stringified monotonically-increasing version.
    :param run_id: Source run id parsed from ``runs:/<id>/...`` URIs;
        ``None`` if the URI did not follow that scheme.
    """

    name: str
    version: str
    run_id: str | None


@dataclass(frozen=True)
class RegisterCall:
    """Captured invocation of :meth:`FakeModelRegistry.register`."""

    model_uri: str
    name: str
    version: str


@dataclass(frozen=True)
class SetAliasCall:
    """Captured invocation of :meth:`FakeModelRegistry.set_alias`."""

    name: str
    alias: str
    version: str


class FakeModelRegistry:
    """In-memory fake for :class:`IModelRegistry`.

    Maintains a ``name -> [versions]`` map and a ``(name, alias) ->
    version`` index. Aliases are mutable and overwrite freely; the
    registry does not enforce alias uniqueness across versions.
    """

    def __init__(self) -> None:
        # name -> ordered list of FakeModelVersion entries (insertion order).
        self._versions: dict[str, list[FakeModelVersion]] = {}
        # (name, alias) -> version string.
        self._aliases: dict[tuple[str, str], str] = {}
        # Call logs.
        self.register_calls: list[RegisterCall] = []
        self.set_alias_calls: list[SetAliasCall] = []
        self.resolve_alias_calls: list[tuple[str, str]] = []
        # Chaos state.
        self._fail_remaining: int = 0
        self._fail_kind: type[Exception] = TransientRegistryError

    # ------------------------------------------------------------------
    # Chaos surface
    # ------------------------------------------------------------------

    def fail_next_n_calls(
        self,
        n: int,
        kind: type[Exception] = TransientRegistryError,
    ) -> None:
        """Program the next ``n`` calls to raise.

        :param n: Non-negative count of failures.
        :param kind: Exception class to raise.
        :raises ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError("fail_next_n_calls requires non-negative count")
        self._fail_remaining = n
        self._fail_kind = kind

    def reset_chaos(self) -> None:
        """Clear chaos state."""
        self._fail_remaining = 0

    # ------------------------------------------------------------------
    # Inspection helpers (test convenience)
    # ------------------------------------------------------------------

    def list_versions(self, name: str) -> list[FakeModelVersion]:
        """Return every registered version for ``name`` (insertion order)."""
        return list(self._versions.get(name, []))

    def aliases_for(self, name: str) -> dict[str, str]:
        """Return a ``{alias: version}`` snapshot for ``name``."""
        return {
            alias: ver
            for (n, alias), ver in self._aliases.items()
            if n == name
        }

    # ------------------------------------------------------------------
    # IModelRegistry surface
    # ------------------------------------------------------------------

    def register(self, model_uri: str, name: str) -> FakeModelVersion:
        """Register a new version for ``name`` pointing at ``model_uri``.

        :param model_uri: Source URI (typically ``runs:/<run_id>/<path>``).
        :param name: Registered model name.
        :returns: New :class:`FakeModelVersion` with auto-incremented version.
        """
        self._guard()
        versions = self._versions.setdefault(name, [])
        next_version = str(len(versions) + 1)
        match = _RUN_URI_RE.match(model_uri)
        run_id = match.group("run_id") if match is not None else None
        mv = FakeModelVersion(name=name, version=next_version, run_id=run_id)
        versions.append(mv)
        self.register_calls.append(
            RegisterCall(model_uri=model_uri, name=name, version=next_version)
        )
        return mv

    def set_alias(self, name: str, alias: str, version: str) -> None:
        """Point ``alias`` at ``version`` of ``name``.

        :param name: Registered model name.
        :param alias: Alias label (e.g. ``"champion"``).
        :param version: Existing version string.
        :raises UnknownVersionError: If ``version`` is not registered.
        """
        self._guard()
        versions = self._versions.get(name, [])
        if not any(v.version == version for v in versions):
            raise UnknownVersionError(
                f"unknown version for model {name!r}: {version!r}"
            )
        self._aliases[(name, alias)] = version
        self.set_alias_calls.append(
            SetAliasCall(name=name, alias=alias, version=version)
        )

    def resolve_alias(self, name: str, alias: str) -> FakeModelVersion:
        """Return the version currently pointed to by ``alias`` of ``name``.

        :param name: Registered model name.
        :param alias: Alias label.
        :returns: The :class:`FakeModelVersion` the alias references.
        :raises UnknownAliasError: If the alias is not registered.
        """
        self._guard()
        self.resolve_alias_calls.append((name, alias))
        version = self._aliases.get((name, alias))
        if version is None:
            raise UnknownAliasError(
                f"no alias {alias!r} registered for model {name!r}"
            )
        versions = self._versions.get(name, [])
        for v in versions:
            if v.version == version:
                return v
        # Inconsistent state should never happen because set_alias guards;
        # raise loudly so callers notice.
        raise UnknownVersionError(
            f"alias {alias!r} -> version {version!r} no longer exists"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _guard(self) -> None:
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise self._fail_kind("fake_injected_failure")


__all__ = [
    "FakeModelRegistry",
    "FakeModelVersion",
    "RegisterCall",
    "SetAliasCall",
    "TransientRegistryError",
    "UnknownAliasError",
    "UnknownVersionError",
]
