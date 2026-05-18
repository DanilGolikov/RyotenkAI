"""Canonical fake for :class:`IPromptRegistry` Protocol.

Use this in tests instead of ``unittest.mock.Mock(spec=IPromptRegistry)``
— the sentinel :mod:`tests._lint.test_no_protocol_mocking` forbids that.

The fake serves :class:`FakePromptArtifact` records by name/URI and
records timeouts requested by callers. Tests seed prompts via
:meth:`add_prompt`, configure missing-prompt behaviour, and inject
failures via :meth:`fail_next_n_calls`.

Example::

    registry = FakePromptRegistry()
    registry.add_prompt("greet", "1", "Hello {name}")
    artifact = registry.load("greet", timeout_s=1.0)
    assert artifact is not None
    assert artifact.template == "Hello {name}"
"""

from __future__ import annotations

from dataclasses import dataclass


class TransientPromptError(Exception):
    """Default exception raised by :meth:`FakePromptRegistry.fail_next_n_calls`."""


class PromptTimeoutError(Exception):
    """Raised when ``timeout_s`` is below the configured minimum."""


@dataclass(frozen=True)
class FakePromptArtifact:
    """Concrete value object satisfying :class:`PromptArtifact` Protocol.

    :param name: Prompt name (excluding the version suffix).
    :param version: Stringified version.
    :param template: Prompt body / template text.
    """

    name: str
    version: str
    template: str


@dataclass(frozen=True)
class LoadCall:
    """Captured invocation of :meth:`FakePromptRegistry.load`.

    :param name_or_uri: Argument as provided by the caller.
    :param timeout_s: Timeout in seconds.
    :param resolved: Resolved ``(name, version)`` if found, else ``None``.
    """

    name_or_uri: str
    timeout_s: float
    resolved: tuple[str, str] | None


def _parse_name_or_uri(name_or_uri: str) -> tuple[str, str | None]:
    """Split a ``prompts:/name/version`` URI (or bare name) into parts.

    :param name_or_uri: Either a bare prompt name or a
        ``prompts:/<name>/<version>`` URI.
    :returns: ``(name, version_or_None)``.
    """
    if name_or_uri.startswith("prompts:/"):
        suffix = name_or_uri.removeprefix("prompts:/")
        parts = suffix.split("/", maxsplit=1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return parts[0], None
    return name_or_uri, None


class FakePromptRegistry:
    """In-memory fake for :class:`IPromptRegistry`.

    :param raise_on_timeout_below_s: When set, ``load`` raises
        :class:`PromptTimeoutError` if the caller's ``timeout_s`` is
        strictly less than this floor. Defaults to ``None`` (no timeout
        enforcement).
    :param return_none_on_missing: When ``True`` (default), missing
        prompts return ``None`` (matches the Protocol's documented shape).
        When ``False``, missing prompts raise :class:`KeyError`.
    """

    def __init__(
        self,
        *,
        raise_on_timeout_below_s: float | None = None,
        return_none_on_missing: bool = True,
    ) -> None:
        self._timeout_floor = raise_on_timeout_below_s
        self._return_none_on_missing = return_none_on_missing
        # (name, version) -> FakePromptArtifact
        self._prompts: dict[tuple[str, str], FakePromptArtifact] = {}
        # name -> latest version string (insertion order rather than numeric).
        self._latest: dict[str, str] = {}
        # Call log.
        self.load_calls: list[LoadCall] = []
        # Chaos state.
        self._fail_remaining: int = 0
        self._fail_kind: type[Exception] = TransientPromptError

    # ------------------------------------------------------------------
    # Chaos surface
    # ------------------------------------------------------------------

    def fail_next_n_calls(
        self,
        n: int,
        kind: type[Exception] = TransientPromptError,
    ) -> None:
        """Program the next ``n`` :meth:`load` calls to raise.

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
    # Seed helpers (test convenience)
    # ------------------------------------------------------------------

    def add_prompt(self, name: str, version: str, template: str) -> FakePromptArtifact:
        """Insert (or overwrite) a prompt artifact.

        The most recently added version becomes the implicit "latest"
        used by bare-name lookups.

        :param name: Prompt name.
        :param version: Version string.
        :param template: Prompt body / template text.
        :returns: The newly registered :class:`FakePromptArtifact`.
        """
        artifact = FakePromptArtifact(name=name, version=version, template=template)
        self._prompts[(name, version)] = artifact
        self._latest[name] = version
        return artifact

    def remove_prompt(self, name: str, version: str | None = None) -> None:
        """Delete prompts for ``name``.

        :param name: Prompt name.
        :param version: When ``None`` (default) all versions are dropped;
            otherwise only the named version is removed.
        """
        if version is None:
            self._prompts = {
                k: v for k, v in self._prompts.items() if k[0] != name
            }
            self._latest.pop(name, None)
        else:
            self._prompts.pop((name, version), None)
            if self._latest.get(name) == version:
                # Reset latest to any surviving version, else drop.
                surviving = [v for (n, v) in self._prompts if n == name]
                if surviving:
                    self._latest[name] = surviving[-1]
                else:
                    self._latest.pop(name, None)

    # ------------------------------------------------------------------
    # IPromptRegistry surface
    # ------------------------------------------------------------------

    def load(self, name_or_uri: str, timeout_s: float) -> FakePromptArtifact | None:
        """Resolve a prompt by name or ``prompts:/`` URI.

        :param name_or_uri: Bare prompt name (resolves to latest version)
            or a ``prompts:/<name>/<version>`` URI.
        :param timeout_s: Timeout requested by the caller.
        :returns: The matching :class:`FakePromptArtifact`, or ``None`` if
            no prompt is registered (subject to ``return_none_on_missing``).
        :raises PromptTimeoutError: If the timeout floor is configured and
            ``timeout_s`` is below it.
        :raises KeyError: If ``return_none_on_missing`` is ``False`` and
            the prompt is absent.
        """
        self._guard()
        if (
            self._timeout_floor is not None
            and timeout_s < self._timeout_floor
        ):
            self.load_calls.append(
                LoadCall(name_or_uri=name_or_uri, timeout_s=timeout_s, resolved=None)
            )
            raise PromptTimeoutError(
                f"timeout_s={timeout_s!r} is below configured floor "
                f"{self._timeout_floor!r}"
            )

        name, version = _parse_name_or_uri(name_or_uri)
        resolved_version = version or self._latest.get(name)
        if resolved_version is None:
            self.load_calls.append(
                LoadCall(name_or_uri=name_or_uri, timeout_s=timeout_s, resolved=None)
            )
            if self._return_none_on_missing:
                return None
            raise KeyError(f"no prompt registered for name {name!r}")

        artifact = self._prompts.get((name, resolved_version))
        if artifact is None:
            self.load_calls.append(
                LoadCall(name_or_uri=name_or_uri, timeout_s=timeout_s, resolved=None)
            )
            if self._return_none_on_missing:
                return None
            raise KeyError(
                f"no prompt registered for ({name!r}, {resolved_version!r})"
            )

        self.load_calls.append(
            LoadCall(
                name_or_uri=name_or_uri,
                timeout_s=timeout_s,
                resolved=(name, resolved_version),
            )
        )
        return artifact

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _guard(self) -> None:
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise self._fail_kind("fake_injected_failure")


__all__ = [
    "FakePromptArtifact",
    "FakePromptRegistry",
    "LoadCall",
    "PromptTimeoutError",
    "TransientPromptError",
]
