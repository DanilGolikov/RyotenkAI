"""Canonical fake for :class:`IArtifactSink` Protocol.

Use this in tests instead of ``unittest.mock.Mock(spec=IArtifactSink)``
— the sentinel :mod:`tests._lint.test_no_protocol_mocking` forbids that.

The fake records every upload request and (optionally) verifies the
sha256 checksum of the local file. No real filesystem mutation outside
the source path being read; the artifact target is purely in-memory.

Example::

    sink = FakeArtifactSink()
    sink.upload_file(
        run_id="run-1",
        local_path=Path("/tmp/journal.jsonl"),
        artifact_path="journal/events.jsonl",
        checksum_sha256=None,
    )
    assert len(sink.upload_calls) == 1
    assert sink.upload_calls[0].artifact_path == "journal/events.jsonl"
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class TransientArtifactError(Exception):
    """Default exception raised by :meth:`FakeArtifactSink.fail_next_n_calls`."""


class ChecksumMismatchError(Exception):
    """Raised when ``verify_checksums=True`` and the actual sha256 differs."""


@dataclass(frozen=True)
class UploadCall:
    """Captured invocation of :meth:`FakeArtifactSink.upload_file`.

    :param run_id: Owning run id.
    :param local_path: Local filesystem path passed by the caller.
    :param artifact_path: Logical artifact path inside the run.
    :param checksum_sha256: Expected sha256 hex string, or ``None`` if
        the caller did not request verification.
    :param size_bytes: Size of the local file at upload time (``None``
        if the file did not exist when ``upload_file`` was called).
    """

    run_id: str
    local_path: Path
    artifact_path: str
    checksum_sha256: str | None
    size_bytes: int | None


class FakeArtifactSink:
    """In-memory fake for :class:`IArtifactSink`.

    :param verify_checksums: When ``True``, :meth:`upload_file` reads the
        local file and asserts its sha256 matches ``checksum_sha256``.
        Defaults to ``False`` so tests can pass any path that exists
        (or doesn't) without computing a real digest.
    :param require_existing_file: When ``True`` (default), the local
        path must exist or :class:`FileNotFoundError` is raised. Set
        ``False`` to allow purely symbolic upload calls.
    """

    def __init__(
        self,
        *,
        verify_checksums: bool = False,
        require_existing_file: bool = True,
    ) -> None:
        self._verify_checksums = verify_checksums
        self._require_existing_file = require_existing_file
        self.upload_calls: list[UploadCall] = []
        # Map (run_id, artifact_path) -> sha256 of the uploaded file.
        # Useful for assertions like "this artifact was uploaded with
        # the same digest as that one".
        self.artifacts: dict[tuple[str, str], str] = {}
        # Chaos state.
        self._fail_remaining: int = 0
        self._fail_kind: type[Exception] = TransientArtifactError

    # ------------------------------------------------------------------
    # Chaos surface
    # ------------------------------------------------------------------

    def fail_next_n_calls(
        self,
        n: int,
        kind: type[Exception] = TransientArtifactError,
    ) -> None:
        """Program the next ``n`` :meth:`upload_file` calls to raise.

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

    def calls_for_run(self, run_id: str) -> list[UploadCall]:
        """Return all upload calls scoped to ``run_id``."""
        return [c for c in self.upload_calls if c.run_id == run_id]

    def get_digest(self, run_id: str, artifact_path: str) -> str | None:
        """Return the sha256 hex of the uploaded artifact, if known."""
        return self.artifacts.get((run_id, artifact_path))

    # ------------------------------------------------------------------
    # IArtifactSink surface
    # ------------------------------------------------------------------

    def upload_file(
        self,
        run_id: str,
        local_path: Path,
        artifact_path: str,
        checksum_sha256: str | None,
    ) -> None:
        """Record an upload request.

        :param run_id: Target run id.
        :param local_path: Local filesystem path of the artifact.
        :param artifact_path: Logical artifact path inside the run.
        :param checksum_sha256: Expected sha256 hex digest (or ``None``).
        :raises FileNotFoundError: If ``require_existing_file`` is set
            and ``local_path`` does not exist.
        :raises ChecksumMismatchError: If ``verify_checksums`` is set and
            the computed digest differs from ``checksum_sha256``.
        """
        self._guard()

        size_bytes: int | None = None
        actual_digest: str | None = None
        if local_path.exists() and local_path.is_file():
            data = local_path.read_bytes()
            size_bytes = len(data)
            actual_digest = hashlib.sha256(data).hexdigest()
        elif self._require_existing_file:
            raise FileNotFoundError(f"local_path does not exist: {local_path}")

        if self._verify_checksums and checksum_sha256 is not None:
            if actual_digest != checksum_sha256:
                raise ChecksumMismatchError(
                    f"checksum mismatch for {local_path}: "
                    f"expected={checksum_sha256!r} actual={actual_digest!r}"
                )

        self.upload_calls.append(
            UploadCall(
                run_id=run_id,
                local_path=local_path,
                artifact_path=artifact_path,
                checksum_sha256=checksum_sha256,
                size_bytes=size_bytes,
            )
        )
        if actual_digest is not None:
            self.artifacts[(run_id, artifact_path)] = actual_digest

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _guard(self) -> None:
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise self._fail_kind("fake_injected_failure")


__all__ = [
    "ChecksumMismatchError",
    "FakeArtifactSink",
    "TransientArtifactError",
    "UploadCall",
]
