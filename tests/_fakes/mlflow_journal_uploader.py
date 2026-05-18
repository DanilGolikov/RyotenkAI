"""Canonical fake for :class:`IJournalUploader` Protocol.

Use this in tests instead of ``unittest.mock.Mock(spec=IJournalUploader)``
— the sentinel :mod:`tests._lint.test_no_protocol_mocking` forbids that.

The fake enforces the at-most-once contract documented on
:class:`IJournalUploader`: a re-upload with the same ``sha256`` is
treated as a successful no-op (and recorded as ``deduped`` for tests),
while a different ``sha256`` for the same run is recorded as a
``conflict`` and surfaced via :class:`JournalConflictError`.

Example::

    uploader = FakeJournalUploader()
    uploader.upload("run-1", Path("/tmp/events.jsonl"), "abc123")
    uploader.upload("run-1", Path("/tmp/events.jsonl"), "abc123")  # dedup
    assert uploader.deduped_count == 1
    assert len(uploader.upload_calls) == 2
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class TransientJournalError(Exception):
    """Default exception raised by :meth:`FakeJournalUploader.fail_next_n_calls`."""


class JournalConflictError(Exception):
    """Raised when two uploads for the same run carry different sha256 values."""


@dataclass(frozen=True)
class JournalUploadCall:
    """Captured invocation of :meth:`FakeJournalUploader.upload`.

    :param run_id: Owning run id.
    :param journal_path: Local filesystem path of the journal file.
    :param sha256: Expected sha256 of the journal contents.
    :param was_deduped: ``True`` if this call was a no-op because the
        same digest was already uploaded for ``run_id``.
    """

    run_id: str
    journal_path: Path
    sha256: str
    was_deduped: bool


class FakeJournalUploader:
    """In-memory fake for :class:`IJournalUploader`.

    :param raise_on_conflict: When ``True`` (default), differing sha256
        values for the same ``run_id`` raise :class:`JournalConflictError`.
        Set ``False`` to record conflicts silently in ``conflicts``.
    """

    def __init__(self, *, raise_on_conflict: bool = True) -> None:
        self._raise_on_conflict = raise_on_conflict
        # run_id -> uploaded sha256
        self._tag_index: dict[str, str] = {}
        self.upload_calls: list[JournalUploadCall] = []
        self.conflicts: list[tuple[str, str, str]] = []
        self.deduped_count: int = 0
        # Chaos state.
        self._fail_remaining: int = 0
        self._fail_kind: type[Exception] = TransientJournalError

    # ------------------------------------------------------------------
    # Chaos surface
    # ------------------------------------------------------------------

    def fail_next_n_calls(
        self,
        n: int,
        kind: type[Exception] = TransientJournalError,
    ) -> None:
        """Program the next ``n`` :meth:`upload` calls to raise.

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

    def get_uploaded_sha(self, run_id: str) -> str | None:
        """Return the sha256 stored for ``run_id`` (``None`` if absent)."""
        return self._tag_index.get(run_id)

    def has_been_uploaded(self, run_id: str) -> bool:
        """Return ``True`` iff a successful upload exists for ``run_id``."""
        return run_id in self._tag_index

    # ------------------------------------------------------------------
    # IJournalUploader surface
    # ------------------------------------------------------------------

    def upload(self, run_id: str, journal_path: Path, sha256: str) -> None:
        """Record an upload, enforcing the idempotency contract.

        :param run_id: Owning run id.
        :param journal_path: Local filesystem path of the journal file.
        :param sha256: Expected sha256 hex digest.
        :raises JournalConflictError: If a previous upload for ``run_id``
            used a different sha256 and ``raise_on_conflict`` is ``True``.
        """
        self._guard()
        existing = self._tag_index.get(run_id)
        if existing is not None and existing != sha256:
            self.conflicts.append((run_id, existing, sha256))
            self.upload_calls.append(
                JournalUploadCall(
                    run_id=run_id,
                    journal_path=journal_path,
                    sha256=sha256,
                    was_deduped=False,
                )
            )
            if self._raise_on_conflict:
                raise JournalConflictError(
                    f"journal sha256 mismatch for run_id={run_id!r}: "
                    f"existing={existing!r} new={sha256!r}"
                )
            return

        was_deduped = existing == sha256
        if was_deduped:
            self.deduped_count += 1
        else:
            self._tag_index[run_id] = sha256

        self.upload_calls.append(
            JournalUploadCall(
                run_id=run_id,
                journal_path=journal_path,
                sha256=sha256,
                was_deduped=was_deduped,
            )
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _guard(self) -> None:
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise self._fail_kind("fake_injected_failure")


__all__ = [
    "FakeJournalUploader",
    "JournalConflictError",
    "JournalUploadCall",
    "TransientJournalError",
]
