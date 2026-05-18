"""Idempotent SSOT-journal artifact uploader.

Concrete :class:`~.protocols.IJournalUploader` implementation.

Uploads ``workspace/runs/<run_id>/events.jsonl`` (the unified-event-system
SSOT journal, per ADR-0009) as an MLflow artifact under ``events/``.
Idempotency is enforced via a server-side sha256 marker stamped on the
parent run as the :data:`~.taxonomy.TagKey.JOURNAL_SHA256` tag — a
second :meth:`upload` call with the same checksum is a no-op.

This replaces the narrow upload-half of the legacy
``MlflowFinalizer`` event handler (the lifecycle-tag half migrates to
``control.pipeline.mlflow.lifecycle.finalizer``).

Never raises
------------
Per the design doc (§Write path step 7), upload failures must not
propagate — the run's lifecycle is independent of journal upload
success. On final retry-budget exhaustion the failure is structurally
logged so the orchestrator surfaces it via the manifest, but the
caller continues.

Per ``docs/plans/vectorized-fluttering-mist.md`` §Target architecture.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from ryotenkai_shared.infrastructure.mlflow.taxonomy import TagKey
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from ryotenkai_shared.infrastructure.mlflow.transport import MlflowTransport

logger = get_logger(__name__)


class JournalUploader:
    """Concrete :class:`~.protocols.IJournalUploader`.

    :param client: Configured :class:`MlflowTransport`.
    :param retry_delays_s: Sleep durations (seconds) between retry
        attempts. A tuple of length N triggers up to N+1 total
        attempts (initial + N retries). Default
        ``(1.0, 5.0, 30.0)`` matches the legacy
        ``MlflowFinalizer`` cadence.
    """

    def __init__(
        self,
        client: MlflowTransport,
        retry_delays_s: tuple[float, ...] = (1.0, 5.0, 30.0),
    ) -> None:
        self._client = client
        self._retry_delays_s = tuple(float(d) for d in retry_delays_s)

    def upload(self, run_id: str, journal_path: Path, sha256: str) -> None:
        """Upload ``journal_path`` to MLflow under ``events/``.

        Algorithm:

        1. Read the run's current
           :data:`~.taxonomy.TagKey.JOURNAL_SHA256` tag. If it equals
           ``sha256`` the artifact is already uploaded — no-op.
        2. Else attempt ``log_artifact(run_id, journal_path,
           artifact_path="events")`` with bounded retries.
        3. On success, stamp the sha256 tag so subsequent calls are
           idempotent.

        Failures are logged but never raised — the caller's lifecycle
        proceeds regardless.

        :param run_id: Target run.
        :param journal_path: Local path to the ``events.jsonl`` file.
        :param sha256: Hex digest of the file's contents; used as the
            idempotency key.
        """
        if not sha256:
            logger.warning(
                "[JOURNAL] empty sha256 for run=%s; skipping upload (idempotency "
                "guard requires a digest)",
                run_id,
            )
            return
        if not journal_path.exists():
            logger.warning(
                "[JOURNAL] journal file missing at %s; skipping upload "
                "for run=%s",
                journal_path,
                run_id,
            )
            return

        # Step 1: idempotency check.
        if self._already_uploaded(run_id, sha256):
            logger.info(
                "[JOURNAL] run=%s already has matching sha256 tag; skip upload",
                run_id,
            )
            return

        # Step 2: upload with retries.
        if not self._upload_with_retries(run_id, journal_path):
            logger.warning(
                "[JOURNAL] upload exhausted retry budget for run=%s "
                "(attempts=%d); marking failed downstream via manifest.",
                run_id,
                len(self._retry_delays_s) + 1,
            )
            return

        # Step 3: stamp idempotency tag.
        try:
            self._client.set_tags(run_id, {TagKey.JOURNAL_SHA256.value: sha256})
            logger.info(
                "[JOURNAL] uploaded events.jsonl for run=%s sha256=%s",
                run_id,
                sha256[:12],
            )
        except Exception as exc:  # noqa: BLE001 — boundary; never raise
            logger.warning(
                "[JOURNAL] sha256 tag set failed for run=%s after successful "
                "upload (next call will redo upload): %s",
                run_id,
                exc,
            )

    # -- internal helpers -------------------------------------------

    def _already_uploaded(self, run_id: str, sha256: str) -> bool:
        """Return ``True`` if the run already carries the matching
        sha256 tag."""
        try:
            run = self._client.client.get_run(run_id)
        except Exception as exc:  # noqa: BLE001 — read failure → re-upload
            logger.debug(
                "[JOURNAL] get_run failed for run=%s during idempotency "
                "check (will attempt upload): %s",
                run_id,
                exc,
            )
            return False
        existing = run.data.tags.get(TagKey.JOURNAL_SHA256.value)
        return existing == sha256

    def _upload_with_retries(self, run_id: str, journal_path: Path) -> bool:
        """Attempt ``log_artifact`` with the configured delay schedule.

        :returns: ``True`` on success, ``False`` after all attempts fail.
        """
        attempts = (0.0, *self._retry_delays_s)
        for attempt_no, delay in enumerate(attempts):
            if delay > 0:
                time.sleep(delay)
            try:
                self._client.client.log_artifact(
                    run_id,
                    str(journal_path),
                    artifact_path="events",
                )
                return True
            except Exception as exc:  # noqa: BLE001 — boundary, classify next
                logger.warning(
                    "[JOURNAL] upload attempt %d/%d failed for run=%s: %s",
                    attempt_no + 1,
                    len(attempts),
                    run_id,
                    exc,
                )
        return False


__all__ = ["JournalUploader"]
