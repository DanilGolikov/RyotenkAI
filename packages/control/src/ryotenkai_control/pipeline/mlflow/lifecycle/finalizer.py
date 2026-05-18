"""Idempotent close path for MLflow runs.

The legacy
:mod:`ryotenkai_control.pipeline.mlflow_attempt.manager.MLflowAttemptManager`
had four parallel close paths (orchestrator ``finally``, atexit,
SIGTERM, reconciliation thread). Each one repeated the same
"upload journal -> set tags -> set_terminated" sequence with subtly
different error handling. The redesign converges all of them onto a
single :meth:`MlflowFinalizer.finalize` invocation guarded by the
:data:`~.taxonomy.TagKey.LIFECYCLE_FINALIZED` server-side tag.

Idempotency contract
--------------------

1. Read the run's current tags via :meth:`IRunQuery.get_run`.
2. If ``ryotenkai.lifecycle.finalized == "true"``: log + return.
3. Else: upload the journal (best-effort -- failures swallowed by
   the uploader), stamp the lifecycle tags, then
   :meth:`ITrackingClient.set_terminated`.

Never raises
------------
Per R-17 in the plan, this function MUST complete -- the
:class:`RunLifecycleCoord` mutex that wraps it cannot tolerate an
exception escaping into the atexit / signal handler that called it.
Every step is wrapped in ``try/except Exception`` and degrades to a
``logger.warning``. The orchestrator therefore observes finalize as
"always succeeds" from the caller's perspective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_shared.infrastructure.mlflow.taxonomy import TagKey
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from ryotenkai_shared.infrastructure.mlflow.protocols import (
        IJournalUploader,
        IRunQuery,
        ITrackingClient,
        RunStatus,
    )
    from ryotenkai_shared.infrastructure.mlflow.run_handle import RunHandle


logger = get_logger(__name__)


__all__ = ["MlflowFinalizer"]


class MlflowFinalizer:
    """Single, idempotent close path for an MLflow run.

    :param client: Concrete :class:`ITrackingClient` used for
        :meth:`ITrackingClient.set_tags` and
        :meth:`ITrackingClient.set_terminated`.
    :param journal_uploader: Implementation of :class:`IJournalUploader`
        responsible for uploading the SSOT journal artifact. Already
        idempotent on the upload side via its own sha256 tag.
    :param run_query: Implementation of :class:`IRunQuery` used to
        read the existing lifecycle tags before re-closing.
    """

    def __init__(
        self,
        client: ITrackingClient,
        journal_uploader: IJournalUploader,
        run_query: IRunQuery,
    ) -> None:
        self._client = client
        self._journal_uploader = journal_uploader
        self._run_query = run_query

    def finalize(
        self,
        *,
        run: RunHandle,
        status: RunStatus,
        journal_path: Path | None,
        journal_sha256: str | None,
        exit_reason: str | None = None,
    ) -> None:
        """Close ``run`` with ``status``. Never raises.

        Order of operations:

        1. **Idempotency check** -- if the run already has
           ``ryotenkai.lifecycle.finalized="true"`` we skip every
           subsequent step.
        2. **Journal upload** -- when both ``journal_path`` and
           ``journal_sha256`` are supplied, delegate to
           :class:`IJournalUploader`. The uploader is itself
           idempotent (see
           :mod:`ryotenkai_shared.infrastructure.mlflow.journal_uploader`).
        3. **Lifecycle tags** -- stamp
           :data:`~.taxonomy.TagKey.LIFECYCLE_FINALIZED`,
           :data:`~.taxonomy.TagKey.LIFECYCLE_STATUS`, and
           :data:`~.taxonomy.TagKey.EXIT_REASON` so subsequent reads
           short-circuit step 1 on retry.
        4. **set_terminated** -- mark the run as terminated with
           ``status``.

        :param run: :class:`RunHandle` returned by
            :class:`ParentRunOpener`.
        :param status: Terminal :class:`RunStatus`.
        :param journal_path: Local path to the SSOT journal file, or
            ``None`` to skip the upload step.
        :param journal_sha256: Hex digest of the journal contents.
            ``None`` disables journal upload (same as omitting
            ``journal_path``).
        :param exit_reason: Free-form reason stamped onto
            :data:`~.taxonomy.TagKey.EXIT_REASON`. ``None`` is rendered
            as the empty string so the tag is always present (eases
            UI filtering).
        """
        # Step 1: idempotency check.
        if self._already_finalized(run.run_id):
            logger.info(
                "[MLFLOW_FINALIZER] run=%s already finalized; skipping",
                run.run_id,
            )
            return

        # Step 2: journal upload (best-effort; uploader swallows its own errors).
        if journal_path is not None and journal_sha256:
            try:
                self._journal_uploader.upload(
                    run.run_id, journal_path, journal_sha256
                )
                logger.info(
                    "[MLFLOW_FINALIZER] journal uploaded run=%s sha256=%s",
                    run.run_id,
                    journal_sha256[:12],
                )
            except Exception as exc:  # noqa: BLE001 -- never raise
                logger.warning(
                    "[MLFLOW_FINALIZER] journal upload failed run=%s: %s",
                    run.run_id,
                    exc,
                )

        # Step 3: lifecycle tags.
        try:
            self._client.set_tags(
                run.run_id,
                {
                    TagKey.LIFECYCLE_FINALIZED.value: "true",
                    TagKey.LIFECYCLE_STATUS.value: status.value,
                    TagKey.EXIT_REASON.value: exit_reason or "",
                },
            )
            logger.info(
                "[MLFLOW_FINALIZER] lifecycle tags set run=%s status=%s "
                "exit_reason=%s",
                run.run_id,
                status.value,
                exit_reason or "",
            )
        except Exception as exc:  # noqa: BLE001 -- never raise
            logger.warning(
                "[MLFLOW_FINALIZER] set_tags failed run=%s: %s",
                run.run_id,
                exc,
            )

        # Step 4: terminate.
        try:
            self._client.set_terminated(run.run_id, status)
            logger.info(
                "[MLFLOW_FINALIZER] terminated run=%s status=%s",
                run.run_id,
                status.value,
            )
        except Exception as exc:  # noqa: BLE001 -- never raise
            logger.warning(
                "[MLFLOW_FINALIZER] set_terminated failed run=%s: %s",
                run.run_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _already_finalized(self, run_id: str) -> bool:
        """Return ``True`` iff the run has already been finalized.

        A read failure is treated as "not finalized" so the caller
        re-attempts the close path; double-finalization on the
        server side is harmless because :meth:`ITrackingClient.set_tags`
        is idempotent on the value and ``set_terminated`` is a no-op
        on an already-terminal run.
        """
        try:
            handle = self._run_query.get_run(run_id)
        except Exception as exc:  # noqa: BLE001 -- treat as "not finalized"
            logger.debug(
                "[MLFLOW_FINALIZER] get_run failed during idempotency check "
                "run=%s (will attempt close): %s",
                run_id,
                exc,
            )
            return False
        tags = getattr(handle, "tags", None)
        if not isinstance(tags, dict):
            return False
        return tags.get(TagKey.LIFECYCLE_FINALIZED.value) == "true"
