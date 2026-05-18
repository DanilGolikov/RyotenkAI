"""Single-shot opener for root and nested attempt MLflow runs.

Encapsulates the two ``ITrackingClient.start_run`` call sites used by
the long-running pipeline:

* :meth:`ParentRunOpener.open` -- opens the **root** run for the
  logical pipeline run id (one per pipeline invocation; survives all
  attempt retries).
* :meth:`ParentRunOpener.open_attempt` -- opens a **nested attempt
  child** of the root for each retry.
* :meth:`ParentRunOpener.adopt_root` -- resume path: re-attach to an
  existing root run instead of opening a new one.

Tag policy
----------

All ``ryotenkai.lineage.*`` and ``ryotenkai.lifecycle.opened_by`` tags
are passed as the ``tags=`` argument to :meth:`ITrackingClient.start_run`
so they hit the server **as part of the run creation** -- there is no
second ``set_tags`` round-trip. This:

* Removes the legacy double-write ``log_pipeline_config`` /
  ``log_dataset_config`` pattern that wrote tags both on the attempt
  and on the trainer run.
* Guarantees the run is *never* visible without the lineage tags --
  there is no window between create and tag in which a reader could
  observe an untagged run.

Every key passes through :class:`ReservedPrefixGuard` before being
sent to the transport. ``mlflow.*`` keys without an explicit whitelist
entry raise immediately (programming error), preventing accidental
collisions with MLflow system tags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_shared.infrastructure.mlflow.taxonomy import (
    ReservedPrefixGuard,
    TagKey,
)
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from ryotenkai_shared.infrastructure.mlflow.protocols import ITrackingClient
    from ryotenkai_shared.infrastructure.mlflow.run_handle import RunHandle


logger = get_logger(__name__)


__all__ = ["ParentRunOpener"]


class ParentRunOpener:
    """Open / adopt root and attempt MLflow runs with required tags.

    :param client: Concrete :class:`ITrackingClient`.
    :param opened_by: ``host:user`` identifier stamped onto the
        :data:`~.taxonomy.TagKey.LIFECYCLE_OPENED_BY` tag on every
        opened root. Callers should pre-compute this once at process
        start; the opener does NOT introspect ``os.getlogin`` /
        ``socket.gethostname`` itself so the construction is testable
        and there are no surprise FQDN lookups inside the hot path.
    """

    def __init__(
        self,
        client: ITrackingClient,
        *,
        opened_by: str,
    ) -> None:
        if not opened_by:
            msg = "opened_by must be a non-empty 'host:user' identifier."
            raise ValueError(msg)
        self._client = client
        self._opened_by = opened_by

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(
        self,
        *,
        experiment: str,
        logical_run_id: str,
        config_sha256: str,
        code_commit: str,
        engine_kind: str,
        provider_kind: str,
        provider_gpu: str,
    ) -> RunHandle:
        """Open the root run for ``logical_run_id``.

        :param experiment: MLflow experiment name.
        :param logical_run_id: Pipeline-scoped identifier (e.g.
            ``"pipeline-2026-05-18T12-00-00"``); also stamped as
            :data:`~.taxonomy.TagKey.LINEAGE_RUN_ID`.
        :param config_sha256: SHA-256 hex digest of the canonical
            config file used to launch the pipeline; stamped as
            :data:`~.taxonomy.TagKey.LINEAGE_CONFIG_SHA256`.
        :param code_commit: Git SHA stamped as
            :data:`~.taxonomy.TagKey.LINEAGE_CODE_COMMIT`.
        :param engine_kind: One of ``sft|dpo|grpo|sapo``; stamped as
            :data:`~.taxonomy.TagKey.ENGINE_KIND`.
        :param provider_kind: Provider class name (e.g.
            ``"single_node"``, ``"runpod"``).
        :param provider_gpu: Provider GPU SKU (e.g. ``"H100-80GB"``).
        :returns: Frozen :class:`RunHandle` for the new root run.
        """
        tags = self._build_root_tags(
            logical_run_id=logical_run_id,
            config_sha256=config_sha256,
            code_commit=code_commit,
            engine_kind=engine_kind,
            provider_kind=provider_kind,
            provider_gpu=provider_gpu,
        )
        handle = self._client.start_run(
            experiment=experiment,
            name=logical_run_id,
            tags=tags,
            params={},
        )
        logger.info(
            "[MLFLOW_OPENER] opened root run_id=%s experiment=%s logical_run_id=%s "
            "engine=%s provider=%s/%s",
            handle.run_id,
            experiment,
            logical_run_id,
            engine_kind,
            provider_kind,
            provider_gpu,
        )
        return handle

    def open_attempt(
        self,
        *,
        root_run: RunHandle,
        logical_run_id: str,
        attempt_id: str,
        attempt_no: int,
    ) -> RunHandle:
        """Open a nested attempt run under ``root_run``.

        :param root_run: Parent :class:`RunHandle` returned by
            :meth:`open` or :meth:`adopt_root`.
        :param logical_run_id: Same identifier used for the root.
        :param attempt_id: Stable id for the retry (e.g. UUID4 hex);
            stamped as :data:`~.taxonomy.TagKey.ATTEMPT_ID`.
        :param attempt_no: 1-based ordinal of the retry; stamped as
            :data:`~.taxonomy.TagKey.ATTEMPT_NO`.
        :returns: Frozen :class:`RunHandle` for the attempt run.
        :raises ValueError: If ``attempt_no`` is not >= 1.
        """
        if attempt_no < 1:
            msg = f"attempt_no must be >= 1, got {attempt_no!r}"
            raise ValueError(msg)
        if not attempt_id:
            msg = "attempt_id must be non-empty."
            raise ValueError(msg)
        tags = self._build_attempt_tags(
            attempt_id=attempt_id,
            attempt_no=attempt_no,
        )
        handle = self._client.start_nested_run(
            parent_run_id=root_run.run_id,
            name=f"{logical_run_id}_attempt_{attempt_no}",
            tags=tags,
        )
        logger.info(
            "[MLFLOW_OPENER] opened attempt run_id=%s parent=%s attempt_no=%d "
            "attempt_id=%s",
            handle.run_id,
            root_run.run_id,
            attempt_no,
            attempt_id,
        )
        return handle

    def adopt_root(self, root_run_id: str) -> RunHandle:
        """Re-attach to an existing root run (resume path).

        Does NOT stamp lineage tags -- they were set on the original
        :meth:`open` call. The finalizer's idempotency check ensures
        we do not re-finalize an already-closed root.

        :param root_run_id: MLflow run id of the existing root.
        :returns: Frozen :class:`RunHandle`.
        """
        if not root_run_id:
            msg = "root_run_id must be non-empty."
            raise ValueError(msg)
        handle = self._client.adopt_run(root_run_id)
        logger.info(
            "[MLFLOW_OPENER] adopted existing root run_id=%s",
            handle.run_id,
        )
        return handle

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_root_tags(
        self,
        *,
        logical_run_id: str,
        config_sha256: str,
        code_commit: str,
        engine_kind: str,
        provider_kind: str,
        provider_gpu: str,
    ) -> dict[str, str]:
        """Assemble the full ``ryotenkai.*`` tag bundle for a root run.

        Every key is passed through :class:`ReservedPrefixGuard` so a
        future refactor that adds an ``mlflow.*`` key by mistake fails
        loudly here, not silently on the server.
        """
        tags: dict[str, str] = {
            TagKey.LINEAGE_PIPELINE_ID.value: logical_run_id,
            TagKey.LINEAGE_RUN_ID.value: logical_run_id,
            TagKey.LINEAGE_CONFIG_SHA256.value: config_sha256,
            TagKey.LINEAGE_CODE_COMMIT.value: code_commit,
            TagKey.LIFECYCLE_OPENED_BY.value: self._opened_by,
            TagKey.ENGINE_KIND.value: engine_kind,
            TagKey.PROVIDER_KIND.value: provider_kind,
            TagKey.PROVIDER_GPU.value: provider_gpu,
        }
        for key in tags:
            ReservedPrefixGuard.assert_safe(key)
        return tags

    def _build_attempt_tags(
        self,
        *,
        attempt_id: str,
        attempt_no: int,
    ) -> dict[str, str]:
        """Assemble the attempt-level tag bundle.

        Lineage tags propagate from the parent through MLflow's
        own search; we only stamp the attempt-scoped extras here so
        downstream queries can filter by ``ryotenkai.attempt.no``.
        """
        tags: dict[str, str] = {
            TagKey.ATTEMPT_ID.value: attempt_id,
            TagKey.ATTEMPT_NO.value: str(attempt_no),
        }
        for key in tags:
            ReservedPrefixGuard.assert_safe(key)
        return tags
