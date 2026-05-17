"""Upload ``events.jsonl`` + manifest to MLflow at run finalize (Phase 6.a).

Algorithm (run finally block):

1. Orchestrator's :meth:`ControlEventEmitter.close` flushed the journal
   (final fsync). The journal file is now fully written.
2. :meth:`MlflowFinalizer.upload` opens the journal via
   :class:`JournalReader`, streams every envelope to compute:

   * ``total_events``
   * ``first_offset_per_source`` / ``last_offset_per_source``
   * ``first_time`` / ``last_time`` (ISO-8601 with ``Z`` suffix)
   * ``type_histogram`` keyed on the envelope's ``kind`` discriminator
   * ``schema_versions_present``
   * ``events_sha256`` — SHA-256 over the raw bytes of the journal file

3. Writes ``events_manifest.json`` next to the journal so a later manual
   upload can use it without recomputing.
4. Uploads ``events.jsonl`` and ``events_manifest.json`` to MLflow under
   the ``events/`` artifact path, with exponential backoff retry
   (1s/5s/30s — 3 attempts total).
5. On retry exhaustion: rewrites the manifest with
   ``mlflow_uploaded=false`` and returns ``False``. The journal stays on
   disk; the workspace retention policy is expected to skip cleanup when
   ``mlflow_uploaded=false`` (Phase 6.a only writes the flag; retention
   coordination ships separately).

Cancellation:

* The caller passes ``cancellation_reason`` (a string) and
  ``journal_complete=False`` when the run was cancelled. The manifest
  carries these verbatim so consumers can reason about the partial
  artifact.

Contracts:

* Returns ``True`` on full success (both artifacts uploaded).
* Returns ``False`` when any retry budget is exhausted.
* NEVER raises — orchestrator's ``finally`` block must not crash on a
  flaky MLflow.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from typing import TYPE_CHECKING, Any

from ryotenkai_control.events.journal_reader import JournalReader
from ryotenkai_shared.events import UNKNOWN_OFFSET
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from pathlib import Path

    from ryotenkai_shared.infrastructure.mlflow.protocol import IMLflowManager


logger = get_logger(__name__)


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_RETRY_DELAYS_S",
    "MANIFEST_FILENAME",
    "MlflowFinalizer",
]


DEFAULT_RETRY_DELAYS_S: tuple[float, ...] = (1.0, 5.0, 30.0)
DEFAULT_ARTIFACT_PATH = "events"
MANIFEST_FILENAME = "events_manifest.json"
MANIFEST_SCHEMA_VERSION = 1


class MlflowFinalizer:
    """Compute the events manifest and upload journal + manifest to MLflow."""

    def __init__(
        self,
        mlflow_manager: IMLflowManager,
        *,
        retry_delays_s: tuple[float, ...] = DEFAULT_RETRY_DELAYS_S,
        artifact_path: str = DEFAULT_ARTIFACT_PATH,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._mlflow = mlflow_manager
        self._retry_delays = retry_delays_s
        self._artifact_path = artifact_path
        # Tests inject a fake sleep so retry timing is deterministic.
        self._sleep = sleep or time.sleep

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def upload(
        self,
        run_id: str,
        journal_path: Path,
        *,
        cancellation_reason: str | None = None,
        journal_complete: bool = True,
    ) -> bool:
        """Build manifest, upload both artifacts. Returns True on success.

        ``run_id`` is the MLflow run id (used as ``run_id`` argument to
        :meth:`IMLflowManager.log_artifact`); ``journal_path`` is the
        absolute path to ``events.jsonl``.
        """
        try:
            manifest = self._build_manifest(
                journal_path=journal_path,
                cancellation_reason=cancellation_reason,
                journal_complete=journal_complete,
            )
        except Exception as exc:
            logger.warning(
                "[MlflowFinalizer] failed to build manifest for %s: %s: %s",
                journal_path,
                type(exc).__name__,
                exc,
            )
            return False

        manifest_path = journal_path.parent / MANIFEST_FILENAME
        # Write the manifest first — even if upload fails we leave a
        # local on-disk record consumers can rely on.
        try:
            self._write_manifest(manifest_path, manifest)
        except Exception as exc:
            logger.warning(
                "[MlflowFinalizer] failed to write local manifest %s: %s: %s",
                manifest_path,
                type(exc).__name__,
                exc,
            )
            return False

        if not journal_path.exists():
            logger.warning(
                "[MlflowFinalizer] journal missing at upload time: %s",
                journal_path,
            )
            manifest["mlflow_uploaded"] = False
            with contextlib.suppress(Exception):
                self._write_manifest(manifest_path, manifest)
            return False

        success = self._upload_with_retry(
            run_id=run_id,
            journal_path=journal_path,
            manifest_path=manifest_path,
        )
        manifest["mlflow_uploaded"] = success
        # Rewrite manifest so the on-disk artifact reflects the final
        # outcome — downstream retention/cleanup uses this flag to
        # decide whether to keep the journal around for a manual
        # upload.
        with contextlib.suppress(Exception):
            self._write_manifest(manifest_path, manifest)
        return success

    # ------------------------------------------------------------------
    # Manifest construction
    # ------------------------------------------------------------------

    def _build_manifest(
        self,
        *,
        journal_path: Path,
        cancellation_reason: str | None,
        journal_complete: bool,
    ) -> dict[str, Any]:
        total = 0
        first_offset_per_source: dict[str, int] = {}
        last_offset_per_source: dict[str, int] = {}
        first_time: datetime | None = None
        last_time: datetime | None = None
        type_histogram: dict[str, int] = {}
        schema_versions: set[int] = set()

        if journal_path.exists():
            reader = JournalReader(journal_path)
            for envelope in reader.iter_envelopes():
                # Skip torn-write residue — UnknownEvent with
                # ``offset=UNKNOWN_OFFSET`` carries no usable
                # bookkeeping data.
                if envelope.offset == UNKNOWN_OFFSET:
                    continue
                total += 1
                src = envelope.source
                off = envelope.offset
                if src not in first_offset_per_source or off < first_offset_per_source[src]:
                    first_offset_per_source[src] = off
                if src not in last_offset_per_source or off > last_offset_per_source[src]:
                    last_offset_per_source[src] = off
                if first_time is None or envelope.time < first_time:
                    first_time = envelope.time
                if last_time is None or envelope.time > last_time:
                    last_time = envelope.time
                kind = envelope.kind
                type_histogram[kind] = type_histogram.get(kind, 0) + 1
                schema_versions.add(int(envelope.schema_version))

        events_sha256 = _file_sha256(journal_path)

        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "total_events": total,
            "first_offset_per_source": first_offset_per_source,
            "last_offset_per_source": last_offset_per_source,
            "first_time": _to_iso_z(first_time),
            "last_time": _to_iso_z(last_time),
            "type_histogram": type_histogram,
            "schema_versions_present": sorted(schema_versions),
            "events_sha256": events_sha256,
            "journal_complete": journal_complete,
            "cancellation_reason": cancellation_reason,
            # ``mlflow_uploaded`` is filled by :meth:`upload` after the
            # MLflow round trip; the initial manifest carries the
            # optimistic default so an early local read still has the
            # key present.
            "mlflow_uploaded": True,
        }
        return manifest

    @staticmethod
    def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Pretty-print so the artifact is grep-able. ``sort_keys=True``
        # keeps diffs stable across runs of the same fixture.
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Upload with retry
    # ------------------------------------------------------------------

    def _upload_with_retry(
        self,
        *,
        run_id: str,
        journal_path: Path,
        manifest_path: Path,
    ) -> bool:
        last_exc: Exception | None = None
        for attempt_idx, delay in enumerate(self._retry_delays):
            if attempt_idx > 0:
                # Sleep BEFORE the retry attempt (not after the first
                # try). The first delay value drives the sleep between
                # attempt 1's failure and attempt 2's start.
                with contextlib.suppress(Exception):
                    self._sleep(delay)
            try:
                self._mlflow.log_artifact(
                    str(journal_path),
                    self._artifact_path,
                    run_id=run_id,
                )
                self._mlflow.log_artifact(
                    str(manifest_path),
                    self._artifact_path,
                    run_id=run_id,
                )
                return True
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[MlflowFinalizer] MLflow upload attempt %d/%d failed: "
                    "%s: %s",
                    attempt_idx + 1,
                    len(self._retry_delays),
                    type(exc).__name__,
                    exc,
                )
        logger.warning(
            "[MlflowFinalizer] all %d upload attempts exhausted for run_id=%s; "
            "manifest will record mlflow_uploaded=false (last error: %s)",
            len(self._retry_delays),
            run_id,
            last_exc,
        )
        return False


def _file_sha256(path: Path) -> str:
    """Return hex SHA-256 of ``path``, or sha256 of empty bytes when missing.

    Streams in 64 KiB chunks — the journal can be hundreds of megabytes
    on a long run; loading it into memory would dwarf MLflow's own
    upload buffer.
    """
    h = hashlib.sha256()
    if not path.exists():
        return h.hexdigest()
    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(64 * 1024)
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        # Mirror the "never raises" contract; return the partial hash so
        # consumers see something deterministic.
        return h.hexdigest()
    return h.hexdigest()


def _to_iso_z(value: datetime | None) -> str | None:
    """Format ``value`` as ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` or ``None``.

    The manifest never persists a naive datetime — events carry
    timezone-aware UTC values by envelope contract, so the conversion
    here is purely a serialization step.
    """
    if value is None:
        return None
    # ``isoformat()`` on a tz-aware datetime emits ``+00:00`` for UTC;
    # we normalize to the trailing ``Z`` form used everywhere else in
    # the manifest schema.
    iso = value.isoformat()
    if iso.endswith("+00:00"):
        iso = iso[:-6] + "Z"
    return iso
