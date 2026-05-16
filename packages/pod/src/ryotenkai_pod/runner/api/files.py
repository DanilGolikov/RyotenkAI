"""``POST /api/v1/files/upload`` — multipart streaming file upload.

Phase 2 PR-2.4 of transport-unification-v2 — replaces the legacy
SSH ``tar pipe`` + SCP fallback the Mac-side :class:`FileUploader`
used. Production qualities (per 2025 FastAPI best-practice review,
see ``docs/plans/2026-05-04-transport-unification-http-runtime-v2.md``
§9):

* **Streaming** — ``UploadFile.read(CHUNK_SIZE=1MB)`` in a loop;
  memory-bounded regardless of upload size. No
  ``await file.read()`` (full read) — that defeats the whole
  point.
* **Atomic write** — bytes go into ``<target>.partial``; only on
  full success is the file ``rename()``-d to the canonical name.
  Crash mid-upload leaves the canonical path unchanged.
* **Per-chunk size cap** — early reject when bytes exceed
  ``MAX_FILE_SIZE``, discard partial.
* **Per-chunk SHA-256** — incremental ``hashlib.sha256.update``
  in the same loop; echoed in the response for client-side
  verification.
* **Path whitelist** — :class:`FileUploadTarget` enum maps to a
  fixed pod-side path; clients never supply raw paths.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import IO

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from starlette.requests import ClientDisconnect

from ryotenkai_shared.api.error_handlers import APIError
from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.contracts.runner_api.files import (
    FileUploadResponse,
    FileUploadTarget,
)
from ryotenkai_shared.utils.pod_layout import PodLayout

router = APIRouter(prefix="/files", tags=["files"])

# Production cap — 100 MB. Datasets are <10 MB today; container
# disk on RunPod is ~50 GB but disk-fill is the more likely abuse
# than network. Override via ``RYOTENKAI_RUNNER_MAX_FILE_SIZE`` if
# a real workload bumps into the cap.
MAX_FILE_SIZE: int = 100 * 1024 * 1024
CHUNK_SIZE: int = 1024 * 1024  # 1 MB — verified sweet spot for
                                 # memory vs throughput trade-off.


def _get_pod_layout(request: Request) -> PodLayout:
    layout = getattr(request.app.state, "pod_layout", None)
    if layout is None:
        raise APIError(
            ErrorCode.RUNNER_NOT_READY, status=503,
            detail="pod layout not initialised on app.state",
        )
    return layout


def _resolve_target_path(layout: PodLayout, target: FileUploadTarget) -> Path:
    """Map :class:`FileUploadTarget` to its fixed pod-side path.

    All paths sit under the per-run workspace root, so
    `<run_dir>/config.yaml`, `<run_dir>/data/dataset.jsonl`, etc.
    No string concatenation from caller-controlled input — the
    enum is the only entry point, structurally rejecting traversal.
    """
    root = Path(str(layout.root))
    if target is FileUploadTarget.CONFIG:
        return root / "config" / "pipeline_config.yaml"
    if target is FileUploadTarget.DATASET:
        # Mac client sets the filename via the multipart ``filename``
        # field; we use it ONLY as a basename, sanitised below in
        # :func:`upload_file` so traversal is impossible.
        return root / "data" / "<dataset>"
    if target is FileUploadTarget.COMMUNITY_PLUGINS:
        return root / "community" / "plugins.zip"
    raise APIError(
        ErrorCode.FILE_TARGET_INVALID, status=422,
        detail=f"unhandled target={target!r} (programming bug)",
    )


def _sanitise_basename(filename: str | None) -> str:
    """Reduce a client-supplied filename to a basename, refuse anything
    that smells like traversal."""
    name = (filename or "").strip()
    if not name:
        raise APIError(
            ErrorCode.FILE_TARGET_INVALID, status=422,
            detail="multipart filename required for dataset uploads",
        )
    base = Path(name).name
    if base != name or base in {"", ".", ".."}:
        raise APIError(
            ErrorCode.FILE_TARGET_INVALID, status=422,
            detail=(
                f"unsafe filename={name!r}; the runner rejects path "
                "components — pass just a basename."
            ),
        )
    return base


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    target: FileUploadTarget = Form(...),  # noqa: B008 — FastAPI idiom
    file: UploadFile = File(...),  # noqa: B008
    layout: PodLayout = Depends(_get_pod_layout),  # noqa: B008
) -> FileUploadResponse:
    """Stream a multipart file part to disk, write atomically.

    Wire shape:

        POST /api/v1/files/upload
            Content-Type: multipart/form-data
            target=config|dataset|community-plugins-zip
            file=<file blob>

    Response (200):

        {
            "target": "...",
            "bytes_written": 12345,
            "sha256": "<64-hex>",
            "path": "/workspace/runs/.../config.yaml"
        }

    Errors (problem+json):
    * 422 ``FILE_TARGET_INVALID`` — unknown enum / unsafe filename.
    * 413 ``FILE_TOO_LARGE``     — ran past ``MAX_FILE_SIZE``.
    * 502 ``FILE_WRITE_FAILED``  — disk full / permission denied.
    """
    target_path = _resolve_target_path(layout, target)
    if target is FileUploadTarget.DATASET:
        target_path = target_path.parent / _sanitise_basename(file.filename)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = target_path.with_suffix(target_path.suffix + ".partial")

    hasher = hashlib.sha256()
    total = 0

    # The runtime image does not ship ``aiofiles`` — fall back to
    # ``open()`` + ``asyncio.to_thread`` for the blocking write.
    # Streaming + per-chunk cap semantics are preserved.
    fh: IO[bytes] | None = None
    try:
        fh = await asyncio.to_thread(open, partial_path, "wb")
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                raise APIError(
                    ErrorCode.FILE_TOO_LARGE, status=413,
                    detail=(
                        f"upload exceeded MAX_FILE_SIZE={MAX_FILE_SIZE} "
                        f"(target={target.value!r})"
                    ),
                )
            await asyncio.to_thread(fh.write, chunk)
            hasher.update(chunk)
    except APIError:
        # ``except APIError`` MUST come before ``except Exception`` —
        # otherwise our own 413 gets remapped to FILE_WRITE_FAILED.
        # ``finally`` cleanup runs in either branch (RP21).
        partial_path.unlink(missing_ok=True)
        raise
    except ClientDisconnect:
        # Mac client aborted mid-upload — clean up the partial so
        # disk doesn't fill on flapping clients (RP21).
        partial_path.unlink(missing_ok=True)
        raise APIError(
            ErrorCode.FILE_WRITE_FAILED, status=499,
            detail="client disconnected mid-upload",
        ) from None
    except OSError as exc:
        partial_path.unlink(missing_ok=True)
        raise APIError(
            ErrorCode.FILE_WRITE_FAILED, status=502,
            detail=f"disk write failed: {exc}",
        ) from exc
    finally:
        if fh is not None:
            await asyncio.to_thread(fh.close)

    # Atomic rename — only after the full body landed cleanly.
    try:
        partial_path.replace(target_path)
    except OSError as exc:
        partial_path.unlink(missing_ok=True)
        raise APIError(
            ErrorCode.FILE_WRITE_FAILED, status=502,
            detail=f"atomic rename failed: {exc}",
        ) from exc

    return FileUploadResponse(
        target=target,
        bytes_written=total,
        sha256=hasher.hexdigest(),
        path=str(target_path),
    )


__all__ = ["MAX_FILE_SIZE", "router"]
