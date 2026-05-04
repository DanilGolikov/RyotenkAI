"""File upload DTOs (Phase 2 PR-2.4).

``POST /api/v1/files/upload`` — multipart streaming upload from Mac
to pod. Replaces the legacy tar-pipe + SCP protocol the Mac-side
:class:`FileUploader` used.

Anti-path-traversal: the endpoint accepts a closed
:class:`FileUploadTarget` enum, never a raw path. Each enum value
maps server-side (in :mod:`ryotenkai_pod.runner.api.files`) to a
fixed sub-path under the run workspace.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from ._strict import _StrictModel


class FileUploadTarget(StrEnum):
    """Whitelist of upload destinations.

    Each value resolves to a fixed pod-side path under the run
    workspace (see ``ryotenkai_pod.runner.api.files`` for the
    mapping). Unknown values trigger 422 ``FILE_TARGET_INVALID``.

    * ``CONFIG``           — pipeline_config.yaml (small, ~10 KB)
    * ``DATASET``           — JSONL dataset (≤10 MB typical)
    * ``COMMUNITY_PLUGINS`` — plugin ZIP bundle (≤5 MB typical)
    """

    CONFIG = "config"
    DATASET = "dataset"
    COMMUNITY_PLUGINS = "community-plugins-zip"


class FileUploadResponse(_StrictModel):
    """Echoed metadata for a successful upload.

    Mac client checks ``sha256`` against its own pre-upload hash
    when ``checksum`` was provided in the request, and uses
    ``bytes_written`` to confirm the streaming did not silently
    drop tail bytes (chunked write is faithful but confirming on
    the wire keeps the contract honest).
    """

    target: FileUploadTarget
    bytes_written: int = Field(ge=0)
    sha256: str = Field(
        min_length=64, max_length=64,
        description="Hex-encoded SHA-256 of the bytes received.",
    )
    path: str = Field(
        description=(
            "Resolved pod-side path the bytes were written to. "
            "Diagnostic only — clients must not parse this; the "
            "target enum is the contract."
        ),
    )


__all__ = [
    "FileUploadResponse",
    "FileUploadTarget",
]
