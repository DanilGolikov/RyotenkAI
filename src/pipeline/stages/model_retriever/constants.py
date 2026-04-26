"""Constants for the ModelRetriever stage (HF download + SSH upload helpers)."""

from __future__ import annotations

from src.constants import HF_UPLOAD_TIMEOUT_S, SSH_CMD_TIMEOUT, SSH_PORT_DEFAULT

# HuggingFace HTTP error codes that the retriever inspects for friendly
# fail-fast messages.
HTTP_STATUS_NOT_FOUND = 404
HTTP_STATUS_UNAUTHORIZED = 401

# How long an HF metadata fetch result stays cached (seconds). Keeps
# repeated dry-runs cheap without breaking real upload flows.
HF_CACHE_TTL = 1800

# SSH defaults reused by the retriever's remote-side calls.
MR_SSH_PORT_DEFAULT = SSH_PORT_DEFAULT
MR_SSH_CMD_TIMEOUT = SSH_CMD_TIMEOUT

# Length of the short commit SHA we surface in logs / artifact metadata.
MR_SHA12_LENGTH = 12

# Upload timeout — re-exported under the stage's preferred name; the
# underlying value lives in src.constants because other infrastructure
# (run_training, providers) shares it.
MR_UPLOAD_TIMEOUT = HF_UPLOAD_TIMEOUT_S


__all__ = [
    "HF_CACHE_TTL",
    "HTTP_STATUS_NOT_FOUND",
    "HTTP_STATUS_UNAUTHORIZED",
    "MR_SHA12_LENGTH",
    "MR_SSH_CMD_TIMEOUT",
    "MR_SSH_PORT_DEFAULT",
    "MR_UPLOAD_TIMEOUT",
]
