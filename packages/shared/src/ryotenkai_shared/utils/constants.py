"""Shared constants for utility modules (docker, ssh_client, etc.)."""

from __future__ import annotations

# Output truncation lengths (characters) — used when slicing stderr/stdout for logs/errors.
# "Short" is for brief Err() messages; "long" is for detailed diagnostic output.
LOG_OUTPUT_SHORT_CHARS = 200
LOG_OUTPUT_LONG_CHARS = 500
