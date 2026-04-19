"""Backward-compatibility shim. Real module lives in src/pipeline/deletion.

TuiDeleteBackend is a deprecated alias for RunDeleter. Prefer the new name
in new code.
"""

from __future__ import annotations

from src.pipeline.deletion import DeleteIssue, DeleteMode, DeleteResult, RunDeleter

# Deprecated alias — kept for TUI and pre-rename tests.
TuiDeleteBackend = RunDeleter

__all__ = ["DeleteIssue", "DeleteMode", "DeleteResult", "RunDeleter", "TuiDeleteBackend"]
