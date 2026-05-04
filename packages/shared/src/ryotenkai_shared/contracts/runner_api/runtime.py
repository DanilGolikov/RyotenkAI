"""Runtime import-check DTOs (Phase 2 PR-2.5).

``POST /api/v1/runtime/import-check`` — verify a list of Python
modules is importable inside the pod runtime, replacing the SSH
``runtime_check.py --check-source`` invocation the Mac-side
:class:`CodeSyncer` used.

Per-module subprocess isolation: each module is verified in its
own ``python -c "import X"`` subprocess so a heavy import (torch
loading CUDA stubs) doesn't pollute the long-running uvicorn
runner. Slower than in-process import for many modules, but
deterministic and isolation-safe.
"""

from __future__ import annotations

from pydantic import Field

from ._strict import _StrictModel

# Cap on how many modules a single request may carry. Hard limit so
# a misbehaving Mac client can't pin the pod's CPU on subprocess
# fanout. Production usage is ~6 modules from CodeSyncer's required
# list; 50 is a generous safety margin.
MAX_MODULES_PER_REQUEST = 50


class ImportCheckRequest(_StrictModel):
    """List of fully-qualified Python module names to verify.

    Validators:

    * ``MAX_MODULES_PER_REQUEST`` cap — 422 ``IMPORT_CHECK_TOO_MANY_MODULES``.
    * ``[a-z_.][a-z_0-9.]*`` regex per module name — 422
      ``IMPORT_CHECK_INVALID_MODULE_NAME``. Rejects shell injection
      attempts (``"os.system('rm -rf /')"`` etc.) — defence in
      depth even though the runner runs each in subprocess with
      ``-c "import X"`` (not eval).
    """

    modules: list[str] = Field(min_length=1)


class ImportResult(_StrictModel):
    """Outcome for one module.

    ``importable=True`` ⇒ subprocess returned 0 + nothing else;
    ``False`` ⇒ subprocess raised an :class:`ImportError` or other
    exception, which the response carries in ``error`` for the
    operator to read.
    """

    module: str
    importable: bool
    error: str | None = Field(
        default=None,
        description=(
            "When ``importable=False``: ``ExceptionType: message`` "
            "string from the subprocess stderr. Stripped to the "
            "first line for compactness."
        ),
    )


class ImportCheckReport(_StrictModel):
    """Per-module verification report."""

    results: list[ImportResult]

    @property
    def all_importable(self) -> bool:
        return all(r.importable for r in self.results)

    @property
    def failed(self) -> list[str]:
        return [r.module for r in self.results if not r.importable]


__all__ = [
    "MAX_MODULES_PER_REQUEST",
    "ImportCheckReport",
    "ImportCheckRequest",
    "ImportResult",
]
