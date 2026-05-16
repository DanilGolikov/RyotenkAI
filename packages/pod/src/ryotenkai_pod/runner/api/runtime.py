"""``POST /api/v1/runtime/import-check`` — verify modules importable.

Phase 2 PR-2.5 of transport-unification-v2 — replaces the SSH
``python3 /opt/helix/runtime_check.py --check-source`` invocation
the Mac-side :class:`CodeSyncer` used after rsync.

Subprocess isolation: each requested module is verified by spawning
``python -c "import X"`` with a per-module timeout. Heavy imports
(torch + CUDA, transformers) don't get loaded into the long-running
uvicorn process — necessary because the runner is supposed to stay
small (RAM headroom for the trainer subprocess) and idempotent
(re-import-checking module N twice in a session is a no-op).
"""

from __future__ import annotations

import re
import subprocess
import sys

from fastapi import APIRouter

from ryotenkai_shared.api.error_handlers import APIError
from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.contracts.runner_api.runtime import (
    MAX_MODULES_PER_REQUEST,
    ImportCheckReport,
    ImportCheckRequest,
    ImportResult,
)

router = APIRouter(prefix="/runtime", tags=["runtime"])


# Module name pattern — same as Python identifiers separated by dots.
# Anchored so an attacker can't smuggle anything past the regex.
_MODULE_NAME_RE = re.compile(r"\A[a-z_][a-z_0-9]*(\.[a-z_][a-z_0-9]*)*\Z", re.IGNORECASE)

# Per-module subprocess timeout. Cold ``import torch`` can take a
# few seconds on a fresh pod; 30 s is generous.
_PER_MODULE_TIMEOUT_S = 30.0


def _validate_module_name(name: str) -> None:
    if not _MODULE_NAME_RE.fullmatch(name):
        raise APIError(
            ErrorCode.IMPORT_CHECK_INVALID_MODULE_NAME, status=422,
            detail=(
                f"module name {name!r} does not match "
                f"[a-z_.][a-z_0-9.]* — refusing to subprocess-import."
            ),
        )


def _check_one_module(name: str) -> ImportResult:
    """Run ``python -c "import {name}"`` and classify the outcome.

    Stdout is ignored; stderr's first line carries the
    ``ExceptionType: message`` for the operator. Any subprocess
    error (``OSError``, missing python) maps to ``importable=False``
    with a synthetic ``error`` message — the endpoint stays 200 and
    surfaces the failure inline.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-c", f"import {name}"],
            capture_output=True,
            text=True,
            timeout=_PER_MODULE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ImportResult(
            module=name,
            importable=False,
            error=f"TimeoutExpired (>{_PER_MODULE_TIMEOUT_S}s)",
        )
    except OSError as exc:
        return ImportResult(
            module=name, importable=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    if completed.returncode == 0:
        return ImportResult(module=name, importable=True)

    # LAST non-empty line of stderr is the canonical exception
    # spelling Python emits; e.g. ``ModuleNotFoundError: No module
    # named 'foo'``. The first stderr line is just the
    # ``Traceback (most recent call last):`` framing.
    stderr_lines = [line.strip() for line in (completed.stderr or "").splitlines() if line.strip()]
    error_line = stderr_lines[-1] if stderr_lines else "subprocess returned non-zero with no stderr"
    return ImportResult(module=name, importable=False, error=error_line)


@router.post("/import-check", response_model=ImportCheckReport)
def check_imports(request: ImportCheckRequest) -> ImportCheckReport:
    """Verify each module in ``request.modules`` is importable on the
    pod's PYTHONPATH.

    Errors:
    * 422 ``IMPORT_CHECK_TOO_MANY_MODULES`` — > MAX_MODULES_PER_REQUEST.
    * 422 ``IMPORT_CHECK_INVALID_MODULE_NAME`` — name fails regex.
    * 200 with ``ImportResult.importable=False`` — module failed to
      import; client decides what to do (CodeSyncer halts pipeline).
    """
    if len(request.modules) > MAX_MODULES_PER_REQUEST:
        raise APIError(
            ErrorCode.IMPORT_CHECK_TOO_MANY_MODULES, status=422,
            detail=(
                f"got {len(request.modules)} modules, max is "
                f"{MAX_MODULES_PER_REQUEST}"
            ),
        )

    for name in request.modules:
        _validate_module_name(name)

    results = [_check_one_module(name) for name in request.modules]
    return ImportCheckReport(results=results)


__all__ = ["router"]
