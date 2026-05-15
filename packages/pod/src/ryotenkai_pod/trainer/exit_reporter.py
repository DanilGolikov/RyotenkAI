"""Trainer subprocess exit-payload reporter — Phase D.

Encapsulates the logic that turns a top-level trainer exception into a
:class:`~ryotenkai_shared.contracts.trainer_exit.TrainerExitPayload`
on disk.

Extracted from :mod:`run_training` so the mapping (Exception →
``code``/``message``/``traceback_summary``) and the atomic write can
be unit-tested without spawning a real subprocess.

Contract
--------

``write_failure_payload(workdir, exc, *, started_at, exit_code)``:

* If ``exc`` is a :class:`~ryotenkai_shared.errors.RyotenkAIError`
  subclass, use its pinned ``code`` + ``exc.detail or str(exc)`` for
  the message.
* Otherwise, fall back to ``code=INTERNAL_ERROR`` and ``message=
  str(exc)``.
* In both cases ``traceback_summary`` is the sanitised tail of
  :func:`traceback.format_exc`.
* The file is written atomically via the contract's :meth:`write_to`
  helper. Any IOError is suppressed by the caller (atexit must not
  raise harder than the original exception).

This module is intentionally tiny: callers wire it into a single
``try/except`` site, and tests stub the disk side via ``tmp_path``.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from ryotenkai_shared.contracts.problem_details import ErrorCode
from ryotenkai_shared.contracts.trainer_exit import (
    TRAINER_EXIT_FILENAME,
    TrainerExitPayload,
    sanitize_traceback,
)
from ryotenkai_shared.errors import RyotenkAIError

if TYPE_CHECKING:
    pass

__all__ = [
    "build_failure_payload",
    "write_failure_payload",
]


def build_failure_payload(
    exc: BaseException,
    *,
    started_at: float,
    exit_code: int,
    now: float | None = None,
) -> TrainerExitPayload:
    """Construct a :class:`TrainerExitPayload` for ``exc``.

    Pure (no IO) — easy to unit test. ``started_at`` is a
    :func:`time.monotonic` reading from trainer start; ``now`` defaults
    to current :func:`time.monotonic` (parameter exists so tests can
    pin elapsed wall time).
    """
    if now is None:
        now = time.monotonic()
    elapsed = max(0.0, now - started_at)

    # ``traceback.format_exc`` reads the currently-handled exception
    # from :mod:`sys`. We pass ``exc`` explicitly via
    # :func:`traceback.format_exception` so the helper works outside
    # a ``except:`` block too (matters for tests + for callers that
    # caught the exception, did some bookkeeping, then handed us the
    # captured value).
    tb_raw = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    tb_clean = sanitize_traceback(tb_raw) if tb_raw else None

    if isinstance(exc, RyotenkAIError):
        code = exc.code
        # ``RyotenkAIError`` carries detail or falls back to its title;
        # keep ``str(exc)`` if neither is informative (defensive — the
        # base class formats ``"{code}: {detail or title}"``).
        message = exc.detail or exc.title or str(exc)
    else:
        code = ErrorCode.INTERNAL_ERROR
        message = str(exc) or type(exc).__name__

    return TrainerExitPayload(
        code=code,
        message=message,
        traceback_summary=tb_clean,
        exit_code=exit_code,
        wall_seconds=elapsed,
    )


def write_failure_payload(
    workdir: Path | None,
    exc: BaseException,
    *,
    started_at: float,
    exit_code: int,
) -> Path | None:
    """Build + write the payload. Best-effort.

    Returns the path the payload was written to, or ``None`` if no
    workdir was supplied / the write failed. Any :class:`OSError` is
    swallowed: the atexit caller has already lost the trainer; we
    don't make the failure louder than necessary.

    The supervisor reads this file at
    ``<workdir>/<TRAINER_EXIT_FILENAME>`` so both sides agree on the
    location without an environment variable.
    """
    if workdir is None:
        return None
    workdir = Path(workdir)
    target = workdir / TRAINER_EXIT_FILENAME
    try:
        payload = build_failure_payload(
            exc, started_at=started_at, exit_code=exit_code,
        )
        payload.write_to(target)
        return target
    except (OSError, ValueError):
        # OSError on disk-full / permission; ValueError defensive
        # (e.g. Pydantic refusing a NaN wall_seconds). Either way we
        # silently drop — supervisor's exit-code heuristic kicks in.
        return None
