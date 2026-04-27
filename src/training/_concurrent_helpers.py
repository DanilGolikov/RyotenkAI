"""Phase 9.B — small concurrency helpers for the training subprocess.

The standalone home is intentional: this is a 30-line utility module
(no ML deps, no orchestrator imports) that callers across
``src/training/`` can pull in without dragging the heavy training
package init. Importing it from :mod:`src.training._concurrent_helpers`
is safe inside HF callbacks and inside slim-venv tests.

Why not :mod:`signal.alarm`?
----------------------------

The original Phase 9 plan (v1) reached for ``signal.alarm`` for the
hard 5-second budget around the MLflow finalize path. That path is
brittle:

* Unix-only — ``signal.alarm`` is not available on Windows.
* Main-thread-only — Python's signal module rejects calls from
  worker threads, which breaks pytest workers running tests in
  parallel.
* ``signal.alarm`` doesn't compose with ``asyncio`` event loops —
  the runner's main code path uses both freely.

``concurrent.futures.ThreadPoolExecutor`` + ``Future.result(timeout)``
gives us a portable, mock-friendly "run this and bail in N seconds"
primitive that survives every environment we care about.

Failure semantics
-----------------

When the deadline expires, the helper does NOT cancel the underlying
work — Python ``Future.cancel()`` cannot interrupt synchronous code
that's mid-flight (e.g. an MLflow HTTP call stuck on a socket
recv). The thread continues running until it returns or the process
exits. We only stop *waiting* for it.

Callers must therefore:

1. Treat ``TimeoutError`` as "we don't know whether the work
   succeeded, only that it didn't reply in time".
2. Side-effects committed by the work that lands AFTER the deadline
   still count — e.g. a buffered MLflow flush that finishes in 6s
   does land in MLflow even though the caller saw the timeout at 5s.
3. Avoid stacking calls that need ordering across timeouts. The
   callee must be self-consistent.

The helper is intentionally narrow — no kwargs forwarding, no
exception remapping. Anything richer belongs in the caller.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from typing import TypeVar


__all__ = ["TimeoutExceededError", "with_timeout"]


T = TypeVar("T")


class TimeoutExceededError(TimeoutError):
    """Raised by :func:`with_timeout` when the deadline expires.

    Subclasses :class:`TimeoutError` so generic ``except TimeoutError``
    handlers (e.g. :func:`asyncio.wait_for`'s) catch us too — but
    callers that want to distinguish "I timed this out via
    with_timeout" from "the network library timed out internally"
    can match on the more specific class.
    """


def with_timeout(
    fn: Callable[[], T],
    *,
    timeout_seconds: float,
) -> T:
    """Run ``fn`` in a single-worker thread pool; bail at deadline.

    The pool is created and torn down per call — the use case is
    "one-shot finalization with a hard budget", not a hot path.
    Per-call construction makes the function trivially testable and
    keeps the API surface a single function.

    Args:
        fn: Zero-arg callable. Any exceptions it raises propagate
            through unchanged (after the work completes within budget).
        timeout_seconds: Hard ceiling. Must be > 0 — values ≤ 0
            raise ValueError immediately so callers can't silently
            disable the budget.

    Returns:
        Whatever ``fn`` returned.

    Raises:
        TimeoutExceededError: when the deadline expires before
            ``fn`` returns. The underlying thread keeps running but
            is no longer awaited.
        ValueError: when ``timeout_seconds`` ≤ 0.
        Anything ``fn`` raises: re-raised on the calling thread.
    """
    if timeout_seconds <= 0:
        raise ValueError(
            f"timeout_seconds must be positive, got {timeout_seconds}",
        )

    # ``ThreadPoolExecutor`` is used as a context manager so the worker
    # thread is reaped on exit even if our timeout fires. Note: the
    # thread continues running until it returns naturally; ``shutdown``
    # waits for it. We accept that — the alternative (force-killing a
    # thread mid-syscall) is unsafe in CPython.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutExceededError(
                f"call exceeded {timeout_seconds}s budget",
            ) from exc
