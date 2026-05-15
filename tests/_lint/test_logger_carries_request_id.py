"""Sentinel: shared logger setup installs the ``RequestIDLogFilter``.

Phase C (sharded-stargazing-wigderson, 2026-05-16): the
``RequestIDMiddleware`` and the ``current_request_id`` reader are not
useful on their own -- the operator-facing benefit only materialises
when the per-request id flows from the contextvar into every log
record, so server logs and the response body share a correlation
id.

This sentinel guards two failure modes:

1. An agent edits ``shared/utils/logger.py`` and removes the
   ``addFilter(_REQUEST_ID_FILTER)`` lines (perhaps as part of a
   cleanup that misses the cross-cutting intent).
2. The filter is silently dropped from the import-time root
   attachment (e.g. by a future ``_seal_root_logger`` refactor).

The sentinel verifies the live state: after importing the logger
module, both the ``ryotenkai`` and the root logger must have the
filter attached. We test the *behaviour* (request_id is stamped on
records emitted while the contextvar is set), not just the
attribute, because attribute-only checks miss the case where the
filter is attached but broken.
"""

from __future__ import annotations

import logging

from ryotenkai_shared.api.request_id import REQUEST_ID
from ryotenkai_shared.utils.logger import (
    RequestIDLogFilter,
    logger as ryotenkai_logger,
)


def _has_request_id_filter(target: logging.Logger) -> bool:
    """Return True iff a RequestIDLogFilter is attached to ``target``."""
    return any(isinstance(f, RequestIDLogFilter) for f in target.filters)


def test_ryotenkai_logger_has_request_id_filter() -> None:
    """The ``ryotenkai`` logger must carry a RequestIDLogFilter so
    every record emitted by application code is stamped with the
    request-scope id."""
    assert _has_request_id_filter(ryotenkai_logger), (
        "ryotenkai logger missing RequestIDLogFilter -- Phase C wiring "
        "must remain intact; see packages/shared/src/ryotenkai_shared/"
        "utils/logger.py (setup_logger) and docs/plans/"
        "sharded-stargazing-wigderson.md (Layer 7)."
    )


def test_root_logger_has_request_id_filter() -> None:
    """The root logger must carry a RequestIDLogFilter so third-party
    library logs (mlflow, transformers, httpx, ...) emitted inside a
    request scope also pick up the correlation id."""
    root = logging.getLogger()
    assert _has_request_id_filter(root), (
        "root logger missing RequestIDLogFilter -- third-party logs "
        "would be missing the request_id correlation field."
    )


def test_request_id_filter_stamps_record_inside_request_scope() -> None:
    """End-to-end behaviour: setting :data:`REQUEST_ID` and emitting a
    log record produces ``record.request_id == <id>``. This is the
    contract the formatter and the operator both rely on."""
    rid_filter = RequestIDLogFilter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__,
        lineno=1, msg="m", args=(), exc_info=None,
    )

    token = REQUEST_ID.set("test-rid-stamp")
    try:
        assert rid_filter.filter(record) is True
    finally:
        REQUEST_ID.reset(token)

    assert getattr(record, "request_id", None) == "test-rid-stamp", (
        "RequestIDLogFilter did not stamp record.request_id"
    )


def test_request_id_filter_uses_dash_outside_request_scope() -> None:
    """Outside any request, the filter must stamp a ``"-"`` sentinel
    so log formatters never crash on a missing attribute. Picked
    ``"-"`` rather than ``"None"`` so it visually aligns with the
    typical access-log convention for "no value"."""
    rid_filter = RequestIDLogFilter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__,
        lineno=1, msg="m", args=(), exc_info=None,
    )
    # Ensure contextvar is not set (default).
    assert REQUEST_ID.get() is None

    assert rid_filter.filter(record) is True
    assert getattr(record, "request_id", None) == "-"
