"""Interleaved non-attribute-assignment statement closes absorption.

A nested function defined *between* the candidate creation and a later
``client.subscribe_events = _stream`` introduces a forward reference
that cannot be folded into the constructor (``_stream`` isn't defined
yet on the original line).  The codemod must NOT absorb across such
boundaries.
"""

from types import SimpleNamespace

from unittest.mock import AsyncMock


def test_skip_interleaved_statement() -> None:
    client = SimpleNamespace(get_status=AsyncMock(return_value={"state": "running"}))

    async def _stream(_job_id, **_kwargs):  # pragma: no cover - fixture
        yield {"offset": 0, "kind": "step", "payload": {}}

    client.subscribe_events = _stream
    assert client.subscribe_events is _stream
