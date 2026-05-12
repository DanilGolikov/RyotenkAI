"""The common ``runner = MagicMock(); runner.method = AsyncMock(...)`` pattern.

The outer ``runner`` is *just* a data carrier — only its attributes are
ever called (``runner.get_status(...)``).  This is the dominant pattern
in ``tests/unit/control/api/routers/``.  SimpleNamespace handles it
perfectly because the AsyncMock attribute is itself callable.
"""

from types import SimpleNamespace

from unittest.mock import AsyncMock


def test_runner_pattern() -> None:
    runner = SimpleNamespace(get_status=AsyncMock(return_value={"state": "running"}))
    assert runner.get_status is not None
