"""Inline ``MagicMock()`` not bound to a name.

The codemod intentionally leaves this case alone — it only handles the
``var = MagicMock()`` assignment pattern.  A separate pass can handle
inline cases later if needed.
"""

from unittest.mock import MagicMock


def consume(*args):  # pragma: no cover - fixture
    return args


def test_inline_argument() -> None:
    result = consume(MagicMock(), MagicMock())
    assert result is not None
