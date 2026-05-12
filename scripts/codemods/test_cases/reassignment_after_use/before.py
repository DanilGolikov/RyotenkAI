"""Attribute assignment happens *after* the variable is consumed.

The first-pass attribute assignments BEFORE the first use are absorbed
into the kwargs.  Mutations afterwards are left as plain attribute writes
(SimpleNamespace allows this).
"""

from unittest.mock import MagicMock


def use(o):  # pragma: no cover - fixture
    return o.a


def test_reassign_after_use() -> None:
    obj = MagicMock()
    obj.a = 1
    obj.b = 2
    used = use(obj)
    obj.c = 3  # post-use mutation; left as-is
    assert used == 1
    assert obj.c == 3
