"""Nested attribute assignment ``m.foo.bar = X``.

The outer ``m`` is convertible (it's only used to access ``m.foo``),
but the *inner* ``m.foo.bar = X`` cannot be folded into a
``SimpleNamespace`` kwarg because ``m.foo`` doesn't exist yet.

Conservative behaviour: SKIP the outer assignment.  SimpleNamespace
doesn't auto-create the nested ``foo`` attribute, so the rewrite
would break ``m.foo.bar = X``.
"""

from unittest.mock import MagicMock


def test_nested_attr_assignment() -> None:
    m = MagicMock()
    m.foo.bar = 5
    assert m.foo.bar == 5
