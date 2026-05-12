"""``m.foo.return_value = ...`` — chained mock-only attribute usage.

Even though the outer name is bound to a bare ``MagicMock()``, the
chained ``.return_value`` on ``m.foo`` indicates ``m.foo`` is being used
as a callable spy.  Since SimpleNamespace creates plain attributes (not
auto-magic Mock children), this conversion would lose semantics.
"""

from unittest.mock import MagicMock


def test_skip_chained_attr() -> None:
    m = MagicMock()
    m.foo.return_value = 1
    assert m.foo() == 1
