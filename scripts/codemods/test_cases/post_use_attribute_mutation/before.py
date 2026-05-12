"""Post-use *attribute* mutation without any aliasing.

``obj.foo`` is read (a safe use; doesn't alias the object) and then a
new attribute is assigned.  SimpleNamespace supports both, so this
is convertible.  The initial attribute assignments are absorbed; the
post-use one is kept as a plain attribute write.
"""

from unittest.mock import MagicMock


def test_post_use_attribute_mutation() -> None:
    obj = MagicMock()
    obj.a = 1
    obj.b = 2
    seen = obj.a  # read, not alias
    obj.c = 3  # post-use mutation
    assert seen == 1
    assert obj.c == 3
