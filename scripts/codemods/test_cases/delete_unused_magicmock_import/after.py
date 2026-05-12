"""Single-import case: drop ``MagicMock`` from ``unittest.mock`` import
when it is no longer used after conversion.
"""

from types import SimpleNamespace


def test_drop_import() -> None:
    obj = SimpleNamespace(value=1)
    assert obj.value == 1
