"""Single-import case: drop ``MagicMock`` from ``unittest.mock`` import
when it is no longer used after conversion.
"""

from unittest.mock import MagicMock


def test_drop_import() -> None:
    obj = MagicMock()
    obj.value = 1
    assert obj.value == 1
