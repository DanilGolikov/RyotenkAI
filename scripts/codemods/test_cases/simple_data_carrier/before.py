from unittest.mock import MagicMock


def test_simple() -> None:
    obj = MagicMock()
    obj.value = 42
    assert obj.value == 42
