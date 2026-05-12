from unittest.mock import MagicMock


def test_skip_callable() -> None:
    m = MagicMock()
    m.return_value = 7
    assert m() == 7
