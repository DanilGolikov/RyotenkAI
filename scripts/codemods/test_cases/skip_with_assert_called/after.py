from unittest.mock import MagicMock


def do(spy):
    spy("payload")


def test_skip_with_assert_called() -> None:
    spy = MagicMock()
    do(spy)
    spy.assert_called_with("payload")
