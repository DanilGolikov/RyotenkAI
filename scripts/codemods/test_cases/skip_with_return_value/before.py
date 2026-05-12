from unittest.mock import MagicMock


def test_skip_return_value() -> None:
    fn = MagicMock()
    fn.return_value = "hello"
    fn.side_effect = None
    assert fn() == "hello"
