from unittest.mock import patch


def test_nested() -> None:
    with patch.dict("os.environ", {"A": "1"}), patch.dict("os.environ", {"B": "2"}):
        result = "ok"
        assert result == "ok"
