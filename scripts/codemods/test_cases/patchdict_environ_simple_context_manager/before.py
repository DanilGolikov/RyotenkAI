from unittest.mock import patch


def test_something() -> None:
    with patch.dict("os.environ", {"FOO": "bar"}):
        result = "ok"
        assert result == "ok"
