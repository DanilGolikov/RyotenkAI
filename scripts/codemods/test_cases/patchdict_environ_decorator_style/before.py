from unittest.mock import patch


@patch.dict("os.environ", {"FOO": "bar"})
def test_decorated() -> None:
    assert True
