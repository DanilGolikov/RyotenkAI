from unittest.mock import patch


def test_mixed() -> None:
    with patch("some.module.target") as mock_target:
        with patch.dict("os.environ", {"FOO": "bar"}):
            mock_target.return_value = 1
            assert True
