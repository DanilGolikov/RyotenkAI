from unittest.mock import patch


def test_only_dict_patch() -> None:
    with patch.dict("os.environ", {"FOO": "bar"}):
        assert True
