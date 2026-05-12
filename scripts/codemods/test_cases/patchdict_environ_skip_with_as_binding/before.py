from unittest.mock import patch


def test_with_as() -> None:
    with patch.dict("os.environ", {"FOO": "bar"}) as env_dict:
        env_dict["BAZ"] = "qux"
        assert "BAZ" in env_dict
