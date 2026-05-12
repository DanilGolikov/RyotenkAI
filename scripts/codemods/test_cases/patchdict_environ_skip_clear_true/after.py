from unittest.mock import patch


def test_clear_true() -> None:
    # TODO(codemod): manual review needed for clear=True / with-as binding
    with patch.dict("os.environ", {"FOO": "bar"}, clear=True):
        result = "ok"
        assert result == "ok"
