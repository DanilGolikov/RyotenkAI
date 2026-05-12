def test_only_dict_patch(monkeypatch) -> None:
    monkeypatch.setenv("FOO", "bar")
    assert True
