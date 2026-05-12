from types import SimpleNamespace


def test_simple() -> None:
    obj = SimpleNamespace(value=42)
    assert obj.value == 42
