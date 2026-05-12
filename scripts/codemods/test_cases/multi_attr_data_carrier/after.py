from types import SimpleNamespace


def test_multi_attr() -> None:
    cfg = SimpleNamespace(name="alpha", count=3, enabled=True)
    assert cfg.name == "alpha"
    assert cfg.count == 3
    assert cfg.enabled is True
