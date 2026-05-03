from __future__ import annotations


def test_reports_main_importable() -> None:
    # Should not raise; ensures `python -m src.reports` entrypoint module is covered.
    import src.reports.__main__  # noqa: F401
