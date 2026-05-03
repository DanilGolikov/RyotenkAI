"""Unit tests for ``src.data.preview.loader``.

Covers:
  - Local JSONL pagination, offset/limit edge cases
  - Malformed lines skipped with no raise
  - BOM and CRLF input
  - mtime-keyed line count cache
  - Row truncation marker for huge serialised rows
  - schema_hint preserves first-seen order across rows
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.preview.loader import (
    PREVIEW_LIMIT_MAX,
    ROW_SIZE_LIMIT_BYTES,
    DatasetPreviewLoader,
    clear_line_count_cache,
)


@pytest.fixture(autouse=True)
def _wipe_cache():
    clear_line_count_cache()
    yield
    clear_line_count_cache()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "\n".join(json.dumps(r) for r in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def test_preview_local_jsonl_basic(tmp_path: Path):
    f = tmp_path / "ds.jsonl"
    _write_jsonl(f, [{"i": i, "text": f"row-{i}"} for i in range(10)])

    page = DatasetPreviewLoader().preview_local_jsonl(f, offset=0, limit=3)

    assert [r["i"] for r in page.rows] == [0, 1, 2]
    assert page.total == 10
    assert page.has_more is True
    assert page.schema_hint == ["i", "text"]


def test_preview_local_jsonl_offset(tmp_path: Path):
    f = tmp_path / "ds.jsonl"
    _write_jsonl(f, [{"i": i} for i in range(5)])

    page = DatasetPreviewLoader().preview_local_jsonl(f, offset=3, limit=10)
    assert [r["i"] for r in page.rows] == [3, 4]
    assert page.has_more is False


def test_preview_local_jsonl_offset_past_end(tmp_path: Path):
    f = tmp_path / "ds.jsonl"
    _write_jsonl(f, [{"i": i} for i in range(3)])

    page = DatasetPreviewLoader().preview_local_jsonl(f, offset=100, limit=10)
    assert page.rows == []
    assert page.has_more is False
    assert page.total == 3


def test_preview_local_jsonl_skips_malformed_lines(tmp_path: Path):
    f = tmp_path / "ds.jsonl"
    f.write_text(
        '{"i": 1}\n'
        "{not json}\n"
        '{"i": 3}\n',
        encoding="utf-8",
    )

    page = DatasetPreviewLoader().preview_local_jsonl(f, offset=0, limit=10)
    # Malformed middle line is dropped, others come through.
    assert [r["i"] for r in page.rows] == [1, 3]
    # Total counts raw lines (3) including the malformed one — UI sees
    # 3 in the counter so the gap doesn't surprise them.
    assert page.total == 3


def test_preview_local_jsonl_handles_bom(tmp_path: Path):
    f = tmp_path / "ds.jsonl"
    # Write BOM + payload manually so we exercise the `utf-8-sig` path.
    f.write_bytes(b"\xef\xbb\xbf" + b'{"a": 1}\n{"a": 2}\n')

    page = DatasetPreviewLoader().preview_local_jsonl(f, offset=0, limit=10)
    assert [r["a"] for r in page.rows] == [1, 2]


def test_preview_local_jsonl_handles_crlf(tmp_path: Path):
    f = tmp_path / "ds.jsonl"
    f.write_bytes(b'{"a": 1}\r\n{"a": 2}\r\n')

    page = DatasetPreviewLoader().preview_local_jsonl(f, offset=0, limit=10)
    assert [r["a"] for r in page.rows] == [1, 2]


def test_preview_rejects_invalid_range(tmp_path: Path):
    f = tmp_path / "ds.jsonl"
    f.write_text("{}\n", encoding="utf-8")

    loader = DatasetPreviewLoader()
    with pytest.raises(ValueError):
        loader.preview_local_jsonl(f, offset=-1, limit=1)
    with pytest.raises(ValueError):
        loader.preview_local_jsonl(f, offset=0, limit=0)
    with pytest.raises(ValueError):
        loader.preview_local_jsonl(f, offset=0, limit=PREVIEW_LIMIT_MAX + 1)


def test_preview_truncates_huge_rows(tmp_path: Path):
    f = tmp_path / "ds.jsonl"
    big = "x" * (ROW_SIZE_LIMIT_BYTES + 100)
    f.write_text(json.dumps({"big": big, "small": "ok"}) + "\n", encoding="utf-8")

    page = DatasetPreviewLoader().preview_local_jsonl(f, offset=0, limit=1)
    row = page.rows[0]
    assert row.get("__truncated") is True
    assert "<truncated>" in row["big"]
    # Untouched fields stay readable.
    assert row["small"] == "ok"


def test_schema_hint_preserves_first_seen_order(tmp_path: Path):
    f = tmp_path / "ds.jsonl"
    f.write_text(
        json.dumps({"a": 1, "b": 2}) + "\n"
        + json.dumps({"b": 2, "c": 3}) + "\n",
        encoding="utf-8",
    )

    page = DatasetPreviewLoader().preview_local_jsonl(f, offset=0, limit=10)
    assert page.schema_hint == ["a", "b", "c"]


def test_line_count_cache_invalidates_on_mtime(tmp_path: Path):
    f = tmp_path / "ds.jsonl"
    _write_jsonl(f, [{"i": i} for i in range(5)])

    loader = DatasetPreviewLoader()
    page1 = loader.preview_local_jsonl(f, offset=0, limit=1)
    assert page1.total == 5

    # Append rows + change mtime.
    _write_jsonl(f, [{"i": i} for i in range(8)])
    page2 = loader.preview_local_jsonl(f, offset=0, limit=1)
    assert page2.total == 8


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        DatasetPreviewLoader().preview_local_jsonl(
            tmp_path / "nope.jsonl", offset=0, limit=1
        )
