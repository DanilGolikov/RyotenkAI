"""
DatasetPreviewLoader — paginated, memory-bounded dataset reads for the
UI / HTTP preview endpoint.

Goals:
  - NEVER load the full dataset. Local JSONL is consumed line-by-line
    via :func:`itertools.islice`; HuggingFace datasets go through
    ``load_dataset(streaming=True).skip().take()``.
  - Row-size cap: any single row serialising to more than
    :data:`ROW_SIZE_LIMIT_BYTES` is truncated, and the client sees a
    ``__truncated`` marker instead of a 10 MB payload.
  - Line-count cache: counting lines of a multi-GB JSONL is slow, so we
    memoise by ``(path, mtime_ns)``. Cache is per-process; backend
    process restart invalidates.

NB: This module is deliberately NOT used by the pipeline. Pipeline stage
loads the whole dataset (or uses `take()` for fast-mode validation) —
that path is optimised for correctness of a training run, not for
responsive preview pagination. Two code paths, one module each.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

from src.utils.logger import logger

if TYPE_CHECKING:
    from datasets import IterableDataset

Split = Literal["train", "eval"]

# Row size limit — serialised JSON string length. A typical LLM
# instruction row is ~2-10 KB; 1 MB is a generous ceiling that still
# fits comfortably in a single HTTP response with 50 rows/page.
ROW_SIZE_LIMIT_BYTES: int = 1_000_000

# Max rows per single preview request — the UI asks for 50, but we
# accept anything up to this cap. Prevents accidental memory hog if a
# client requests limit=10_000 for a 10 MB-row dataset.
PREVIEW_LIMIT_MAX: int = 200

# LRU-size for line-count cache. Dataset count is small (O(10)) per
# project, so 32 slots comfortably cover several projects in one server.
_LINE_COUNT_CACHE: dict[tuple[str, int], int] = {}
_LINE_COUNT_LOCK = Lock()
_LINE_COUNT_CAPACITY = 32


@dataclass
class PreviewPage:
    """A single page of rows returned to the UI."""

    rows: list[dict[str, Any]]
    total: int | None
    has_more: bool
    schema_hint: list[str] = field(default_factory=list)


class DatasetPreviewLoader:
    """Entry point for paginated dataset reads."""

    def preview_local_jsonl(
        self,
        path: Path,
        offset: int,
        limit: int,
    ) -> PreviewPage:
        """
        Return ``limit`` rows starting at ``offset`` from a local JSONL
        file. ``total`` is computed lazily (mtime-cached) — callers
        should treat it as authoritative for paging.

        Malformed lines (invalid JSON) are skipped with a logged warning
        but do NOT fail the request — the user is in "inspect the data"
        mode and a single broken row shouldn't blind them to the rest.
        Skipped lines still consume an index slot (offset counts raw
        file lines).
        """
        self._validate_range(offset=offset, limit=limit)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        total = _count_lines(path)
        rows: list[dict[str, Any]] = []

        # `utf-8-sig` strips a leading BOM if present; `\r\n` is handled
        # by Python's text-mode readline. Combined, this makes the
        # loader robust to Windows-authored jsonl.
        with path.open(encoding="utf-8-sig") as fh:
            for raw in itertools.islice(fh, offset, offset + limit):
                line = raw.rstrip("\r\n")
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "[PREVIEW] Skipping malformed JSONL at %s (offset %d): %s",
                        path,
                        offset + len(rows),
                        exc,
                    )
                    continue
                if not isinstance(value, dict):
                    # We only render dict rows — JSONL spec doesn't
                    # require dict, but every training dataset does.
                    continue
                rows.append(_truncate_if_huge(value))

        has_more = offset + limit < total
        schema_hint = _schema_hint(rows)
        return PreviewPage(rows=rows, total=total, has_more=has_more, schema_hint=schema_hint)

    def preview_hf(
        self,
        repo_id: str,
        split: Split,
        offset: int,
        limit: int,
        hf_token: str | None = None,
    ) -> PreviewPage:
        """
        Stream ``limit`` rows from HF Hub starting at ``offset``. Total
        row count is unknown for streaming datasets — caller sees
        ``total=None`` and must rely on ``has_more``.
        """
        self._validate_range(offset=offset, limit=limit)
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover — optional dep
            raise RuntimeError("The 'datasets' package is required for HF preview") from exc

        kwargs: dict[str, Any] = {"streaming": True}
        if hf_token:
            kwargs["token"] = hf_token

        try:
            ds = load_dataset(repo_id, split=split, **kwargs)
        except Exception as exc:
            logger.warning("[PREVIEW] HF load_dataset failed (%s / %s): %s", repo_id, split, exc)
            raise

        iterable: IterableDataset = ds  # type: ignore[assignment]
        # `.skip()` on an IterableDataset materialises nothing — it wraps
        # the generator so the first N rows are silently consumed when
        # iterated. We then bound the iterator with `islice` so a server
        # mistake can't tie up HF bandwidth on a 1M-row read.
        iterable = iterable.skip(offset)
        iterable = iterable.take(limit + 1)  # +1 to detect has_more

        rows: list[dict[str, Any]] = []
        more = False
        for idx, row in enumerate(iterable):
            if idx == limit:
                more = True
                break
            if isinstance(row, dict):
                rows.append(_truncate_if_huge(row))

        return PreviewPage(
            rows=rows,
            total=None,
            has_more=more,
            schema_hint=_schema_hint(rows),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _validate_range(*, offset: int, limit: int) -> None:
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        if limit <= 0:
            raise ValueError(f"limit must be > 0, got {limit}")
        if limit > PREVIEW_LIMIT_MAX:
            raise ValueError(f"limit cannot exceed {PREVIEW_LIMIT_MAX}, got {limit}")


# =============================================================================
# Helpers
# =============================================================================


def _count_lines(path: Path) -> int:
    """
    Return total line count for ``path``, cached by (path, mtime_ns).

    Uses a binary read and ``bytes.count(b"\\n")`` which is faster than
    line-iteration for large files (GB+). The last line without a
    trailing newline is still counted, matching jsonl-row semantics.
    """
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    with _LINE_COUNT_LOCK:
        cached = _LINE_COUNT_CACHE.get(key)
        if cached is not None:
            return cached

    # Bytewise count outside the lock — file IO is the slow bit.
    total = 0
    tail_non_empty = False
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            total += chunk.count(b"\n")
            tail_non_empty = not chunk.endswith(b"\n")
    if tail_non_empty:
        total += 1

    with _LINE_COUNT_LOCK:
        if len(_LINE_COUNT_CACHE) >= _LINE_COUNT_CAPACITY:
            # FIFO eviction — good enough for our access pattern.
            _LINE_COUNT_CACHE.pop(next(iter(_LINE_COUNT_CACHE)))
        _LINE_COUNT_CACHE[key] = total
    return total


def _truncate_if_huge(row: dict[str, Any]) -> dict[str, Any]:
    """
    If the serialised form of ``row`` exceeds :data:`ROW_SIZE_LIMIT_BYTES`,
    replace the largest string values with a ``"<TRUNCATED>"`` sentinel
    and attach a ``__truncated: true`` marker so the UI can flag it.
    """
    try:
        blob = json.dumps(row, ensure_ascii=False, default=str)
    except Exception:
        return {"__truncated": True, "reason": "serialisation failed"}

    if len(blob) <= ROW_SIZE_LIMIT_BYTES:
        return row

    # Simple heuristic — sort entries by serialised length, replace the
    # biggest ones first until we're under the limit.
    sized = sorted(
        row.items(),
        key=lambda kv: len(json.dumps(kv[1], ensure_ascii=False, default=str)),
        reverse=True,
    )
    truncated = dict(row)
    running_size = len(blob)
    for key, value in sized:
        if running_size <= ROW_SIZE_LIMIT_BYTES:
            break
        if isinstance(value, str):
            running_size -= len(value)
            truncated[key] = value[:200] + "…<truncated>"
            running_size += 200 + len("…<truncated>")
    truncated["__truncated"] = True
    return truncated


def _schema_hint(rows: list[dict[str, Any]]) -> list[str]:
    """
    Return the union of keys across the sampled rows preserving the
    insertion order of the first time each key appeared. UI uses this as
    a stable column ordering for the structured view.
    """
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            if key not in seen and not key.startswith("__"):
                seen[key] = None
    return list(seen.keys())


def clear_line_count_cache() -> None:
    """Testing helper — wipe the process-local line count memo."""
    with _LINE_COUNT_LOCK:
        _LINE_COUNT_CACHE.clear()


__all__ = [
    "DatasetPreviewLoader",
    "PREVIEW_LIMIT_MAX",
    "PreviewPage",
    "ROW_SIZE_LIMIT_BYTES",
    "Split",
    "clear_line_count_cache",
]
