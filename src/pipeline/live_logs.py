"""Offset-based live log tail.

Domain-agnostic infrastructure: given a file path, read new lines since the last
call via a persistent byte offset. Shared by TUI live monitor, web WebSocket
stream, and any future client. Previously lived in src/tui/live_logs.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class LiveLogTail:
    path: Path | None = None
    offset: int = 0

    def reset(self, path: Path | None) -> None:
        self.path = path
        self.offset = 0

    def load_full(self, path: Path) -> list[str]:
        self.reset(path)
        return self.read_new_lines()

    def read_new_lines(self) -> list[str]:
        if self.path is None or not self.path.exists():
            return []

        file_size = self.path.stat().st_size
        if file_size < self.offset:
            self.offset = 0

        with self.path.open(encoding="utf-8", errors="replace") as handle:
            handle.seek(self.offset)
            lines = [line.rstrip() for line in handle]
            self.offset = handle.tell()
        return lines


__all__ = ["LiveLogTail"]
