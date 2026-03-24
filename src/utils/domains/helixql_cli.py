"""Lightweight wrapper around the Helix CLI for query validation.

Uses ``helix compile --quiet`` which performs parsing, semantic analysis,
and Rust code generation **without** invoking ``cargo check``.  This makes
it safe to call per-sample (~8 ms) whereas ``helix check`` triggers a full
Rust build (minutes) since Helix CLI ≥ 2.3.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

_INIT_TIMEOUT: int = 10


@dataclass(frozen=True)
class CompileResult:
    """Outcome of a single ``helix compile`` invocation."""

    ok: bool
    error_type: str


class HelixCompiler:
    """Stateless, cached wrapper for ``helix init`` + ``helix compile``.

    Instances are cheap — keep one per plugin and share the result cache
    across samples.
    """

    _CLI_NAME: ClassVar[str] = "helix"

    def __init__(self, *, timeout_seconds: int = 10) -> None:
        self._timeout = timeout_seconds
        self._cache: dict[tuple[str, str], CompileResult] = {}

    def validate(self, *, schema: str, query: str) -> CompileResult:
        """Return cached ``CompileResult`` for a *(schema, query)* pair."""
        key = (schema, query)
        if key in self._cache:
            return self._cache[key]
        result = self._compile(schema=schema, query=query)
        self._cache[key] = result
        return result

    def _compile(self, *, schema: str, query: str) -> CompileResult:
        helix_bin = shutil.which(self._CLI_NAME)
        if helix_bin is None:
            return CompileResult(ok=False, error_type="cli_missing")

        with tempfile.TemporaryDirectory(prefix="helix_val_") as tmp:
            project = Path(tmp)

            try:
                subprocess.run(
                    [helix_bin, "init", "--quiet"],
                    cwd=project,
                    capture_output=True,
                    timeout=_INIT_TIMEOUT,
                    check=True,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                return CompileResult(ok=False, error_type="init_failed")

            (project / "db" / "schema.hx").write_text(
                f"{schema.strip()}\n",
                encoding="utf-8",
            )
            (project / "db" / "queries.hx").write_text(
                f"{query.strip()}\n",
                encoding="utf-8",
            )

            try:
                proc = subprocess.run(
                    [helix_bin, "compile", "--quiet"],
                    cwd=project,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                )
            except subprocess.TimeoutExpired:
                return CompileResult(ok=False, error_type="timeout")

        if proc.returncode == 0:
            return CompileResult(ok=True, error_type="ok")

        lowered = f"{proc.stdout}\n{proc.stderr}".lower()
        if "parse error" in lowered:
            return CompileResult(ok=False, error_type="parse_error")
        if "validate" in lowered or "invalid" in lowered:
            return CompileResult(ok=False, error_type="validation_error")
        return CompileResult(ok=False, error_type="compiler_error")


__all__ = ["CompileResult", "HelixCompiler"]
