"""Version info helpers — shared by the ``version`` command and the
``--version`` root flag so both print the same text.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PROJECT = "ryotenkai"


@dataclass(frozen=True, slots=True)
class VersionInfo:
    ryotenkai: str
    python: str
    platform: str
    git_sha: str | None = None

    def format(self) -> str:
        parts = [
            f"{_PROJECT} {self.ryotenkai}",
            f"python {self.python}",
            f"platform {self.platform}",
        ]
        if self.git_sha:
            parts.append(f"git {self.git_sha}")
        return "  ".join(parts)


def collect_version_info() -> VersionInfo:
    try:
        pkg_version = version(_PROJECT)
    except PackageNotFoundError:
        pkg_version = "0.0.0-dev"

    py_version = ".".join(str(s) for s in sys.version_info[:3])
    plat = f"{platform.system().lower()}-{platform.machine().lower()}"
    return VersionInfo(
        ryotenkai=pkg_version,
        python=py_version,
        platform=plat,
        git_sha=_git_short_sha(),
    )


def _git_short_sha(length: int = 7) -> str | None:
    """Return the short git SHA of the current HEAD, or None outside a repo."""
    cwd = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", f"--short={length}", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


__all__ = ["VersionInfo", "collect_version_info"]
