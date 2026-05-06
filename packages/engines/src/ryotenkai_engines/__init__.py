"""ryotenkai_engines — inference engine plugin system.

PR-1 of 10 (engines scaffolding) — module stubs only. Public API is empty;
real implementations land in PR-2 (Protocol + Manifest + Registry).
See ``docs/plans/purring-sleeping-hartmanis.md`` for the full design.
"""

from __future__ import annotations

__version__ = "1.0.0"

# Public API surface — populated in PR-2. Importing from a sub-module before
# that PR lands raises ImportError because the symbols don't exist yet.
__all__: tuple[str, ...] = ()
