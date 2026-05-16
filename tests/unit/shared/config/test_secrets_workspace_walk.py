"""Auto-discovery: ``secrets.env`` lives at the outermost uv workspace root.

Scenario: an operator runs ``ryotenkai`` from a nested checkout — a
tool-managed sub-workspace under the project root, a sibling worktree,
an extracted tarball nested under another monorepo. In all these
layouts the inner directory is a valid uv workspace
(``pyproject.toml`` + ``packages/``) AND so is the outer. The loader
walks up collecting EVERY workspace root and tries each in order
(innermost first → outermost last) so:

* a sub-workspace-local ``secrets.env`` overrides the canonical one
  (operator can shadow team defaults for testing),
* if the sub-workspace doesn't have its own, the project-canonical
  file at the outermost root is used (the common case — operators put
  ``secrets.env`` once in the main checkout).

This module pins that behaviour with a synthetic two-level workspace
constructed under ``tmp_path``. No VCS artefacts involved — pure
filesystem walk-up.

Seven-class coverage per CLAUDE.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ryotenkai_shared.config.secrets.loader import (
    _walk_up_workspace_roots,
    load_secrets,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _make_workspace(root: Path) -> None:
    """Mark a directory as a uv workspace (pyproject.toml + packages/)."""
    root.mkdir(parents=True, exist_ok=True)
    _write(root / "pyproject.toml", "[project]\nname = 'fake'\n")
    (root / "packages").mkdir(exist_ok=True)


def _make_nested_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build outer-workspace ⊃ inner-workspace ⊃ leaf, return all three.

    The "leaf" is where the loader's ``__file__`` would notionally live
    — pass it to ``_walk_up_workspace_roots`` and assert it finds BOTH
    workspaces in order.
    """
    outer = tmp_path / "outer"
    inner = outer / "subtree" / "inner"
    leaf = inner / "packages" / "shared" / "src" / "ryotenkai_shared" / "config" / "secrets" / "loader.py"
    _make_workspace(outer)
    _make_workspace(inner)
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.touch()
    return outer, inner, leaf


# ===========================================================================
# 1. POSITIVE
# ===========================================================================


class TestPositive:
    def test_walk_finds_inner_then_outer(self, tmp_path: Path) -> None:
        """Both nested workspace roots are returned, innermost first."""
        outer, inner, leaf = _make_nested_layout(tmp_path)
        roots = _walk_up_workspace_roots(leaf)
        assert roots == [inner, outer]

    def test_walk_single_workspace_returns_one(self, tmp_path: Path) -> None:
        """A plain (non-nested) checkout returns exactly one workspace root."""
        _make_workspace(tmp_path / "proj")
        leaf = tmp_path / "proj" / "packages" / "x.py"
        leaf.parent.mkdir(parents=True, exist_ok=True)
        leaf.touch()
        roots = _walk_up_workspace_roots(leaf)
        assert roots == [tmp_path / "proj"]


# ===========================================================================
# 2. NEGATIVE
# ===========================================================================


class TestNegative:
    def test_no_workspace_anywhere_returns_empty(self, tmp_path: Path) -> None:
        """When no marker is found on the walk, return an empty list."""
        leaf = tmp_path / "deeply" / "nested" / "file.py"
        leaf.parent.mkdir(parents=True, exist_ok=True)
        leaf.touch()
        assert _walk_up_workspace_roots(leaf) == []

    def test_pyproject_alone_is_not_a_workspace(self, tmp_path: Path) -> None:
        """A directory with ``pyproject.toml`` but no ``packages/`` is
        not a uv workspace — skipped on the walk."""
        proj = tmp_path / "single-package"
        proj.mkdir()
        _write(proj / "pyproject.toml", "[project]\nname='solo'\n")
        # No `packages/` dir — not a workspace.
        leaf = proj / "src" / "x.py"
        leaf.parent.mkdir(parents=True)
        leaf.touch()
        assert _walk_up_workspace_roots(leaf) == []

    def test_packages_alone_is_not_a_workspace(self, tmp_path: Path) -> None:
        """Same: ``packages/`` without ``pyproject.toml`` is incidental."""
        proj = tmp_path / "anything"
        (proj / "packages").mkdir(parents=True)
        leaf = proj / "x.py"
        leaf.touch()
        assert _walk_up_workspace_roots(leaf) == []


# ===========================================================================
# 3. BOUNDARY
# ===========================================================================


class TestBoundary:
    def test_leaf_at_workspace_root_does_not_match_self(
        self, tmp_path: Path,
    ) -> None:
        """``parents`` skips the file itself — a workspace root that
        IS the leaf doesn't count (semantically, the walk searches
        ABOVE the loader file)."""
        _make_workspace(tmp_path / "proj")
        # The "leaf" we pass is the pyproject.toml itself.
        leaf = tmp_path / "proj" / "pyproject.toml"
        roots = _walk_up_workspace_roots(leaf)
        assert roots == [tmp_path / "proj"]

    def test_triple_nesting(self, tmp_path: Path) -> None:
        """Three-level nesting returns all three roots, innermost-first."""
        a = tmp_path / "a"
        b = a / "b"
        c = b / "c"
        _make_workspace(a)
        _make_workspace(b)
        _make_workspace(c)
        # _make_workspace already created c/packages/; reuse it for the leaf.
        leaf = c / "packages" / "x.py"
        leaf.touch()
        roots = _walk_up_workspace_roots(leaf)
        assert roots == [c, b, a]


# ===========================================================================
# 4. INVARIANTS
# ===========================================================================


class TestInvariants:
    def test_innermost_secrets_env_wins_when_both_exist(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pin: innermost workspace's ``secrets.env`` takes precedence
        over the outermost one. Operators can shadow team defaults for
        per-sub-workspace testing."""
        outer, inner, leaf = _make_nested_layout(tmp_path)
        _write(outer / "secrets.env", "RUNPOD_API_KEY=outer_canonical")
        _write(inner / "secrets.env", "RUNPOD_API_KEY=inner_override")

        # Make the loader's walk start at our synthetic leaf.
        monkeypatch.setattr(
            "ryotenkai_shared.config.secrets.loader._walk_up_workspace_roots",
            lambda _start: _walk_up_workspace_roots(leaf),
        )
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        monkeypatch.delenv("RYOTENKAI_SECRETS_FILE", raising=False)

        s = load_secrets()
        assert s.runpod_api_key == "inner_override"

    def test_outer_secrets_env_used_when_inner_lacks_it(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Canonical case: inner sub-workspace has NO ``secrets.env`` of
        its own; loader falls back to the outermost workspace's file.

        This is the exact developer scenario the auto-discovery fix
        targets: ``ryotenkai`` invoked from a tool-managed sub-workspace
        under a project root that has the canonical ``secrets.env``.
        """
        outer, inner, leaf = _make_nested_layout(tmp_path)
        _write(outer / "secrets.env", "RUNPOD_API_KEY=outer_canonical")
        # No file in inner.

        monkeypatch.setattr(
            "ryotenkai_shared.config.secrets.loader._walk_up_workspace_roots",
            lambda _start: _walk_up_workspace_roots(leaf),
        )
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        monkeypatch.delenv("RYOTENKAI_SECRETS_FILE", raising=False)

        s = load_secrets()
        assert s.runpod_api_key == "outer_canonical"


# ===========================================================================
# 5. DEPENDENCY ERRORS
# ===========================================================================


class TestDependencyErrors:
    def test_walk_swallowed_by_loader_when_is_file_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``load_secrets`` wraps the walk in try/except so a filesystem
        error during root collection doesn't crash the loader.

        Targets the contract at the call boundary in ``load_secrets``,
        not the helper itself — the helper is allowed to propagate, the
        loader is required to catch.
        """
        leaf = tmp_path / "x.py"
        leaf.touch()

        # Make ``_walk_up_workspace_roots`` raise; ``load_secrets`` must
        # catch and fall back to "no workspace candidates" cleanly.
        def boom(_start: Path) -> list[Path]:
            raise OSError("permission denied")

        monkeypatch.setattr(
            "ryotenkai_shared.config.secrets.loader._walk_up_workspace_roots",
            boom,
        )
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("RYOTENKAI_SECRETS_FILE", raising=False)
        # No env_file passed; auto-discovery raises; loader must still
        # return a Secrets object with optional fields None.
        s = load_secrets()
        assert s.runpod_api_key is None


# ===========================================================================
# 6. REGRESSIONS
# ===========================================================================


class TestRegressions:
    def test_developer_nested_workspace_scenario(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression for 2026-05-16 bug report.

        Original repro: developer runs ``ryotenkai run start`` from a
        nested sub-workspace at
        ``<project>/.claude/worktrees/<name>/`` whose own
        ``secrets.env`` is missing, but the project root at
        ``<project>/`` has the file with all credentials.

        Pre-fix: loader's walk-up stopped at the FIRST workspace root
        match (the sub-workspace) and reported "RUNPOD_API_KEY required"
        even though the project root had the file two levels up.

        Post-fix: walk collects every workspace root and tries each.
        The project root's ``secrets.env`` is found and loaded.
        """
        outer, inner, leaf = _make_nested_layout(tmp_path)
        _write(outer / "secrets.env", "RUNPOD_API_KEY=from_project_root")
        # No file in the sub-workspace — exactly the user's situation.

        monkeypatch.setattr(
            "ryotenkai_shared.config.secrets.loader._walk_up_workspace_roots",
            lambda _start: _walk_up_workspace_roots(leaf),
        )
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        monkeypatch.delenv("RYOTENKAI_SECRETS_FILE", raising=False)

        s = load_secrets()
        assert s.runpod_api_key == "from_project_root"


# ===========================================================================
# 7. LOGIC SPECIFIC
# ===========================================================================


class TestLogicSpecific:
    @pytest.mark.parametrize(
        "depth, expected_count",
        [
            (1, 1),
            (2, 2),
            (4, 4),
        ],
    )
    def test_walk_collects_n_levels(
        self, tmp_path: Path, depth: int, expected_count: int,
    ) -> None:
        """Parametrise over nesting depths; every level marked as
        workspace is returned in inner-first order."""
        current = tmp_path
        for i in range(depth):
            current = current / f"level_{i}"
            _make_workspace(current)
        leaf = current / "x.py"
        leaf.touch()
        roots = _walk_up_workspace_roots(leaf)
        assert len(roots) == expected_count
        # Innermost first.
        assert roots[0].name == f"level_{depth - 1}"
        assert roots[-1].name == "level_0"
