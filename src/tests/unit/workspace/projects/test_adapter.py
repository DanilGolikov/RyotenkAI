"""Tests for the project → spawn-launch adapter.

The adapter's job is narrow: read a project's filesystem, return a
:class:`ResolvedProject` (paths + env + metadata) that the launcher
combines with a ``LaunchRequest`` to spawn the worker subprocess.
The YAML is not loaded here — that happens inside the worker.

These tests pin every contract corner: registry lookup, config-override
semantics, env.json round-trip, metadata invariants, walk-up resume
discovery, registry-missing fallback for moved projects, and actor
resolution precedence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.workspace.projects.adapter import (
    ProjectNotFoundError,
    ResolvedProject,
    resolve_project_launch_inputs,
    resolve_project_launch_inputs_from_run_dir,
)
from src.workspace.projects.registry import ProjectRegistry
from src.workspace.projects.store import ProjectStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_project(
    workspace_root: Path,
    *,
    project_id: str = "p1",
    name: str = "Project 1",
    config_yaml: str = "model:\n  name: stub\n",
    env: dict[str, str] | None = None,
) -> ProjectRegistry:
    """Lay down a registered project under ``workspace_root``.

    Returns the registry so the test can pass it to the resolvers
    for isolation (no global filesystem mutation).
    """
    registry = ProjectRegistry(root=workspace_root)
    project_path = registry.default_project_path(project_id)
    store = ProjectStore(project_path)
    store.create(id=project_id, name=name)
    # Overwrite seeded empty config.yaml with given content.
    store.current_config_path.write_text(config_yaml, encoding="utf-8")
    if env is not None:
        store.write_env(env)
    registry.register(project_id=project_id, name=name, path=project_path)
    return registry


# ---------------------------------------------------------------------------
# resolve_project_launch_inputs
# ---------------------------------------------------------------------------


class TestResolveProjectLaunchInputs:
    """Explicit-id resolver. Used by CLI ``run start --project X`` and
    by the walk-up resolver internally."""

    def test_positive_returns_resolved_project_with_paths_only(
        self, tmp_path: Path
    ) -> None:
        registry = _make_project(tmp_path, env={"FOO": "bar"})
        resolved = resolve_project_launch_inputs("p1", registry=registry)

        assert isinstance(resolved, ResolvedProject)
        assert resolved.config_path.name == "current.yaml"
        assert resolved.config_path.is_file()
        assert resolved.env == {"FOO": "bar"}
        assert resolved.metadata["project_id"] == "p1"
        assert resolved.metadata["actor"]
        assert resolved.metadata["config_version_hash"]
        assert resolved.runs_base_dir.is_dir()

    def test_positive_config_override_replaces_path_and_records_breadcrumb(
        self, tmp_path: Path
    ) -> None:
        registry = _make_project(tmp_path)
        override = tmp_path / "override.yaml"
        override.write_text("model: {name: o}\n", encoding="utf-8")

        resolved = resolve_project_launch_inputs(
            "p1", config_override=override, registry=registry,
        )
        assert resolved.config_path == override.resolve()
        assert resolved.metadata["config_override_path"] == str(override.resolve())

    def test_negative_unknown_project_raises_not_found(
        self, tmp_path: Path
    ) -> None:
        registry = ProjectRegistry(root=tmp_path)
        with pytest.raises(ProjectNotFoundError):
            resolve_project_launch_inputs("nope", registry=registry)

    def test_negative_missing_directory_raises(self, tmp_path: Path) -> None:
        registry = _make_project(tmp_path)
        # Wipe the directory but keep the registry entry.
        import shutil
        shutil.rmtree(registry.default_project_path("p1"))
        with pytest.raises(ProjectNotFoundError):
            resolve_project_launch_inputs("p1", registry=registry)

    def test_boundary_no_env_json_yields_empty_env(self, tmp_path: Path) -> None:
        registry = _make_project(tmp_path, env=None)
        resolved = resolve_project_launch_inputs("p1", registry=registry)
        assert resolved.env == {}

    def test_invariant_metadata_always_has_three_keys(
        self, tmp_path: Path
    ) -> None:
        registry = _make_project(tmp_path)
        resolved = resolve_project_launch_inputs("p1", registry=registry)
        assert {"project_id", "actor", "config_version_hash"} <= resolved.metadata.keys()

    def test_invariant_resolved_project_is_frozen(self, tmp_path: Path) -> None:
        registry = _make_project(tmp_path)
        resolved = resolve_project_launch_inputs("p1", registry=registry)
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            resolved.config_path = Path("/elsewhere")  # type: ignore[misc]

    def test_logic_actor_explicit_arg_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = _make_project(tmp_path)
        monkeypatch.setenv("RYOTENKAI_ACTOR", "agent:bot")
        resolved = resolve_project_launch_inputs(
            "p1", actor="explicit", registry=registry,
        )
        assert resolved.metadata["actor"] == "explicit"

    def test_logic_config_version_hash_changes_with_yaml_content(
        self, tmp_path: Path
    ) -> None:
        registry = _make_project(tmp_path, config_yaml="model:\n  name: a\n")
        first = resolve_project_launch_inputs("p1", registry=registry)
        # Edit the config file in place.
        registry.resolve("p1")  # warm up
        ProjectStore(registry.default_project_path("p1")).current_config_path.write_text(
            "model:\n  name: b\n", encoding="utf-8",
        )
        second = resolve_project_launch_inputs("p1", registry=registry)
        assert first.metadata["config_version_hash"] != second.metadata["config_version_hash"]


# ---------------------------------------------------------------------------
# resolve_project_launch_inputs_from_run_dir
# ---------------------------------------------------------------------------


class TestResolveLaunchInputsFromRunDir:
    """Walk-up resolver. Used by Web API ``POST /launch`` and CLI
    ``resume/restart`` to re-acquire project context from the run dir."""

    def test_positive_run_dir_inside_project_workspace_resolves(
        self, tmp_path: Path
    ) -> None:
        registry = _make_project(tmp_path, env={"K": "v"})
        store = ProjectStore(registry.default_project_path("p1"))
        run_dir = store.runs_dir / "run_xyz"
        run_dir.mkdir(parents=True, exist_ok=True)

        resolved = resolve_project_launch_inputs_from_run_dir(
            run_dir, registry=registry,
        )
        assert resolved is not None
        assert resolved.metadata["project_id"] == "p1"
        assert resolved.env == {"K": "v"}

    def test_negative_run_dir_outside_any_project_returns_none(
        self, tmp_path: Path
    ) -> None:
        ad_hoc_run = tmp_path / "lonely_run"
        ad_hoc_run.mkdir()
        assert resolve_project_launch_inputs_from_run_dir(ad_hoc_run) is None

    def test_logic_walk_up_uses_on_disk_project_id_not_registry_lookup(
        self, tmp_path: Path
    ) -> None:
        """Project moved/renamed without re-registering — walk-up still
        resolves via the directory's own ``project.json``. Filesystem
        is authoritative for the run."""
        registry = _make_project(tmp_path, project_id="p1", env={"X": "y"})
        store = ProjectStore(registry.default_project_path("p1"))
        run_dir = store.runs_dir / "abandoned_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Drop the project from the registry but leave the directory.
        registry.unregister("p1")

        resolved = resolve_project_launch_inputs_from_run_dir(
            run_dir, registry=registry,
        )
        assert resolved is not None
        assert resolved.metadata["project_id"] == "p1"
        assert resolved.env == {"X": "y"}

    def test_boundary_walk_up_finds_nearest_project_json(
        self, tmp_path: Path
    ) -> None:
        """Walk-up picks up the nearest enclosing project.json
        regardless of nesting depth."""
        registry = _make_project(tmp_path, project_id="p1")
        store = ProjectStore(registry.default_project_path("p1"))
        nested = store.runs_dir / "a" / "b" / "c"
        nested.mkdir(parents=True)
        resolved = resolve_project_launch_inputs_from_run_dir(
            nested, registry=registry,
        )
        assert resolved is not None
        assert resolved.metadata["project_id"] == "p1"

    def test_dependency_error_malformed_project_json_returns_none(
        self, tmp_path: Path
    ) -> None:
        """Project's ``project.json`` is unparseable JSON — defensive
        return ``None`` (treat as no project context, don't crash)."""
        registry = _make_project(tmp_path)
        store = ProjectStore(registry.default_project_path("p1"))
        store.metadata_path.write_text("{not valid json", encoding="utf-8")
        run_dir = store.runs_dir / "run_xyz"
        run_dir.mkdir(parents=True)

        assert resolve_project_launch_inputs_from_run_dir(
            run_dir, registry=registry,
        ) is None
