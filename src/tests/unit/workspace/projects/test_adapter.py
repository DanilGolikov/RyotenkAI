"""Tests for the project → orchestrator adapter (Step 5 of Variant 1).

The adapter's job is narrow: read a project's filesystem, return a
:class:`ProjectInputs` with ``(config, env, metadata)`` ready for the
orchestrator. These tests pin every contract corner: registry lookup,
config-override semantics, env.json round-trip, metadata invariants,
actor resolution precedence, and isolation guarantees.

Categories covered: positive, negative, boundary, invariants,
dependency-error, regression, logic-specific. Combinatorial coverage of
``(project_exists × config_override × env_json_present)`` lives at the
bottom.

We mock :func:`src.utils.config.load_config` so tests don't need a full
PipelineConfig YAML — the adapter's only requirement of the loaded
config is that it round-trips through the dataclass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.workspace.projects.adapter import (
    ProjectInputs,
    ProjectNotFoundError,
    ResolvedProject,
    load_project_inputs,
    resolve_project_launch_inputs,
    resolve_project_launch_inputs_from_run_dir,
)
from src.workspace.projects.registry import ProjectRegistry
from src.workspace.projects.store import ProjectStore, ProjectStoreError

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

    Returns the registry so the test can pass it to
    :func:`load_project_inputs` for isolation.
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


def _stub_config() -> Any:
    """Minimal stand-in for a loaded PipelineConfig.

    The adapter only stores what ``load_config`` returns into
    ``ProjectInputs.config``, so a sentinel object is enough — every
    test that touches ``inputs.config`` checks identity, not internals.
    """
    sentinel = MagicMock(name="PipelineConfig")
    sentinel.integrations = MagicMock()
    return sentinel


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_returns_project_inputs_with_all_three_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = _make_project(tmp_path, env={"FOO": "bar"})
        cfg = _stub_config()

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config", return_value=cfg
        ):
            inputs = load_project_inputs("p1", registry=registry)

        assert isinstance(inputs, ProjectInputs)
        assert inputs.config is cfg
        assert inputs.env == {"FOO": "bar"}
        assert inputs.metadata["project_id"] == "p1"

    def test_project_with_no_env_json_yields_empty_env(
        self, tmp_path: Path
    ) -> None:
        registry = _make_project(tmp_path, env=None)

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ):
            inputs = load_project_inputs("p1", registry=registry)

        assert inputs.env == {}

    def test_actor_explicit_arg_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = _make_project(tmp_path)
        monkeypatch.setenv("RYOTENKAI_ACTOR", "agent:bot")
        monkeypatch.setenv("USER", "human")

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ):
            inputs = load_project_inputs(
                "p1", actor="explicit-caller", registry=registry,
            )

        assert inputs.metadata["actor"] == "explicit-caller"

    def test_extra_metadata_merged(
        self, tmp_path: Path
    ) -> None:
        registry = _make_project(tmp_path)

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ):
            inputs = load_project_inputs(
                "p1",
                extra_metadata={"session_id": "s-1", "request_id": "r-9"},
                registry=registry,
            )

        assert inputs.metadata["session_id"] == "s-1"
        assert inputs.metadata["request_id"] == "r-9"
        # Invariants still present.
        assert inputs.metadata["project_id"] == "p1"

    def test_config_override_replaces_project_yaml(
        self, tmp_path: Path
    ) -> None:
        registry = _make_project(tmp_path)
        override = tmp_path / "override.yaml"
        override.write_text("# override yaml\n", encoding="utf-8")

        cfg = _stub_config()
        with patch(
            "src.workspace.projects.adapter.load_pipeline_config", return_value=cfg,
        ) as load_mock:
            inputs = load_project_inputs(
                "p1", config_override=override, registry=registry,
            )

        # load_config was called with the override path, not the
        # project's current.yaml.
        called_path = load_mock.call_args.args[0]
        assert called_path == override.resolve()
        # config_override_path breadcrumb stamped in metadata.
        assert inputs.metadata["config_override_path"] == str(
            override.resolve()
        )


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_unknown_project_id_raises(self, tmp_path: Path) -> None:
        registry = ProjectRegistry(root=tmp_path)

        with pytest.raises(ProjectNotFoundError) as exc:
            load_project_inputs("does-not-exist", registry=registry)

        assert exc.value.project_id == "does-not-exist"

    def test_registered_but_directory_missing_raises(
        self, tmp_path: Path
    ) -> None:
        """Registry points to a path that's been deleted from disk."""
        registry = _make_project(tmp_path)
        # Wipe the project dir but leave the registry entry pointing
        # to it.
        import shutil

        entry = registry.resolve("p1")
        shutil.rmtree(entry.path)

        with pytest.raises(ProjectNotFoundError) as exc:
            load_project_inputs("p1", registry=registry)

        assert "missing" in str(exc.value)

    def test_no_config_in_project_raises(self, tmp_path: Path) -> None:
        registry = _make_project(tmp_path)
        # Remove the seeded config.
        store = ProjectStore(registry.resolve("p1").path)
        store.current_config_path.unlink()

        with pytest.raises(ProjectNotFoundError) as exc:
            load_project_inputs("p1", registry=registry)

        assert "no config" in str(exc.value)

    def test_config_override_not_found_raises(self, tmp_path: Path) -> None:
        registry = _make_project(tmp_path)

        with pytest.raises(ProjectNotFoundError):
            load_project_inputs(
                "p1",
                config_override=tmp_path / "missing-override.yaml",
                registry=registry,
            )

    def test_malformed_env_json_surfaces_store_error(
        self, tmp_path: Path
    ) -> None:
        registry = _make_project(tmp_path)
        store = ProjectStore(registry.resolve("p1").path)
        # Write malformed JSON.
        store.env_path.write_text("{not: valid json", encoding="utf-8")

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ), pytest.raises(ProjectStoreError):
            load_project_inputs("p1", registry=registry)


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_empty_env_json_is_empty_dict(self, tmp_path: Path) -> None:
        registry = _make_project(tmp_path)
        store = ProjectStore(registry.resolve("p1").path)
        store.env_path.write_text("{}", encoding="utf-8")

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ):
            inputs = load_project_inputs("p1", registry=registry)

        assert inputs.env == {}

    def test_env_json_with_unicode_values(self, tmp_path: Path) -> None:
        registry = _make_project(tmp_path, env={"GREETING": "héllo wörld"})

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ):
            inputs = load_project_inputs("p1", registry=registry)

        assert inputs.env["GREETING"] == "héllo wörld"

    def test_actor_unknown_when_no_user_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = _make_project(tmp_path)
        monkeypatch.delenv("RYOTENKAI_ACTOR", raising=False)
        monkeypatch.delenv("USER", raising=False)
        monkeypatch.delenv("USERNAME", raising=False)

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ):
            inputs = load_project_inputs("p1", registry=registry)

        assert inputs.metadata["actor"] == "unknown"

    def test_config_override_with_relative_path_resolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = _make_project(tmp_path)
        # Place the override in tmp_path and pass it as a relative
        # path against tmp_path.
        override_abs = tmp_path / "rel-override.yaml"
        override_abs.write_text("# rel\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ) as load_mock:
            inputs = load_project_inputs(
                "p1",
                config_override=Path("rel-override.yaml"),
                registry=registry,
            )

        called = load_mock.call_args.args[0]
        assert called.is_absolute()
        # Compare resolved paths (macOS may add /private prefix).
        assert called.resolve() == override_abs.resolve()
        assert (
            Path(inputs.metadata["config_override_path"]).resolve()
            == override_abs.resolve()
        )


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_metadata_always_carries_three_invariant_keys(
        self, tmp_path: Path
    ) -> None:
        registry = _make_project(tmp_path)

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ):
            inputs = load_project_inputs("p1", registry=registry)

        assert "project_id" in inputs.metadata
        assert "actor" in inputs.metadata
        assert "config_version_hash" in inputs.metadata

    def test_extra_metadata_cannot_override_invariants(
        self, tmp_path: Path
    ) -> None:
        """A caller passing a colliding ``project_id`` in extras should
        NOT be able to spoof the run's origin."""
        registry = _make_project(tmp_path)

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ):
            inputs = load_project_inputs(
                "p1",
                extra_metadata={
                    "project_id": "spoofed",
                    "actor": "spoofed",
                    "config_version_hash": "spoofed",
                },
                registry=registry,
            )

        assert inputs.metadata["project_id"] == "p1"
        assert inputs.metadata["actor"] != "spoofed"
        assert inputs.metadata["config_version_hash"] != "spoofed"

    def test_config_version_hash_is_deterministic(
        self, tmp_path: Path
    ) -> None:
        """Same YAML bytes → same hash. Pinned so the hash can be used
        as a stable lineage key in MLflow tags."""
        registry = _make_project(tmp_path, config_yaml="model:\n  name: stable\n")

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ):
            inputs1 = load_project_inputs("p1", registry=registry)
            inputs2 = load_project_inputs("p1", registry=registry)

        assert (
            inputs1.metadata["config_version_hash"]
            == inputs2.metadata["config_version_hash"]
        )
        # Sanity: it's a non-empty hex string.
        assert len(inputs1.metadata["config_version_hash"]) == 64

    def test_inputs_dataclass_is_frozen(self, tmp_path: Path) -> None:
        """Caller cannot mutate the inputs after the adapter returns —
        downstream code relying on snapshot semantics depends on it."""
        registry = _make_project(tmp_path)

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ):
            inputs = load_project_inputs("p1", registry=registry)

        with pytest.raises((AttributeError, Exception)):
            inputs.config = MagicMock()  # type: ignore[misc]

    def test_env_dict_is_a_copy_not_the_store_reference(
        self, tmp_path: Path
    ) -> None:
        """Sanity: returned ``env`` must be safe to mutate without
        affecting the on-disk env.json. (ProjectStore.read_env builds
        a fresh dict on every call.)"""
        registry = _make_project(tmp_path, env={"KEY": "v"})

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ):
            inputs = load_project_inputs("p1", registry=registry)
        # Mutate returned env.
        inputs.env["NEW"] = "added"
        # Re-read from disk; KEY still there, NEW wasn't persisted.
        store = ProjectStore(registry.resolve("p1").path)
        on_disk = store.read_env()
        assert on_disk == {"KEY": "v"}


# ---------------------------------------------------------------------------
# Logic-specific
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_config_override_keeps_project_env_and_metadata(
        self, tmp_path: Path
    ) -> None:
        """``-c X --project A`` semantics: env + metadata from A,
        config from X."""
        registry = _make_project(
            tmp_path, project_id="proj-a", env={"PROJECT_KEY": "from-A"}
        )
        override = tmp_path / "experimental.yaml"
        override.write_text("# experimental\n", encoding="utf-8")

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ):
            inputs = load_project_inputs(
                "proj-a", config_override=override, registry=registry
            )

        # Env from project A.
        assert inputs.env == {"PROJECT_KEY": "from-A"}
        # Metadata stamped with project_id proj-a.
        assert inputs.metadata["project_id"] == "proj-a"
        # Override breadcrumb captured.
        assert inputs.metadata["config_override_path"] == str(override.resolve())

    def test_actor_precedence_explicit_then_env_then_user(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pin precedence: explicit > RYOTENKAI_ACTOR > USER > unknown."""
        registry = _make_project(tmp_path)
        monkeypatch.setenv("RYOTENKAI_ACTOR", "ryo-actor")
        monkeypatch.setenv("USER", "user-os")

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ):
            # Explicit wins.
            i1 = load_project_inputs("p1", actor="explicit", registry=registry)
            assert i1.metadata["actor"] == "explicit"
            # Drop explicit → RYOTENKAI_ACTOR wins.
            i2 = load_project_inputs("p1", registry=registry)
            assert i2.metadata["actor"] == "ryo-actor"
            # Drop RYOTENKAI_ACTOR → USER wins.
            monkeypatch.delenv("RYOTENKAI_ACTOR", raising=False)
            i3 = load_project_inputs("p1", registry=registry)
            assert i3.metadata["actor"] == "user-os"

    def test_dependency_error_on_load_config_propagates(
        self, tmp_path: Path
    ) -> None:
        """When ``load_pipeline_config`` raises (e.g. malformed YAML),
        the adapter doesn't swallow it — the CLI's top-level handler
        renders a clean ``die``."""
        registry = _make_project(tmp_path)

        sentinel = ValueError("malformed YAML")
        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            side_effect=sentinel,
        ), pytest.raises(ValueError, match="malformed YAML"):
            load_project_inputs("p1", registry=registry)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


class TestRegression:
    def test_default_registry_used_when_not_passed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``registry`` is not provided, the adapter constructs a
        default-rooted one. Patch ``Path.home`` so the default points
        at the test's tmp dir rather than the user's real workspace."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # Build the project under the test-controlled workspace.
        registry = ProjectRegistry()
        project_path = registry.default_project_path("p1")
        store = ProjectStore(project_path)
        store.create(id="p1", name="P1")
        store.current_config_path.write_text("model:\n  name: x\n", encoding="utf-8")
        registry.register(project_id="p1", name="P1", path=project_path)

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ):
            # No ``registry=`` kwarg → adapter constructs default.
            inputs = load_project_inputs("p1")

        assert inputs.metadata["project_id"] == "p1"

    def test_env_json_round_trips_with_special_chars(
        self, tmp_path: Path
    ) -> None:
        """Regression: ProjectStore.read_env handles values with
        equals signs, spaces, JSON-special chars."""
        registry = _make_project(
            tmp_path,
            env={
                "URL": "https://x.example.com/?a=1&b=2",
                "QUOTE": 'has "quote"',
                "EMPTY": "",  # write_env strips these
                "OK_EMPTY_SPACES": "   ",  # not stripped (whitespace, not "")
            },
        )

        with patch(
            "src.workspace.projects.adapter.load_pipeline_config",
            return_value=_stub_config(),
        ):
            inputs = load_project_inputs("p1", registry=registry)

        assert inputs.env["URL"] == "https://x.example.com/?a=1&b=2"
        assert inputs.env["QUOTE"] == 'has "quote"'
        assert "EMPTY" not in inputs.env  # write_env drops "" values
        assert inputs.env["OK_EMPTY_SPACES"] == "   "


# ---------------------------------------------------------------------------
# Combinatorial: (project_exists × config_override × env_present)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "project_exists,config_override,env_present,expects_inputs",
    [
        (True, False, False, True),
        (True, False, True, True),
        (True, True, False, True),
        (True, True, True, True),
        (False, False, False, False),
        (False, False, True, False),
        (False, True, False, False),
        (False, True, True, False),
    ],
)
def test_combinatorial_load_paths(
    tmp_path: Path,
    project_exists: bool,
    config_override: bool,
    env_present: bool,
    expects_inputs: bool,
) -> None:
    if project_exists:
        env = {"KEY": "v"} if env_present else None
        registry = _make_project(tmp_path, env=env)
    else:
        registry = ProjectRegistry(root=tmp_path)

    override: Path | None = None
    if config_override:
        override = tmp_path / "ov.yaml"
        override.write_text("# ov\n", encoding="utf-8")

    with patch(
        "src.workspace.projects.adapter.load_pipeline_config",
        return_value=_stub_config(),
    ):
        if expects_inputs:
            inputs = load_project_inputs(
                "p1", config_override=override, registry=registry,
            )
            assert isinstance(inputs, ProjectInputs)
            if env_present:
                assert inputs.env == {"KEY": "v"}
            else:
                assert inputs.env == {}
            if config_override:
                assert "config_override_path" in inputs.metadata
        else:
            with pytest.raises(ProjectNotFoundError):
                load_project_inputs(
                    "p1", config_override=override, registry=registry,
                )


# ---------------------------------------------------------------------------
# Slim launch-inputs resolvers (replace ``load_project_inputs`` in step 7)
# ---------------------------------------------------------------------------


class TestResolveProjectLaunchInputs:
    """``resolve_project_launch_inputs`` — symmetric to load_project_inputs
    but doesn't load the YAML (worker subprocess does that)."""

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


class TestResolveLaunchInputsFromRunDir:
    """``resolve_project_launch_inputs_from_run_dir`` — walks up to find
    ``project.json``. Used by Web API and CLI resume/restart."""

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

    def test_boundary_walk_up_ignores_unrelated_project_json_higher_up(
        self, tmp_path: Path
    ) -> None:
        """Walk-up stops at the FIRST project.json — no false matches
        from a parent that happens to have a ``project.json``."""
        registry = _make_project(tmp_path, project_id="p1")
        store = ProjectStore(registry.default_project_path("p1"))
        # Create a deeply nested run dir; walk-up should find the
        # project's project.json, not climb past it.
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
