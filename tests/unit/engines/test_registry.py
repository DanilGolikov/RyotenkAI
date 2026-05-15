"""``EngineRegistry`` — discovery + class resolution + image lookup.

Uses synthetic manifests in tmp_path. Every test acquires a fresh registry
via ``EngineRegistry.from_filesystem(roots=[tmp_path])`` to avoid the
shipped (vLLM after PR-3) registry leaking in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from ryotenkai_engines.errors import EngineNotRegisteredError
from ryotenkai_engines.interfaces import BaseEngineConfig
from ryotenkai_engines.registry import EngineRegistry, LoadFailure

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers — write synthetic engine manifests under tmp_path
# ---------------------------------------------------------------------------


_MINIMAL_MANIFEST_TEMPLATE = """
schema_version = 1

[engine]
id = "{eid}"
name = "{name}"
version = "{version}"

[capabilities]
api_dialect              = "openai_compatible"
supports_lora            = true
supports_quantization    = false
supports_streaming       = true
supports_tensor_parallel = true
supported_dtypes         = ["bfloat16"]
default_port             = 8000

[entry_points.runtime]
module = "{runtime_mod}"
class  = "{runtime_cls}"

[entry_points.config_schema]
module = "{config_mod}"
class  = "{config_cls}"
"""


def _write_manifest(
    root: Path,
    *,
    eid: str,
    name: str | None = None,
    version: str = "1.0.0",
    runtime_mod: str = "ryotenkai_engines.tests._fakes",
    runtime_cls: str = "FakeRuntime",
    config_mod: str = "ryotenkai_engines.tests._fakes",
    config_cls: str = "FakeConfig",
) -> Path:
    folder = root / eid
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "engine.toml"
    manifest_path.write_text(
        _MINIMAL_MANIFEST_TEMPLATE.format(
            eid=eid,
            name=name or eid.upper(),
            version=version,
            runtime_mod=runtime_mod,
            runtime_cls=runtime_cls,
            config_mod=config_mod,
            config_cls=config_cls,
        ),
        encoding="utf-8",
    )
    return manifest_path


# ---------------------------------------------------------------------------
# 1. Positive — happy paths
# ---------------------------------------------------------------------------


class TestPositive:
    def test_discovers_single_engine(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, eid="alpha")
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        assert registry.list() == ("alpha",)

    def test_discovers_multiple_engines_sorted(self, tmp_path: Path) -> None:
        for eid in ["zeta", "alpha", "mu"]:
            _write_manifest(tmp_path, eid=eid)
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        assert registry.list() == ("alpha", "mu", "zeta")

    def test_get_manifest_returns_parsed_model(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, eid="alpha", name="Alpha Engine", version="2.1.3")
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        m = registry.get_manifest("alpha")
        assert m.engine.id == "alpha"
        assert m.engine.name == "Alpha Engine"
        assert m.engine.version == "2.1.3"

    def test_no_failures_on_clean_registry(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, eid="alpha")
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        assert registry.failures() == ()


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_get_manifest_unknown_raises(self, tmp_path: Path) -> None:
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        with pytest.raises(EngineNotRegisteredError) as exc_info:
            registry.get_manifest("nonexistent")
        assert "nonexistent" in str(exc_info.value)
        assert exc_info.value.context["reason"] == "engine_not_registered"
        assert exc_info.value.context["engine_id"] == "nonexistent"
        assert exc_info.value.status == 404

    def test_duplicate_engine_id_collected_as_failure(
        self, tmp_path: Path
    ) -> None:
        # Two manifests with same engine.id under different folders.
        _write_manifest(tmp_path, eid="alpha")
        # Second folder claims the same id.
        beta_folder = tmp_path / "beta"
        beta_folder.mkdir()
        (beta_folder / "engine.toml").write_text(
            _MINIMAL_MANIFEST_TEMPLATE.format(
                eid="alpha",   # ← duplicate
                name="ALPHA",
                version="1.0.0",
                runtime_mod="x.y",
                runtime_cls="Z",
                config_mod="x.y",
                config_cls="W",
            ),
            encoding="utf-8",
        )
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        # First wins; second is collected as failure (and folder mismatch
        # also gets collected — beta/ folder name vs alpha id).
        assert registry.list() == ("alpha",)
        failures = registry.failures()
        assert any("DuplicateEngineId" == f.exc_type or "EngineIdFolderMismatch" == f.exc_type for f in failures)

    def test_folder_id_mismatch_collected(self, tmp_path: Path) -> None:
        # Folder name 'foo' but manifest id 'bar'.
        folder = tmp_path / "foo"
        folder.mkdir()
        (folder / "engine.toml").write_text(
            _MINIMAL_MANIFEST_TEMPLATE.format(
                eid="bar",
                name="BAR",
                version="1.0.0",
                runtime_mod="x.y",
                runtime_cls="Z",
                config_mod="x.y",
                config_cls="W",
            ),
            encoding="utf-8",
        )
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        assert registry.list() == ()
        failures = registry.failures()
        assert len(failures) == 1
        assert failures[0].exc_type == "EngineIdFolderMismatch"

    def test_strict_mode_raises_on_folder_mismatch(self, tmp_path: Path) -> None:
        folder = tmp_path / "foo"
        folder.mkdir()
        (folder / "engine.toml").write_text(
            _MINIMAL_MANIFEST_TEMPLATE.format(
                eid="bar",
                name="BAR",
                version="1.0.0",
                runtime_mod="x.y",
                runtime_cls="Z",
                config_mod="x.y",
                config_cls="W",
            ),
            encoding="utf-8",
        )
        with pytest.raises(EngineNotRegisteredError, match="folder name") as exc_info:
            EngineRegistry.from_filesystem(roots=[tmp_path], strict=True)
        assert exc_info.value.context["reason"] == "engine_id_folder_mismatch"

    def test_malformed_toml_collected(self, tmp_path: Path) -> None:
        folder = tmp_path / "alpha"
        folder.mkdir()
        (folder / "engine.toml").write_text("this is === not toml at all", encoding="utf-8")
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        assert registry.list() == ()
        failures = registry.failures()
        assert len(failures) == 1
        assert "TOMLDecodeError" in failures[0].exc_type or "OSError" in failures[0].exc_type

    def test_invalid_manifest_collected(self, tmp_path: Path) -> None:
        folder = tmp_path / "alpha"
        folder.mkdir()
        # schema_version too high.
        (folder / "engine.toml").write_text(
            'schema_version = 99\n[engine]\nid = "alpha"\nname = "x"\nversion = "1"\n',
            encoding="utf-8",
        )
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        assert registry.list() == ()
        failures = registry.failures()
        assert len(failures) == 1
        assert failures[0].exc_type == "ValidationError"


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_empty_registry_root(self, tmp_path: Path) -> None:
        # No manifests in tmp_path.
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        assert registry.list() == ()
        assert registry.failures() == ()

    def test_nonexistent_root_silently_skipped(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist"
        registry = EngineRegistry.from_filesystem(roots=[nonexistent])
        assert registry.list() == ()

    def test_underscore_prefixed_dir_skipped(self, tmp_path: Path) -> None:
        """Folders starting with ``_`` are skipped (e.g. ``_config_union.py``
        sibling-like internal dirs)."""
        # Real engine
        _write_manifest(tmp_path, eid="alpha")
        # Internal dir that should be skipped
        internal = tmp_path / "_internal"
        internal.mkdir()
        (internal / "engine.toml").write_text("invalid", encoding="utf-8")
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        assert registry.list() == ("alpha",)

    def test_dotted_dir_skipped(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, eid="alpha")
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "engine.toml").write_text("invalid", encoding="utf-8")
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        assert registry.list() == ("alpha",)


# ---------------------------------------------------------------------------
# 4. Invariant — list sorted, manifests immutable
# ---------------------------------------------------------------------------


class TestInvariant:
    def test_list_is_sorted(self, tmp_path: Path) -> None:
        for eid in ["zeta", "delta", "alpha"]:
            _write_manifest(tmp_path, eid=eid)
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        ids = registry.list()
        assert list(ids) == sorted(ids)


# ---------------------------------------------------------------------------
# 5. Class resolution + drift detection
# ---------------------------------------------------------------------------


# Module-level fakes that the manifests' entry_points point at.
class FakeRuntime:
    """Stand-in for IInferenceEngine — has the required ClassVars."""
    engine_id = "alpha"
    config_class: type[BaseEngineConfig]   # set after class definition

    def get_capabilities(self):  # type: ignore[no-untyped-def]
        return None

    def build_launch_spec(self, **kwargs):  # type: ignore[no-untyped-def]
        return None

    def build_healthcheck_command(self, **kwargs):  # type: ignore[no-untyped-def]
        return ""

    def build_default_endpoint_url(self, **kwargs):  # type: ignore[no-untyped-def]
        return ""

    def validate_config(self, cfg):  # type: ignore[no-untyped-def]
        return None

    def prepare_model(self, **kwargs):  # type: ignore[no-untyped-def]
        return None


class FakeConfig(BaseEngineConfig):
    kind: Literal["alpha"] = "alpha"


FakeRuntime.config_class = FakeConfig


# Misaligned drift fakes
class DriftingRuntime:
    """engine_id doesn't match the manifest's engine.id."""
    engine_id = "wrong_id"
    config_class: type[BaseEngineConfig]

    def get_capabilities(self):  # type: ignore[no-untyped-def]
        return None
    def build_launch_spec(self, **kwargs):  # type: ignore[no-untyped-def]
        return None
    def build_healthcheck_command(self, **kwargs):  # type: ignore[no-untyped-def]
        return ""
    def build_default_endpoint_url(self, **kwargs):  # type: ignore[no-untyped-def]
        return ""
    def validate_config(self, cfg):  # type: ignore[no-untyped-def]
        return None
    def prepare_model(self, **kwargs):  # type: ignore[no-untyped-def]
        return None


DriftingRuntime.config_class = FakeConfig


class DriftingConfig(BaseEngineConfig):
    kind: Literal["wrong_kind"] = "wrong_kind"


# Patch fakes into the test module's namespace so importlib resolves them.
# NB: tests reference these via "ryotenkai_engines.tests.unit.test_registry".


class TestClassResolution:
    def test_get_runtime_resolves_and_validates(self, tmp_path: Path) -> None:
        # Manifests point at this test module's classes.
        _write_manifest(
            tmp_path,
            eid="alpha",
            runtime_mod="tests.unit.test_registry",
            runtime_cls="FakeRuntime",
            config_mod="tests.unit.test_registry",
            config_cls="FakeConfig",
        )
        # Hack — make the test module importable under a stable name.
        import sys

        sys.modules.setdefault("tests", type(sys)("tests"))  # type: ignore[arg-type]
        sys.modules.setdefault("tests.unit", type(sys)("tests.unit"))  # type: ignore[arg-type]
        sys.modules["tests.unit.test_registry"] = sys.modules[__name__]

        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        cls = registry.get_runtime("alpha")
        assert cls is FakeRuntime

    def test_get_config_class_resolves(self, tmp_path: Path) -> None:
        _write_manifest(
            tmp_path,
            eid="alpha",
            config_mod="tests.unit.test_registry",
            config_cls="FakeConfig",
            runtime_mod="tests.unit.test_registry",
            runtime_cls="FakeRuntime",
        )
        import sys
        sys.modules.setdefault("tests", type(sys)("tests"))  # type: ignore[arg-type]
        sys.modules.setdefault("tests.unit", type(sys)("tests.unit"))  # type: ignore[arg-type]
        sys.modules["tests.unit.test_registry"] = sys.modules[__name__]

        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        cls = registry.get_config_class("alpha")
        assert cls is FakeConfig

    def test_get_runtime_caches(self, tmp_path: Path) -> None:
        _write_manifest(
            tmp_path,
            eid="alpha",
            runtime_mod="tests.unit.test_registry",
            runtime_cls="FakeRuntime",
            config_mod="tests.unit.test_registry",
            config_cls="FakeConfig",
        )
        import sys
        sys.modules.setdefault("tests", type(sys)("tests"))  # type: ignore[arg-type]
        sys.modules.setdefault("tests.unit", type(sys)("tests.unit"))  # type: ignore[arg-type]
        sys.modules["tests.unit.test_registry"] = sys.modules[__name__]

        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        a = registry.get_runtime("alpha")
        b = registry.get_runtime("alpha")
        assert a is b

    def test_get_runtime_missing_module_raises(self, tmp_path: Path) -> None:
        _write_manifest(
            tmp_path,
            eid="alpha",
            runtime_mod="this.module.does.not.exist",
            runtime_cls="X",
        )
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        with pytest.raises(EngineNotRegisteredError, match="could not be resolved") as exc_info:
            registry.get_runtime("alpha")
        assert exc_info.value.context["reason"] == "engine_locator_resolve_failed"

    def test_get_runtime_engine_id_drift_raises(self, tmp_path: Path) -> None:
        _write_manifest(
            tmp_path,
            eid="alpha",
            runtime_mod="tests.unit.test_registry",
            runtime_cls="DriftingRuntime",
            config_mod="tests.unit.test_registry",
            config_cls="FakeConfig",
        )
        import sys
        sys.modules.setdefault("tests", type(sys)("tests"))  # type: ignore[arg-type]
        sys.modules.setdefault("tests.unit", type(sys)("tests.unit"))  # type: ignore[arg-type]
        sys.modules["tests.unit.test_registry"] = sys.modules[__name__]

        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        with pytest.raises(EngineNotRegisteredError, match="engine_id|runtime class") as exc_info:
            registry.get_runtime("alpha")
        assert exc_info.value.context["reason"] == "engine_runtime_id_drift"

    def test_get_config_class_kind_drift_raises(self, tmp_path: Path) -> None:
        _write_manifest(
            tmp_path,
            eid="alpha",
            runtime_mod="tests.unit.test_registry",
            runtime_cls="FakeRuntime",
            config_mod="tests.unit.test_registry",
            config_cls="DriftingConfig",
        )
        import sys
        sys.modules.setdefault("tests", type(sys)("tests"))  # type: ignore[arg-type]
        sys.modules.setdefault("tests.unit", type(sys)("tests.unit"))  # type: ignore[arg-type]
        sys.modules["tests.unit.test_registry"] = sys.modules[__name__]

        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        with pytest.raises(EngineNotRegisteredError, match="kind") as exc_info:
            registry.get_config_class("alpha")
        assert exc_info.value.context["reason"] == "engine_config_kind_drift"


# ---------------------------------------------------------------------------
# 6. Image lookup integration
# ---------------------------------------------------------------------------


class TestImageLookup:
    def test_get_image_uses_convention_default(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, eid="alpha", version="2.1.0")
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        image = registry.get_image("alpha", env={})
        assert image == "ryotenkai/inference-alpha:2.1.0"

    def test_get_image_unknown_raises(self, tmp_path: Path) -> None:
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        with pytest.raises(EngineNotRegisteredError) as exc_info:
            registry.get_image("nonexistent")
        assert exc_info.value.context["reason"] == "engine_not_registered"
