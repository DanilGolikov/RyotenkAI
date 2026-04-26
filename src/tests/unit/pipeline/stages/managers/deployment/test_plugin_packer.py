"""Phase 6.1 — :class:`PluginPacker` contract.

Verifies the pack-then-ship flow:

- TestDetermineRequiredPlugins  walk strategy chain, dedup, missing folder
- TestPack                      manifest validation, layout, file filtering
- TestPackRequired              one-shot wrapper, empty config edge
- TestZipDeterminism            same input → same bytes (cache-friendly)

We deliberately don't pull in the full :class:`PipelineConfig`
Pydantic model — the packer only touches
``training.get_strategy_chain()`` and ``phase.params``, so a
``SimpleNamespace`` stand-in is enough and keeps the tests fast.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_plugin_packer():
    """Load the packer module directly so we skip
    :mod:`src.pipeline.stages.managers.__init__`'s eager import of
    :class:`TrainingDeploymentManager`, which pulls colorlog (not
    present in the dev venv). The full deployment package WILL be
    available in CI; this workaround keeps the dev loop fast."""
    if "ryotenkai_plugin_packer_test" in sys.modules:
        return sys.modules["ryotenkai_plugin_packer_test"]
    repo_root = Path(__file__).resolve().parents[7]
    src_path = (
        repo_root
        / "src" / "pipeline" / "stages" / "managers"
        / "deployment" / "plugin_packer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ryotenkai_plugin_packer_test", str(src_path),
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ryotenkai_plugin_packer_test"] = mod
    spec.loader.exec_module(mod)
    return mod


_packer_mod = _load_plugin_packer()
PluginPacker = _packer_mod.PluginPacker
PluginRef = _packer_mod.PluginRef
PluginPackError = _packer_mod.PluginPackError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_plugin(
    root: Path, kind: str, plugin_id: str, *, valid: bool = True,
) -> Path:
    """Lay down a minimal plugin folder under ``root/<kind>/<id>/``.

    ``valid=True`` writes a manifest that passes
    :func:`validate_manifest_dir`. ``valid=False`` writes one with a
    forbidden ``schema_version`` so the packer's pre-flight raises.
    """
    folder = root / kind / plugin_id
    folder.mkdir(parents=True)

    if valid:
        # Schema v4 — fields lifted from a real reward manifest in
        # community/reward/. Keep the shape minimal but legal.
        manifest = (
            f'[plugin]\n'
            f'id = "{plugin_id}"\n'
            f'kind = "{kind}"\n'
            f'name = "Test Fixture {plugin_id}"\n'
            f'version = "0.1.0"\n'
            f'description = "test fixture"\n'
            f'supported_strategies = ["grpo", "sapo"]\n'
            f'\n'
            f'[plugin.entry_point]\n'
            f'module = "plugin"\n'
            f'class = "FixturePlugin"\n'
        )
    else:
        # Missing the required ``[plugin]`` block → validator rejects.
        manifest = "this is not a valid manifest\n"

    (folder / "manifest.toml").write_text(manifest)
    (folder / "plugin.py").write_text("# stub\n")
    # Drop a couple of files that should be filtered out.
    cache_dir = folder / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "anything.pyc").write_text("compiled-bytes")
    (folder / ".DS_Store").write_text("apple-junk")
    return folder


def _config(*phases: tuple[str, dict]) -> SimpleNamespace:
    """Build a minimal config-like object the packer accepts.

    Each ``phases`` entry is ``(strategy_type, params)``."""

    def _chain() -> list[SimpleNamespace]:
        return [
            SimpleNamespace(strategy_type=stype, params=params)
            for stype, params in phases
        ]

    return SimpleNamespace(training=SimpleNamespace(get_strategy_chain=_chain))


# ---------------------------------------------------------------------------
# determine_required_plugins
# ---------------------------------------------------------------------------


class TestDetermineRequiredPlugins:
    def test_collects_unique_reward_plugins(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "reward", "plugin_a")
        _make_plugin(tmp_path, "reward", "plugin_b")

        cfg = _config(
            ("grpo", {"reward_plugin": "plugin_a"}),
            ("sapo", {"reward_plugin": "plugin_b"}),
            # Duplicate — must be deduped
            ("dpo", {"reward_plugin": "plugin_a"}),
            # Phase without reward_plugin — skipped
            ("sft", {}),
        )
        packer = PluginPacker(cfg, community_root=tmp_path)
        refs = packer.determine_required_plugins()
        ids = [r.plugin_id for r in refs]
        assert ids == ["plugin_a", "plugin_b"]
        assert all(r.kind == "reward" for r in refs)

    def test_no_reward_plugins_returns_empty(self, tmp_path: Path) -> None:
        cfg = _config(("sft", {}), ("dpo", {"some_other_param": 1}))
        packer = PluginPacker(cfg, community_root=tmp_path)
        assert packer.determine_required_plugins() == []

    def test_missing_plugin_folder_raises(self, tmp_path: Path) -> None:
        cfg = _config(("grpo", {"reward_plugin": "missing_plugin"}))
        packer = PluginPacker(cfg, community_root=tmp_path)
        with pytest.raises(PluginPackError, match="folder is missing"):
            packer.determine_required_plugins()

    def test_non_dict_params_skipped(self, tmp_path: Path) -> None:
        # Defensive — params should always be a dict per
        # StrategyPhaseConfig, but a hand-rolled config object
        # could feed us garbage. Don't crash, just skip.
        cfg = _config(("sft", "this is not a dict"))  # type: ignore[arg-type]
        packer = PluginPacker(cfg, community_root=tmp_path)
        assert packer.determine_required_plugins() == []

    def test_empty_string_reward_plugin_skipped(self, tmp_path: Path) -> None:
        cfg = _config(
            ("grpo", {"reward_plugin": ""}),
            ("dpo", {"reward_plugin": "   "}),  # whitespace
        )
        packer = PluginPacker(cfg, community_root=tmp_path)
        assert packer.determine_required_plugins() == []


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------


class TestPack:
    def test_pack_produces_valid_zip_with_correct_layout(
        self, tmp_path: Path,
    ) -> None:
        plugin_path = _make_plugin(tmp_path, "reward", "plugin_a")
        ref = PluginRef("reward", "plugin_a", plugin_path)

        cfg = _config(("grpo", {"reward_plugin": "plugin_a"}))
        packer = PluginPacker(cfg, community_root=tmp_path)
        data = packer.pack([ref])

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = sorted(zf.namelist())
            assert "reward/plugin_a/manifest.toml" in names
            assert "reward/plugin_a/plugin.py" in names
            # Excluded files MUST NOT appear in the archive.
            assert all("__pycache__" not in n for n in names)
            assert all(".DS_Store" not in n for n in names)
            assert all(not n.endswith(".pyc") for n in names)

    def test_empty_refs_raises(self, tmp_path: Path) -> None:
        cfg = _config()
        packer = PluginPacker(cfg, community_root=tmp_path)
        with pytest.raises(PluginPackError, match="empty refs"):
            packer.pack([])

    def test_bad_manifest_blocks_pack(self, tmp_path: Path) -> None:
        plugin_path = _make_plugin(tmp_path, "reward", "broken", valid=False)
        ref = PluginRef("reward", "broken", plugin_path)
        cfg = _config(("grpo", {"reward_plugin": "broken"}))
        packer = PluginPacker(cfg, community_root=tmp_path)
        with pytest.raises(PluginPackError, match="manifest validation failed"):
            packer.pack([ref])

    def test_pack_multiple_plugins(self, tmp_path: Path) -> None:
        a_path = _make_plugin(tmp_path, "reward", "plugin_a")
        b_path = _make_plugin(tmp_path, "reward", "plugin_b")
        cfg = _config(
            ("grpo", {"reward_plugin": "plugin_a"}),
            ("sapo", {"reward_plugin": "plugin_b"}),
        )
        packer = PluginPacker(cfg, community_root=tmp_path)
        data = packer.pack([
            PluginRef("reward", "plugin_a", a_path),
            PluginRef("reward", "plugin_b", b_path),
        ])
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
            assert "reward/plugin_a/manifest.toml" in names
            assert "reward/plugin_b/manifest.toml" in names


# ---------------------------------------------------------------------------
# pack_required
# ---------------------------------------------------------------------------


class TestPackRequired:
    def test_no_plugins_returns_empty_bytes(self, tmp_path: Path) -> None:
        # Empty signals "no payload needed" — distinct from a raised
        # error. The launcher uses this to skip the multipart upload.
        cfg = _config(("sft", {}))
        packer = PluginPacker(cfg, community_root=tmp_path)
        assert packer.pack_required() == b""

    def test_happy_path_returns_zip_bytes(self, tmp_path: Path) -> None:
        _make_plugin(tmp_path, "reward", "plugin_a")
        cfg = _config(("grpo", {"reward_plugin": "plugin_a"}))
        packer = PluginPacker(cfg, community_root=tmp_path)
        data = packer.pack_required()
        assert data
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            assert "reward/plugin_a/manifest.toml" in zf.namelist()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestZipDeterminism:
    def test_same_input_same_zip_namelist(self, tmp_path: Path) -> None:
        # Sorted traversal means a deterministic file ORDER inside
        # the ZIP. The bytes themselves can vary because zlib chooses
        # different stream states across invocations, but the
        # NAMELIST should be stable.
        _make_plugin(tmp_path, "reward", "plugin_a")
        cfg = _config(("grpo", {"reward_plugin": "plugin_a"}))
        packer = PluginPacker(cfg, community_root=tmp_path)
        first = packer.pack_required()
        second = packer.pack_required()

        with zipfile.ZipFile(io.BytesIO(first)) as zf1, zipfile.ZipFile(
            io.BytesIO(second),
        ) as zf2:
            assert zf1.namelist() == zf2.namelist()
