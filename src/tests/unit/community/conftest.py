"""Shared fixtures for community-plugin unit tests.

These fixtures factor out the repetitive scaffolding of "spin up a fake
community/ tree with one plugin in it, point the loader at it, and load":

- :func:`tmp_community_root` — empty disk-backed root, one per test.
- :func:`make_plugin_dir` — factory that drops a plugin folder
  (manifest.toml + plugin.py) under the temp root and returns the path.
- :func:`mock_catalog` — a :class:`CommunityCatalog` rooted at the temp
  tree, ready to call ``ensure_loaded()`` against.
- :func:`fake_secrets` — minimal :class:`Secrets` with arbitrary extras
  so tests can pass plugin-namespaced env keys without round-tripping
  through ``secrets.env``.

The fixtures are intentionally narrow — no automagic discovery, no
implicit registry mutation. Tests assemble exactly the artefacts they
need and rely on standard pytest cleanup for teardown.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.community.catalog import CommunityCatalog
from src.config.secrets.model import Secrets


# ---------------------------------------------------------------------------
# Disk-backed scaffolding
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_community_root(tmp_path: Path) -> Path:
    """Return a fresh temp directory with the standard ``community/`` skeleton.

    Each plugin kind directory exists but is empty — tests fill them via
    :func:`make_plugin_dir`. ``presets/`` is created too so that the
    catalogue's ``_compute_fingerprint`` walks all expected branches.
    """
    root = tmp_path / "community"
    for sub in ("validation", "evaluation", "reward", "reports", "presets"):
        (root / sub).mkdir(parents=True)
    return root


# Type alias for the plugin-dir factory: (kind, plugin_id, *,
# manifest_extras=None, plugin_source=None, class_name=None) -> Path.
MakePluginDir = Callable[..., Path]


@pytest.fixture
def make_plugin_dir(tmp_community_root: Path) -> MakePluginDir:
    """Drop a minimal plugin folder under ``tmp_community_root`` and return it.

    Usage::

        plugin_dir = make_plugin_dir(
            "validation",
            "tiny",
            manifest_extras={"params_schema": {"x": {"type": "integer"}}},
            plugin_source="class TinyPlugin(ValidationPlugin):\\n    pass",
            class_name="TinyPlugin",
        )

    Defaults aim for the smallest manifest that the loader will accept:
    a ``[plugin]`` block + a ``[plugin.entry_point]`` block pointing at a
    one-line class definition. Authors override ``manifest_extras`` to
    add ``[params_schema.*]`` / ``[[required_env]]`` / ``[secrets]``
    blocks specific to their test.
    """

    def _factory(
        kind: str,
        plugin_id: str,
        *,
        manifest_extras: str = "",
        plugin_extras: str = "",
        plugin_source: str | None = None,
        class_name: str = "TestPlugin",
        version: str = "1.0.0",
    ) -> Path:
        """Build a plugin folder under the temp community root.

        ``plugin_extras`` — extra lines inserted INTO the ``[plugin]`` block
        (e.g. ``'supported_strategies = ["grpo"]'`` for reward plugins
        whose manifest validator requires the field).
        ``manifest_extras`` — extra TOML appended at the top level
        (separate sections like ``[secrets]`` or ``[[required_env]]``).
        """
        plugin_dir = tmp_community_root / kind / plugin_id
        plugin_dir.mkdir(parents=True)
        plugin_block_extra = (
            "\n" + plugin_extras.strip() if plugin_extras else ""
        )
        manifest = textwrap.dedent(f"""
            [plugin]
            id = "{plugin_id}"
            kind = "{kind}"
            version = "{version}"{plugin_block_extra}

            [plugin.entry_point]
            module = "plugin"
            class = "{class_name}"
        """).strip()
        if manifest_extras:
            manifest += "\n\n" + manifest_extras.strip() + "\n"
        else:
            manifest += "\n"
        (plugin_dir / "manifest.toml").write_text(manifest)
        if plugin_source is None:
            plugin_source = f"class {class_name}:\n    pass\n"
        (plugin_dir / "plugin.py").write_text(plugin_source)
        return plugin_dir

    return _factory


@pytest.fixture
def mock_catalog(tmp_community_root: Path) -> CommunityCatalog:
    """Return a catalog rooted at the temp tree.

    The catalog is *not* pre-loaded — tests call ``ensure_loaded()`` /
    ``reload()`` themselves once they've populated the tree. This avoids
    fixture ordering pitfalls where ``make_plugin_dir`` runs *after*
    fixture setup would otherwise have triggered a load.
    """
    return CommunityCatalog(root=tmp_community_root)


# ---------------------------------------------------------------------------
# Secrets helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_secrets() -> Callable[..., Secrets]:
    """Return a factory that builds a :class:`Secrets` model with extras.

    Usage::

        secrets = fake_secrets(EVAL_FOO_KEY="abc", DTST_BAR_TOKEN="xyz")

    The :class:`Secrets` model is configured with ``extra="allow"`` so
    arbitrary plugin-namespaced keys flow through ``model_extra``. Keys
    are lowercased on the way in to match the convention enforced by
    :class:`PluginSecretsResolver`.
    """

    def _factory(**extras: Any) -> Secrets:
        normalised = {k.lower(): str(v) for k, v in extras.items()}
        return Secrets(**normalised)

    return _factory


# ---------------------------------------------------------------------------
# Global-state isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _invalidate_global_catalog_after_test():
    """Force the global ``catalog`` to reload before the next test, and
    restore the production ``community_libs`` namespace.

    Two leaks this guards against:

    1. **Plugin registries** — tests that build a fresh
       ``CommunityCatalog(root=tmp_path)`` call ``_populate_registries``
       which mutates the *module-level* singletons (``validation_registry``,
       ``evaluator_registry``, etc.). The global catalog's ``_loaded=True``
       flag would otherwise short-circuit the next ``ensure_loaded`` call
       and leave the registries empty for the next test. Resetting
       ``_loaded`` here costs nothing until the next test actually
       touches the global catalog.

    2. **``community_libs`` namespace** — tests that call
       ``preload_community_libs(tmp_libs)`` swap ``sys.modules['community_libs']``
       to a tmp tree. Pytest's per-plugin ``pytest_collectstart`` hooks
       (``community/<kind>/<plugin>/conftest.py``) execute the
       sibling ``plugin.py`` at collection time, and that module does
       ``from community_libs.<lib> import …`` at module level — so a
       polluted namespace at the moment of collection raises
       ``AttributeError`` *before* the next test's body even runs.
       We restore the namespace at teardown so the next collector sees
       the real tree.
    """
    import sys

    from src.community.constants import LIBS_NAMESPACE

    prefix = f"{LIBS_NAMESPACE}."
    snapshot = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == LIBS_NAMESPACE or name.startswith(prefix)
    }
    yield
    # Drop everything currently in the namespace, then restore the
    # snapshot. If the test never touched the namespace this is a
    # no-op (snapshot == current state).
    for name in list(sys.modules):
        if name == LIBS_NAMESPACE or name.startswith(prefix):
            del sys.modules[name]
    for name, module in snapshot.items():
        sys.modules[name] = module
    from src.community.catalog import catalog as _global_catalog

    _global_catalog._loaded = False


__all__ = [
    "fake_secrets",
    "make_plugin_dir",
    "mock_catalog",
    "tmp_community_root",
]
