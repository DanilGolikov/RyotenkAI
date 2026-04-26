"""Pack reward plugins for delivery to the in-pod runner — Phase 6.1.

Closes risk **R-2** (plugin delivery gap): :class:`CodeSyncer`'s
``REQUIRED_MODULES`` deliberately does NOT include ``community/``,
because shipping the entire 100+-plugin catalog through rsync per
deployment would be slow and would tangle Mac-side plugin
authoring with pod-side runtime needs. Instead, the launcher
walks the active :class:`PipelineConfig`, finds only the reward
plugins this run actually uses, and packs them into a single ZIP
that travels with the multipart ``POST /jobs`` body. The pod-side
:class:`PluginUnpacker` extracts that ZIP into
``/workspace/community/`` so :func:`catalog.ensure_loaded` finds
the plugins at trainer startup.

What we DON'T pack:

- validation / evaluation / reports plugins — those run on Mac
  in the dataset / evaluator stages, never on the pod.
- presets — already resolved into the final YAML before launch.
- plugins not referenced by any phase — keep the payload tight.

ZIP layout (matches the runner's expected ``community/`` tree)::

    plugins.zip
    ├── reward/
    │   ├── <plugin_id_a>/
    │   │   ├── manifest.toml
    │   │   ├── plugin.py
    │   │   └── ...
    │   └── <plugin_id_b>/
    │       └── ...

Reused infrastructure (do NOT reinvent):

- :func:`src.community.validate_manifest.validate_manifest_dir` —
  fail-fast validation; a broken manifest blocks pack rather than
  shipping a payload that the pod-side unpacker would reject.
- File-filtering rules from :mod:`src.community.pack` (excluded
  dirs, suffixes, names) — kept consistent so the pack-then-install
  path behaves identically whether plugins flow via this packer
  or the existing ``ryotenkai community pack`` CLI.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.community.constants import COMMUNITY_ROOT, PLUGIN_KIND_DIRS
from src.community.validate_manifest import validate_manifest_dir

if TYPE_CHECKING:
    from pathlib import Path

    from src.utils.config import PipelineConfig

__all__ = [
    "PluginPackError",
    "PluginPacker",
    "PluginRef",
]


# Mirror :mod:`src.community.pack` so files filtered out of an
# individual ``community pack`` archive are also filtered out of
# our combined runner payload. Lifted instead of imported because
# the originals are private and we'd rather copy three frozensets
# than couple two modules over implementation detail.
_EXCLUDED_DIRS = frozenset({
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
})
_EXCLUDED_SUFFIXES = (".pyc", ".pyo")
_EXCLUDED_NAMES = frozenset({".DS_Store", "Thumbs.db"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PluginRef:
    """A single plugin to ship.

    ``kind`` is currently always ``"reward"`` (only reward plugins
    travel to the pod), but the field is kept first-class so a
    later phase can reuse the packer for validation plugins that
    might run as a sidecar inside the runner.

    ``source_path`` is the absolute path to the plugin folder
    (e.g. ``/repo/community/reward/helixql_compiler_semantic``).
    """

    kind: str
    plugin_id: str
    source_path: Path


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PluginPackError(RuntimeError):
    """Pack failed — missing source folder, broken manifest, etc.

    Always surfaced before any bytes are sent over the wire so a
    bad config never produces a half-valid payload the pod side
    would have to reject mid-stream.
    """


# ---------------------------------------------------------------------------
# Packer
# ---------------------------------------------------------------------------


class PluginPacker:
    """Pack the reward plugins referenced by a :class:`PipelineConfig`.

    Two-call usage:

    1. :meth:`determine_required_plugins` — walks the strategy chain
       and returns the unique :class:`PluginRef` list. Useful by
       itself for preflight (e.g. failing fast when a referenced
       plugin folder is missing).
    2. :meth:`pack` — turns a list of refs into a single ZIP byte
       string ready for :meth:`src.api.clients.JobClient.submit_job`.

    Or one-shot via :meth:`pack_required` which composes both.

    ``community_root`` defaults to the canonical
    :data:`COMMUNITY_ROOT`. Tests override it to a tmp_path with
    fixture plugins so the packer doesn't depend on the real
    catalogue layout.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        community_root: Path = COMMUNITY_ROOT,
    ) -> None:
        self._config = config
        self._community_root = community_root

    # --- discovery --------------------------------------------------------

    def determine_required_plugins(self) -> list[PluginRef]:
        """Walk the strategy chain; collect unique reward plugin refs.

        Order is preserved (first occurrence wins) so the resulting
        ZIP is deterministic across runs of the same config — useful
        for diffing payloads in incident postmortems.

        Raises:
            PluginPackError: a referenced plugin's source folder
                does not exist under ``community_root``. Better to
                fail loud here than to ship a payload that's silently
                missing files.
        """
        seen: set[tuple[str, str]] = set()
        refs: list[PluginRef] = []
        kind_dir = PLUGIN_KIND_DIRS["reward"]
        kind_root = self._community_root / kind_dir

        for phase in self._config.training.get_strategy_chain():
            plugin_id = (
                str(phase.params.get("reward_plugin") or "").strip()
                if isinstance(phase.params, dict) else ""
            )
            if not plugin_id:
                continue
            key = ("reward", plugin_id)
            if key in seen:
                continue
            seen.add(key)
            source = kind_root / plugin_id
            if not source.is_dir():
                raise PluginPackError(
                    f"reward plugin {plugin_id!r} referenced by phase "
                    f"{phase.strategy_type!r} but folder is missing: "
                    f"{source}",
                )
            refs.append(
                PluginRef(
                    kind="reward",
                    plugin_id=plugin_id,
                    source_path=source.resolve(),
                ),
            )
        return refs

    # --- packing ----------------------------------------------------------

    def pack(self, refs: list[PluginRef]) -> bytes:
        """Build a ZIP containing all ``refs`` and return the bytes.

        Each plugin's manifest is validated (via the standalone
        :func:`validate_manifest_dir`) before any byte hits the
        archive. Files matching :data:`_EXCLUDED_DIRS` /
        :data:`_EXCLUDED_SUFFIXES` / :data:`_EXCLUDED_NAMES` are
        filtered out — same rules :mod:`src.community.pack` applies
        to single-plugin archives.

        Empty input is a programming error: the launcher should
        only call ``pack`` when there's at least one plugin to
        deliver. We raise so a silent empty payload doesn't slip
        past the test suite.
        """
        if not refs:
            raise PluginPackError(
                "pack() called with empty refs — caller must skip "
                "the multipart upload entirely when no reward "
                "plugins are required",
            )

        for ref in refs:
            result = validate_manifest_dir(ref.source_path)
            if not result.is_valid:
                # ManifestValidationResult formats issues as
                # ``[level] message``; surface the first to keep the
                # error focused. The full list lives on ``result``
                # if a caller wants to render it nicely.
                first = result.issues[0] if result.issues else None
                detail = first.message if first is not None else "unknown error"
                raise PluginPackError(
                    f"manifest validation failed for {ref.plugin_id!r}: "
                    f"{detail}",
                )

        buf = io.BytesIO()
        # ZIP_DEFLATED matches the existing :mod:`src.community.pack`
        # output — a power user inspecting a pod-shipped payload
        # gets a familiar archive.
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for ref in refs:
                _write_plugin(zf, ref)
        return buf.getvalue()

    def pack_required(self) -> bytes:
        """Convenience: discover + pack in one call.

        Returns ``b""`` when the config doesn't reference any reward
        plugins (e.g. SFT-only) — letting the caller skip the
        multipart upload. The empty-bytes return is the explicit
        signal "no payload needed", separate from the
        :class:`PluginPackError` raised when something is wrong.
        """
        refs = self.determine_required_plugins()
        if not refs:
            return b""
        return self.pack(refs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_excluded(path: Path, *, root: Path) -> bool:
    """Mirror :func:`src.community.pack._is_excluded`."""
    rel = path.relative_to(root)
    if any(part in _EXCLUDED_DIRS for part in rel.parts):
        return True
    if path.name in _EXCLUDED_NAMES:
        return True
    return any(path.name.endswith(suf) for suf in _EXCLUDED_SUFFIXES)


def _write_plugin(zf: zipfile.ZipFile, ref: PluginRef) -> None:
    """Write every file under ``ref.source_path`` into ``zf``,
    rooted at ``<kind>/<plugin_id>/``.

    Sorted traversal so the ZIP byte-content is deterministic for
    a given source tree — same input yields same hash, helpful for
    cache-coherency checks on the pod side later.
    """
    base = f"{ref.kind}/{ref.plugin_id}"
    for path in sorted(ref.source_path.rglob("*")):
        if not path.is_file():
            continue
        if _is_excluded(path, root=ref.source_path):
            continue
        rel = path.relative_to(ref.source_path)
        archive_name = f"{base}/{rel.as_posix()}"
        zf.write(path, archive_name)
