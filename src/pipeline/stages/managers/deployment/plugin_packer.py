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
import tomllib
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.community.constants import COMMUNITY_ROOT, LIBS_DIR_NAME, PLUGIN_KIND_DIRS
from src.community.manifest import PluginManifest
from src.community.validate_manifest import validate_manifest_dir

if TYPE_CHECKING:
    from pathlib import Path

    from src.utils.config import PipelineConfig

__all__ = [
    "LibRef",
    "PluginPackError",
    "PluginPacker",
    "PluginRef",
]


# Mirror :mod:`src.community.pack` so files filtered out of an
# individual ``community pack`` archive are also filtered out of
# our combined runner payload. Lifted instead of imported because
# the originals are private and we'd rather copy three frozensets
# than couple two modules over implementation detail.
_EXCLUDED_DIRS = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)
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


@dataclass(frozen=True)
class LibRef:
    """A single shared lib to ship alongside the plugins that need it.

    Reward plugins can declare ``[[lib_requirements]]`` in their
    manifest pointing at packages under ``community/libs/`` (e.g.
    ``community_libs.helixql``). Without shipping the lib body, the
    pod-side ``import community_libs.helixql`` fails because
    :class:`PluginPacker` only ever copied ``community/reward/<id>/``.
    This struct travels with each lib resolved from a packed plugin's
    requirements.

    ``source_path`` may point at a folder OR a ``.zip`` archive — the
    packer handles both, mirroring :func:`src.community.libs._list_lib_entries`.
    """

    lib_id: str
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
        self._libs_root = community_root / LIBS_DIR_NAME

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
            plugin_id = str(phase.params.get("reward_plugin") or "").strip() if isinstance(phase.params, dict) else ""
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

    def determine_required_libs(
        self,
        plugin_refs: list[PluginRef],
    ) -> list[LibRef]:
        """Walk plugin manifests; collect unique :class:`LibRef`s.

        For each :class:`PluginRef` we parse ``manifest.toml``, read the
        ``[[lib_requirements]]`` blocks, and resolve each requirement's
        ``name`` to a folder OR ``.zip`` archive under
        ``community/libs/<name>/``. Folder wins on collision (mirrors
        :func:`src.community.libs._list_lib_entries`).

        Order is preserved: libs of the first plugin appear first,
        deduplicated across plugins so a lib referenced by two reward
        plugins is shipped exactly once.

        Version constraints declared in ``[[lib_requirements]]`` are
        NOT enforced here — that's :func:`src.community.loader._validate_lib_requirements_satisfied`'s
        job at catalog load time on the pod. The packer only resolves
        identity (name → source path) and ships the bytes.

        Raises:
            PluginPackError: a referenced lib has no matching folder
                or zip under ``community/libs/``. We fail loud rather
                than ship a payload that would ImportError at trainer
                startup.
        """
        seen: set[str] = set()
        refs: list[LibRef] = []

        for plugin in plugin_refs:
            manifest = _load_plugin_manifest(plugin.source_path)
            for req in manifest.lib_requirements:
                if req.name in seen:
                    continue
                source = self._resolve_lib_source(req.name)
                if source is None:
                    raise PluginPackError(
                        f"plugin {plugin.plugin_id!r} requires lib "
                        f"{req.name!r} but it is missing under "
                        f"{self._libs_root}",
                    )
                seen.add(req.name)
                refs.append(LibRef(lib_id=req.name, source_path=source.resolve()))
        return refs

    def _resolve_lib_source(self, lib_id: str) -> Path | None:
        """Find ``community/libs/<lib_id>/`` (folder) or ``<lib_id>.zip``.

        Folder wins on collision. Mirrors
        :func:`src.community.libs._list_lib_entries` so packer/loader
        agree on which source feeds a given lib id.
        """
        folder = self._libs_root / lib_id
        if folder.is_dir():
            return folder
        archive = self._libs_root / f"{lib_id}.zip"
        if archive.is_file():
            return archive
        return None

    # --- packing ----------------------------------------------------------

    def pack(
        self,
        plugin_refs: list[PluginRef],
        lib_refs: list[LibRef] | None = None,
    ) -> bytes:
        """Build a ZIP containing all plugin + lib refs; return bytes.

        Each plugin's manifest is validated (via the standalone
        :func:`validate_manifest_dir`) before any byte hits the
        archive. Files matching :data:`_EXCLUDED_DIRS` /
        :data:`_EXCLUDED_SUFFIXES` / :data:`_EXCLUDED_NAMES` are
        filtered out — same rules :mod:`src.community.pack` applies
        to single-plugin archives.

        Libs are NOT manifest-validated by this packer. The Mac side
        already loaded them via ``catalog`` for any prior dev work,
        and the pod side runs ``load_libs`` at trainer startup which
        will surface a broken lib manifest in the same place a
        normally-installed lib would.

        Empty plugin input is a programming error: the launcher should
        only call ``pack`` when there's at least one plugin to deliver.
        Empty ``lib_refs`` is fine — most reward plugins don't need
        any shared libs.
        """
        if not plugin_refs:
            raise PluginPackError(
                "pack() called with empty plugin_refs — caller must skip "
                "the multipart upload entirely when no reward "
                "plugins are required",
            )

        for ref in plugin_refs:
            result = validate_manifest_dir(ref.source_path)
            if not result.is_valid:
                # ManifestValidationResult formats issues as
                # ``[level] message``; surface the first to keep the
                # error focused. The full list lives on ``result``
                # if a caller wants to render it nicely.
                first = result.issues[0] if result.issues else None
                detail = first.message if first is not None else "unknown error"
                raise PluginPackError(
                    f"manifest validation failed for {ref.plugin_id!r}: " f"{detail}",
                )

        buf = io.BytesIO()
        # ZIP_DEFLATED matches the existing :mod:`src.community.pack`
        # output — a power user inspecting a pod-shipped payload
        # gets a familiar archive.
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for ref in plugin_refs:
                _write_plugin(zf, ref)
            for lib in lib_refs or ():
                _write_lib(zf, lib)
        return buf.getvalue()

    def pack_required(self) -> bytes:
        """Convenience: discover + pack plugins **and their libs** in one call.

        Returns ``b""`` when the config doesn't reference any reward
        plugins (e.g. SFT-only) — letting the caller skip the
        multipart upload. The empty-bytes return is the explicit
        signal "no payload needed", separate from the
        :class:`PluginPackError` raised when something is wrong.
        """
        plugin_refs = self.determine_required_plugins()
        if not plugin_refs:
            return b""
        lib_refs = self.determine_required_libs(plugin_refs)
        return self.pack(plugin_refs, lib_refs)


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


def _load_plugin_manifest(plugin_dir: Path) -> PluginManifest:
    """Parse ``<plugin_dir>/manifest.toml`` into a :class:`PluginManifest`.

    Raises :class:`PluginPackError` on read/parse/validation failure so
    the packer's failure surface stays uniform — callers don't have to
    distinguish ``OSError`` vs ``tomllib.TOMLDecodeError`` vs
    ``ValidationError``.
    """
    manifest_path = plugin_dir / "manifest.toml"
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PluginPackError(
            f"cannot read manifest at {manifest_path}: {exc}",
        ) from exc

    try:
        payload = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise PluginPackError(
            f"manifest at {manifest_path} is invalid TOML: {exc}",
        ) from exc

    try:
        return PluginManifest.model_validate(payload)
    except Exception as exc:
        raise PluginPackError(
            f"manifest at {manifest_path} failed validation: {exc}",
        ) from exc


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


def _write_lib(zf: zipfile.ZipFile, ref: LibRef) -> None:
    """Write a lib's source tree into ``zf`` under ``libs/<lib_id>/``.

    Two source shapes are supported (mirrors
    :func:`src.community.libs._list_lib_entries`):

    - ``<community>/libs/<lib_id>/`` (folder) — recurse like
      :func:`_write_plugin`, applying the same exclusion rules so
      ``__pycache__`` and ``.pyc`` artefacts don't bloat the payload.
    - ``<community>/libs/<lib_id>.zip`` — copy each member through
      preserving its in-archive layout. Excluded by directory name
      (``__pycache__``) and suffix to keep parity with the folder
      path.
    """
    base = f"{LIBS_DIR_NAME}/{ref.lib_id}"

    if ref.source_path.is_dir():
        for path in sorted(ref.source_path.rglob("*")):
            if not path.is_file():
                continue
            if _is_excluded(path, root=ref.source_path):
                continue
            rel = path.relative_to(ref.source_path)
            archive_name = f"{base}/{rel.as_posix()}"
            zf.write(path, archive_name)
        return

    # Archive form. Re-zip into the unified payload so the unpacker
    # only deals with one wire format.
    with zipfile.ZipFile(ref.source_path, "r") as src_zip:
        for info in sorted(src_zip.infolist(), key=lambda i: i.filename):
            if info.is_dir():
                continue
            rel_name = info.filename
            # Mirror _is_excluded() rules without going through Path so
            # the in-archive forward-slash convention stays intact.
            parts = rel_name.split("/")
            if any(p in _EXCLUDED_DIRS for p in parts):
                continue
            name = parts[-1]
            if name in _EXCLUDED_NAMES:
                continue
            if any(name.endswith(suf) for suf in _EXCLUDED_SUFFIXES):
                continue
            with src_zip.open(info) as src:
                zf.writestr(f"{base}/{rel_name}", src.read())
