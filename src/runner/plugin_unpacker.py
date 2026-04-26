"""Pod-side plugin extractor — Phase 6.2.

The Mac client packs reward plugins into a single ZIP via
:class:`src.pipeline.stages.managers.deployment.plugin_packer.PluginPacker`
and ships it as the multipart ``plugins_payload`` part of
``POST /api/v1/jobs``. This module is the receiving end on the pod —
it extracts that ZIP into ``<workspace>/community/<kind>/<id>/`` so
that when the trainer subprocess starts up later,
:func:`src.community.catalog.ensure_loaded` finds the plugins on
disk in the layout it expects.

Why we DON'T import :func:`src.community.install.install_local`:

The :mod:`src.community` package transitively pulls in
:mod:`src.utils.logger` (colorlog) plus a long chain of
infrastructure modules that the runner's bare docker image does
not ship. Pulling them in just to reach ``install_local`` would
roughly double the runner image size and add a startup-time
import cost the runner pays whether or not plugins are needed.

The Mac side already validated every manifest in
:meth:`PluginPacker.pack`; the pod side trusts that and limits
itself to safe extraction. If a plugin slipped past Mac validation
the trainer subprocess will surface the issue at
:func:`catalog.ensure_loaded` — same place a normally-installed
plugin would surface a manifest break.

Wire format expected (matches :class:`PluginPacker`)::

    plugins.zip
    ├── reward/
    │   ├── <plugin_id_a>/
    │   │   ├── manifest.toml
    │   │   ├── plugin.py
    │   │   └── ...
    │   └── <plugin_id_b>/
    │       └── ...

Anything outside the ``<kind>/<id>/`` layout (loose files at the
ZIP root, unknown kinds) is logged and skipped — defensive against
malformed payloads. The zip-bomb / path-traversal protections are
inline:

- :meth:`zipfile.ZipFile.infolist` is iterated; each entry's
  resolved path is checked to be under the destination dir before
  any bytes are written.
- Symlinks are rejected; only regular files and directories.
- A per-archive uncompressed size cap (:data:`_MAX_UNCOMPRESSED_BYTES`)
  prevents a maliciously-crafted payload from exhausting disk.
"""

from __future__ import annotations

import io
import logging
import shutil
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "RECOGNISED_KINDS",
    "PluginUnpackError",
    "PluginUnpackResult",
    "PluginUnpacker",
]


logger = logging.getLogger(__name__)


# Plugin kinds we accept on the pod. ``reward`` is the only one
# that ever travels here today; the tuple is centralised so adding
# a future kind (e.g. a runtime-loaded validation plugin) is a
# one-line change with a clear contract.
RECOGNISED_KINDS: tuple[str, ...] = ("reward",)


# Soft cap on total uncompressed bytes per payload. 256 MiB is
# overkill for the plugin sizes we ship (typical reward plugin is
# under 200 KiB) but generous enough that a future bundled-models
# scenario doesn't blow up. The cap exists to defend against zip
# bombs from a compromised Mac client, NOT to enforce policy.
_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024


# ---------------------------------------------------------------------------
# Result / errors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PluginUnpackResult:
    """What :meth:`PluginUnpacker.unpack` produced.

    ``installed`` lists ``<kind>/<plugin_id>`` strings — handy for
    logging and for the runner to publish a structured event so the
    Mac client can verify the right plugins landed.

    ``skipped`` lists archive entries the unpacker chose to ignore
    (loose files at the root, unknown kinds, paths that failed
    safety checks). Tests use this to assert defensive behaviour.
    """

    installed: tuple[str, ...]
    skipped: tuple[str, ...]
    total_bytes: int


class PluginUnpackError(RuntimeError):
    """Unpack failed — corrupt ZIP, oversize payload, or path-traversal
    attempt. Distinct from ``shutil`` / ``zipfile`` errors so the
    caller can distinguish "the payload was bad" from "I/O glitch".
    """


# ---------------------------------------------------------------------------
# Unpacker
# ---------------------------------------------------------------------------


class PluginUnpacker:
    """Extract a multi-plugin ZIP into the runner's community root.

    Construct once per runner instance and call
    :meth:`unpack` per ``POST /jobs`` invocation — the unpacker is
    stateless beyond its config (no per-job mutable state, no caches
    that could leak between requests).

    Args:
        workspace_dir: persistent volume mount on the pod
            (default ``/workspace``). The community root lives at
            ``<workspace>/community/`` — same convention as
            :data:`src.community.constants.COMMUNITY_ROOT` so the
            trainer subprocess finds plugins via the same path it
            uses on Mac.
        max_uncompressed_bytes: zip-bomb cap. Tests override this
            to assert the cap is enforced.

    Thread safety: not thread-safe — assumes one unpack call per
    job, serialised by the supervisor's "one active job at a time"
    invariant.
    """

    def __init__(
        self,
        workspace_dir: Path,
        *,
        max_uncompressed_bytes: int = _MAX_UNCOMPRESSED_BYTES,
    ) -> None:
        self._workspace_dir = workspace_dir
        self._max_uncompressed_bytes = max_uncompressed_bytes

    @property
    def community_root(self) -> Path:
        """Where plugins land on disk: ``<workspace>/community``."""
        return self._workspace_dir / "community"

    def unpack(
        self,
        payload: bytes,
        *,
        force: bool = True,
    ) -> PluginUnpackResult:
        """Extract ``payload`` into :attr:`community_root`.

        Args:
            payload: ZIP bytes from the multipart upload. An empty
                bytes object is allowed and short-circuits to a
                no-op result — the launcher passes ``b""`` for jobs
                with no reward plugins (e.g. SFT-only).
            force: when ``True`` (the default) overwrite an existing
                ``<kind>/<plugin_id>/`` folder in the community
                root. The runner is single-tenant per job so we
                always want the freshest plugin payload; the flag
                exists mainly for tests.

        Returns:
            :class:`PluginUnpackResult` summarising what landed.

        Raises:
            PluginUnpackError: corrupt ZIP, total uncompressed
                size > ``max_uncompressed_bytes``, or a path-traversal
                attempt.
        """
        if not payload:
            # Nothing to unpack — make sure community root exists
            # so the trainer's catalog walk doesn't crash with
            # FileNotFoundError when there are zero plugins.
            self.community_root.mkdir(parents=True, exist_ok=True)
            return PluginUnpackResult(
                installed=(), skipped=(), total_bytes=0,
            )

        try:
            archive = zipfile.ZipFile(io.BytesIO(payload))
        except zipfile.BadZipFile as exc:
            raise PluginUnpackError(
                f"plugins payload is not a valid ZIP: {exc}",
            ) from exc

        with archive:
            return self._unpack_archive(archive, force=force)

    # --- internals --------------------------------------------------------

    def _unpack_archive(
        self,
        archive: zipfile.ZipFile,
        *,
        force: bool,
    ) -> PluginUnpackResult:
        """Validate every entry, then copy what passes.

        Two-pass:
        1. Walk infolist(); collect (kind, plugin_id, relpath, size)
           tuples. Reject path traversal / oversize / unknown kinds.
        2. Extract: clear existing target dirs, write each file.

        Two passes because a partial extract on a malicious payload
        is worse than no extract at all — we want to commit
        atomically per plugin once we know the whole archive is
        legal.
        """
        # Group entries by (kind, plugin_id) so each plugin can be
        # written atomically.
        per_plugin: dict[tuple[str, str], list[tuple[str, zipfile.ZipInfo]]] = {}
        skipped: list[str] = []
        total = 0
        target_root = self.community_root.resolve()

        for info in archive.infolist():
            if info.is_dir():
                # Directories are recreated implicitly when files
                # land underneath; standalone dir entries don't
                # need their own write.
                continue

            name = info.filename
            # Reject absolute paths and traversal.
            if name.startswith("/") or ".." in name.split("/"):
                raise PluginUnpackError(
                    f"path traversal attempt in plugins payload: {name!r}",
                )
            # External attribute hints at symlinks (POSIX 0xA0000000).
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise PluginUnpackError(
                    f"plugins payload contains a symlink: {name!r}",
                )

            parts = name.split("/")
            # Need at least <kind>/<plugin_id>/<file>; anything
            # shallower (or empty) is malformed and we skip.
            if len(parts) < 3 or not parts[2]:
                skipped.append(name)
                continue
            kind, plugin_id = parts[0], parts[1]
            if kind not in RECOGNISED_KINDS:
                skipped.append(name)
                continue
            relpath = "/".join(parts[2:])
            if not relpath:
                skipped.append(name)
                continue

            # Resolved destination must stay under community_root —
            # belt-and-braces on top of the path-traversal check.
            dest = (target_root / kind / plugin_id / relpath).resolve()
            try:
                dest.relative_to(target_root)
            except ValueError:
                raise PluginUnpackError(
                    f"path resolves outside community root: {name!r}",
                ) from None

            total += info.file_size
            if total > self._max_uncompressed_bytes:
                raise PluginUnpackError(
                    f"plugins payload exceeds {self._max_uncompressed_bytes} "
                    f"uncompressed bytes (zip-bomb defence)",
                )

            per_plugin.setdefault((kind, plugin_id), []).append((relpath, info))

        # Pass 2 — extract.
        target_root.mkdir(parents=True, exist_ok=True)
        installed: list[str] = []
        for (kind, plugin_id), entries in sorted(per_plugin.items()):
            plugin_dir = target_root / kind / plugin_id
            if plugin_dir.exists():
                if not force:
                    raise PluginUnpackError(
                        f"plugin {kind}/{plugin_id} already installed; "
                        f"force=False",
                    )
                shutil.rmtree(plugin_dir)
            plugin_dir.mkdir(parents=True)
            for relpath, info in entries:
                dest = plugin_dir / relpath
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            installed.append(f"{kind}/{plugin_id}")
            logger.info("plugin installed: %s/%s → %s", kind, plugin_id, plugin_dir)

        return PluginUnpackResult(
            installed=tuple(installed),
            skipped=tuple(skipped),
            total_bytes=total,
        )
