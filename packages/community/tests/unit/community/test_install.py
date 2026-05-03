"""Unit tests for ``src.community.install``.

The tests avoid real network / git by:

- Using ``install_local`` for the bulk of the install machinery (folder
  layout, manifest validation, kind/id derivation, destination
  protection).
- Mocking ``shutil.which`` and ``subprocess.run`` for the git path so
  the tests run on hosts without git installed.
- Using ``zipfile`` to build real ``.zip`` fixtures inside ``tmp_path``
  for the archive path.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

from src.community.constants import (
    ALL_PLUGIN_KINDS,
    MANIFEST_FILENAME,
    PRESET_DIR_NAME,
)
from src.community.install import (
    ALLOWED_KINDS,
    InstallError,
    InstallResult,
    install_git,
    install_local,
)
from src.community.manifest import LATEST_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _plugin_manifest(plugin_id: str = "min_samples", kind: str = "validation") -> str:
    return (
        f"schema_version = {LATEST_SCHEMA_VERSION}\n"
        "[plugin]\n"
        f'id = "{plugin_id}"\n'
        f'kind = "{kind}"\n'
        'name = "Test Plugin"\n'
        'version = "1.0.0"\n'
        'description = "test"\n'
        + ('supported_strategies = ["grpo"]\n' if kind == "reward" else "")
        + "\n"
        "[plugin.entry_point]\n"
        'module = "plugin"\n'
        'class = "Cls"\n'
    )


def _preset_manifest(preset_id: str = "demo-preset") -> str:
    return (
        "[preset]\n"
        f'id = "{preset_id}"\n'
        'name = "Demo"\n'
        'description = "demo"\n'
        'version = "1.0.0"\n'
        "\n"
        "[preset.entry_point]\n"
        'file = "preset.yaml"\n'
    )


def _make_plugin_folder(
    parent: Path, plugin_id: str = "min_samples", kind: str = "validation",
) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / plugin_id
    folder.mkdir()
    (folder / MANIFEST_FILENAME).write_text(
        _plugin_manifest(plugin_id, kind), encoding="utf-8"
    )
    (folder / "plugin.py").write_text("class Cls: pass\n", encoding="utf-8")
    (folder / "__pycache__").mkdir()
    (folder / "__pycache__" / "junk.pyc").write_bytes(b"\x00")
    return folder


def _make_preset_folder(parent: Path, preset_id: str = "demo-preset") -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / preset_id
    folder.mkdir()
    (folder / MANIFEST_FILENAME).write_text(
        _preset_manifest(preset_id), encoding="utf-8"
    )
    (folder / "preset.yaml").write_text("model:\n  name: x\n", encoding="utf-8")
    return folder


@pytest.fixture()
def community_root(tmp_path: Path) -> Path:
    """Sandboxed community/ root so tests never touch the real one."""
    return tmp_path / "community"


# ---------------------------------------------------------------------------
# Local install — folder
# ---------------------------------------------------------------------------


def test_install_local_folder_validation_plugin(
    tmp_path: Path, community_root: Path,
) -> None:
    src = _make_plugin_folder(tmp_path / "src", "min_samples", "validation")
    result = install_local(src, community_root=community_root)
    assert isinstance(result, InstallResult)
    assert result.plugin_id == "min_samples"
    assert result.kind == "validation"
    assert result.source_kind == "folder"
    assert result.target_path == community_root / "validation" / "min_samples"
    assert result.overwritten is False
    assert (result.target_path / MANIFEST_FILENAME).exists()
    assert (result.target_path / "plugin.py").exists()


def test_install_local_strips_caches(tmp_path: Path, community_root: Path) -> None:
    src = _make_plugin_folder(tmp_path / "src")
    result = install_local(src, community_root=community_root)
    # __pycache__ from the source must NOT propagate
    assert not (result.target_path / "__pycache__").exists()


def test_install_local_preset(tmp_path: Path, community_root: Path) -> None:
    src = _make_preset_folder(tmp_path / "src")
    result = install_local(src, community_root=community_root)
    assert result.kind == "preset"
    # Presets land under PRESET_DIR_NAME, not under the kind name.
    assert result.target_path == community_root / PRESET_DIR_NAME / "demo-preset"


def test_install_local_each_plugin_kind(
    tmp_path: Path, community_root: Path,
) -> None:
    """Cover every kind so the kind→subdir mapping never silently breaks."""
    for kind in ALL_PLUGIN_KINDS:
        src = _make_plugin_folder(tmp_path / kind, plugin_id=f"p_{kind}", kind=kind)
        result = install_local(src, community_root=community_root)
        assert result.kind == kind
        assert result.target_path.parent.name == kind


# ---------------------------------------------------------------------------
# Error paths — folder install
# ---------------------------------------------------------------------------


def test_install_local_missing_source_raises(
    tmp_path: Path, community_root: Path,
) -> None:
    with pytest.raises(InstallError) as exc:
        install_local(tmp_path / "nope", community_root=community_root)
    assert exc.value.code == "source_not_found"


def test_install_local_invalid_source_type(
    tmp_path: Path, community_root: Path,
) -> None:
    bad = tmp_path / "not-a-zip-or-dir.txt"
    bad.write_text("nope")
    with pytest.raises(InstallError) as exc:
        install_local(bad, community_root=community_root)
    assert exc.value.code == "source_invalid"


def test_install_local_missing_manifest(
    tmp_path: Path, community_root: Path,
) -> None:
    bad = tmp_path / "src"
    bad.mkdir()
    with pytest.raises(InstallError) as exc:
        install_local(bad, community_root=community_root)
    assert exc.value.code == "manifest_invalid"


def test_install_local_invalid_manifest(
    tmp_path: Path, community_root: Path,
) -> None:
    bad = tmp_path / "src"
    bad.mkdir()
    (bad / MANIFEST_FILENAME).write_text("[plugin]\nid = 1\n", encoding="utf-8")
    with pytest.raises(InstallError) as exc:
        install_local(bad, community_root=community_root)
    assert exc.value.code == "manifest_invalid"


def test_install_local_kind_mismatch(
    tmp_path: Path, community_root: Path,
) -> None:
    src = _make_plugin_folder(tmp_path / "src", "p", "validation")
    with pytest.raises(InstallError) as exc:
        install_local(src, expected_kind="evaluation", community_root=community_root)
    assert exc.value.code == "kind_mismatch"


def test_install_local_unknown_kind_value(
    tmp_path: Path, community_root: Path,
) -> None:
    src = _make_plugin_folder(tmp_path / "src")
    with pytest.raises(InstallError) as exc:
        install_local(src, expected_kind="nonsense", community_root=community_root)
    assert exc.value.code == "kind_unknown"


def test_install_local_destination_exists_without_force(
    tmp_path: Path, community_root: Path,
) -> None:
    src = _make_plugin_folder(tmp_path / "src", "p", "validation")
    install_local(src, community_root=community_root)
    src2 = _make_plugin_folder(tmp_path / "src2", "p", "validation")
    with pytest.raises(InstallError) as exc:
        install_local(src2, community_root=community_root)
    assert exc.value.code == "destination_exists"


def test_install_local_destination_exists_with_force_overwrites(
    tmp_path: Path, community_root: Path,
) -> None:
    src = _make_plugin_folder(tmp_path / "src", "p", "validation")
    first = install_local(src, community_root=community_root)
    # User edits the installed copy — overwriting must replace it.
    (first.target_path / "user_marker.txt").write_text("manual edit")

    src2 = _make_plugin_folder(tmp_path / "src2", "p", "validation")
    (src2 / "fresh_marker.txt").write_text("from source")
    result = install_local(src2, community_root=community_root, force=True)
    assert result.overwritten is True
    assert not (result.target_path / "user_marker.txt").exists()
    assert (result.target_path / "fresh_marker.txt").exists()


# ---------------------------------------------------------------------------
# Archive install
# ---------------------------------------------------------------------------


def _zip_folder(folder: Path, archive_path: Path) -> Path:
    """Zip ``folder`` into ``archive_path`` (no top-level wrapper)."""
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in folder.rglob("*"):
            if item.is_dir():
                continue
            zf.write(item, arcname=str(item.relative_to(folder)))
    return archive_path


def test_install_local_zip_archive(tmp_path: Path, community_root: Path) -> None:
    folder = _make_plugin_folder(tmp_path / "src", "zipped", "validation")
    archive = _zip_folder(folder, tmp_path / "zipped.zip")
    result = install_local(archive, community_root=community_root)
    assert result.source_kind == "archive"
    assert result.plugin_id == "zipped"
    assert (result.target_path / "plugin.py").exists()


def test_install_local_zip_with_top_level_wrapper(
    tmp_path: Path, community_root: Path,
) -> None:
    """``resolve_extraction_root`` must descend into a single wrapper folder."""
    inner = _make_plugin_folder(tmp_path / "wrapper", "wrapped", "validation")
    archive = tmp_path / "wrapped.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in inner.rglob("*"):
            if item.is_dir():
                continue
            # Prefix with the wrapper directory name to simulate a zip
            # that holds the plugin inside a single top-level folder.
            arcname = "wrapper/" + str(item.relative_to(inner))
            zf.write(item, arcname=arcname)
    result = install_local(archive, community_root=community_root)
    assert result.plugin_id == "wrapped"


def test_install_local_corrupt_zip(tmp_path: Path, community_root: Path) -> None:
    bad = tmp_path / "broken.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(InstallError) as exc:
        install_local(bad, community_root=community_root)
    assert exc.value.code == "source_invalid"


# ---------------------------------------------------------------------------
# Git install — subprocess mocked
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> dict[str, Any]:
    """Mock ``shutil.which("git")`` + ``subprocess.run`` for git calls.

    The fake ``git clone`` materialises a valid plugin folder at the
    destination path so the rest of the install path runs unchanged.
    Returns a dict the test can mutate (``plugin_id``, ``ref_in_clone``,
    ``clone_should_fail``, ``checkout_should_fail``).
    """
    state: dict[str, Any] = {
        "plugin_id": "from_git",
        "kind": "validation",
        "ref_in_clone": "abc1234abc1234abc1234abc1234abc1234abc1",
        "clone_should_fail": False,
        "checkout_should_fail": False,
    }
    monkeypatch.setattr(shutil, "which", lambda exe: "/usr/bin/git" if exe == "git" else None)

    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        if not (isinstance(cmd, list) and cmd and cmd[0] == "git"):
            return real_run(cmd, *args, **kwargs)
        if cmd[1] == "clone":
            if state["clone_should_fail"]:
                return _completed(returncode=1, stderr="boom")
            target = Path(cmd[-1])
            target.mkdir(parents=True, exist_ok=True)
            (target / MANIFEST_FILENAME).write_text(
                _plugin_manifest(state["plugin_id"], state["kind"]),
                encoding="utf-8",
            )
            (target / "plugin.py").write_text("class Cls: pass\n", encoding="utf-8")
            return _completed(returncode=0)
        if cmd[1] == "-C":
            sub = cmd[3]
            if sub == "checkout":
                return _completed(
                    returncode=1 if state["checkout_should_fail"] else 0,
                    stderr="checkout boom" if state["checkout_should_fail"] else "",
                )
            if sub == "rev-parse":
                return _completed(returncode=0, stdout=state["ref_in_clone"] + "\n")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return state


def _completed(*, returncode: int = 0, stdout: str = "", stderr: str = "") -> Any:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_install_git_with_pinned_sha(
    fake_git: dict[str, Any], community_root: Path,
) -> None:
    sha = "a" * 40
    fake_git["ref_in_clone"] = sha
    result = install_git(
        "https://example.com/repo.git", ref=sha, community_root=community_root,
    )
    assert result.source_kind == "git"
    assert result.plugin_id == "from_git"


def test_install_git_with_branch_requires_allow_untrusted(
    fake_git: dict[str, Any], community_root: Path,
) -> None:
    with pytest.raises(InstallError) as exc:
        install_git(
            "https://example.com/repo.git", ref="main",
            community_root=community_root,
        )
    assert exc.value.code == "untrusted_git_ref"


def test_install_git_with_branch_and_allow_untrusted(
    fake_git: dict[str, Any], community_root: Path,
) -> None:
    result = install_git(
        "https://example.com/repo.git", ref="main",
        allow_untrusted=True, community_root=community_root,
    )
    assert result.kind == "validation"


def test_install_git_clone_failure(
    fake_git: dict[str, Any], community_root: Path,
) -> None:
    fake_git["clone_should_fail"] = True
    with pytest.raises(InstallError) as exc:
        install_git(
            "https://example.com/repo.git", ref="a" * 40,
            community_root=community_root,
        )
    assert exc.value.code == "git_clone_failed"


def test_install_git_checkout_failure(
    fake_git: dict[str, Any], community_root: Path,
) -> None:
    fake_git["checkout_should_fail"] = True
    with pytest.raises(InstallError) as exc:
        install_git(
            "https://example.com/repo.git", ref="a" * 40,
            community_root=community_root,
        )
    assert exc.value.code == "git_checkout_failed"


def test_install_git_ref_drift_aborts(
    fake_git: dict[str, Any], community_root: Path,
) -> None:
    """If rev-parse returns a different sha than requested, we abort."""
    requested = "a" * 40
    fake_git["ref_in_clone"] = "b" * 40
    with pytest.raises(InstallError) as exc:
        install_git(
            "https://example.com/repo.git", ref=requested,
            community_root=community_root,
        )
    assert exc.value.code == "git_ref_drift"


def test_install_git_executable_missing(
    monkeypatch: pytest.MonkeyPatch, community_root: Path,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda exe: None)
    with pytest.raises(InstallError) as exc:
        install_git(
            "https://example.com/repo.git", ref="a" * 40,
            community_root=community_root,
        )
    assert exc.value.code == "git_unavailable"


# ---------------------------------------------------------------------------
# ALLOWED_KINDS surface
# ---------------------------------------------------------------------------


def test_allowed_kinds_includes_preset_and_all_plugin_kinds() -> None:
    assert "preset" in ALLOWED_KINDS
    for kind in ALL_PLUGIN_KINDS:
        assert kind in ALLOWED_KINDS
