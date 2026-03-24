from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True, slots=True)
class ModuleImportFailure:
    module_name: str
    error_type: str
    error_message: str


@dataclass(frozen=True, slots=True)
class DiscoveryDiagnostics:
    package_name: str
    candidate_modules: tuple[str, ...]
    imported_modules: tuple[str, ...]
    failed_modules: tuple[ModuleImportFailure, ...]

    @property
    def failed_module_names(self) -> tuple[str, ...]:
        return tuple(item.module_name for item in self.failed_modules)


def discover_modules(
    package_name: str,
    *,
    recursive: bool = True,
    exclude_stems: Iterable[str] = (),
    exclude_dirs: Iterable[str] = ("__pycache__",),
    exclude_prefixes: Iterable[str] = ("test_",),
) -> list[str]:
    package = importlib.import_module(package_name)
    package_file = getattr(package, "__file__", None)
    if not package_file:
        raise ValueError(f"Cannot discover modules for package {package_name!r}: package has no __file__")

    package_root = Path(package_file).resolve().parent
    iterator = package_root.rglob("*.py") if recursive else package_root.glob("*.py")
    excluded_stems = set(exclude_stems)
    excluded_dirs = set(exclude_dirs)
    excluded_prefixes = tuple(exclude_prefixes)

    modules: list[str] = []
    for path in iterator:
        if any(part in excluded_dirs for part in path.parts):
            continue
        if path.name == "__init__.py":
            continue
        if path.stem in excluded_stems:
            continue
        if path.stem.startswith(excluded_prefixes):
            continue
        if any(part.startswith(excluded_prefixes) for part in path.relative_to(package_root).parts):
            continue

        relative = path.relative_to(package_root).with_suffix("")
        module_name = ".".join((package_name, *relative.parts))
        modules.append(module_name)

    return sorted(set(modules))


def import_modules(
    package_name: str,
    module_names: Iterable[str],
    *,
    logger: logging.Logger | None = None,
    reload_modules: bool = False,
) -> DiscoveryDiagnostics:
    active_logger = logger or logging.getLogger("ryotenkai")
    candidates = tuple(sorted(set(module_names)))
    imported: list[str] = []
    failures: list[ModuleImportFailure] = []

    for module_name in candidates:
        try:
            if reload_modules and module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
                importlib.import_module(module_name)
            imported.append(module_name)
        except Exception as exc:
            failures.append(
                ModuleImportFailure(
                    module_name=module_name,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            active_logger.warning(
                "[PLUGIN_DISCOVERY] package=%s module=%s failed error_type=%s error=%s",
                package_name,
                module_name,
                type(exc).__name__,
                exc,
            )

    diagnostics = DiscoveryDiagnostics(
        package_name=package_name,
        candidate_modules=candidates,
        imported_modules=tuple(imported),
        failed_modules=tuple(failures),
    )

    active_logger.info(
        "[PLUGIN_DISCOVERY] package=%s candidates=%d imported=%d failed=%d",
        package_name,
        len(diagnostics.candidate_modules),
        len(diagnostics.imported_modules),
        len(diagnostics.failed_modules),
    )
    return diagnostics


def discover_and_import_modules(
    package_name: str,
    *,
    recursive: bool = True,
    exclude_stems: Iterable[str] = (),
    exclude_dirs: Iterable[str] = ("__pycache__",),
    exclude_prefixes: Iterable[str] = ("test_",),
    logger: logging.Logger | None = None,
    reload_modules: bool = False,
) -> DiscoveryDiagnostics:
    module_names = discover_modules(
        package_name,
        recursive=recursive,
        exclude_stems=exclude_stems,
        exclude_dirs=exclude_dirs,
        exclude_prefixes=exclude_prefixes,
    )
    return import_modules(package_name, module_names, logger=logger, reload_modules=reload_modules)


__all__ = [
    "DiscoveryDiagnostics",
    "ModuleImportFailure",
    "discover_and_import_modules",
    "discover_modules",
    "import_modules",
]
