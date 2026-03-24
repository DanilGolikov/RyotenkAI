from __future__ import annotations

from src.utils.logger import get_logger
from src.utils.plugin_discovery import DiscoveryDiagnostics, discover_and_import_modules

logger = get_logger(__name__)

_discovery_cache: DiscoveryDiagnostics | None = None


def ensure_evaluation_plugins_discovered(*, force: bool = False) -> DiscoveryDiagnostics:
    global _discovery_cache

    if _discovery_cache is not None and not force:
        return _discovery_cache

    if force:
        from src.evaluation.plugins.registry import EvaluatorPluginRegistry

        EvaluatorPluginRegistry.clear()

    _discovery_cache = discover_and_import_modules(
        "src.evaluation.plugins",
        exclude_stems={"base", "registry", "utils", "secrets", "interface", "interfaces", "discovery"},
        logger=logger,
        reload_modules=force,
    )
    return _discovery_cache


__all__ = ["ensure_evaluation_plugins_discovered"]
