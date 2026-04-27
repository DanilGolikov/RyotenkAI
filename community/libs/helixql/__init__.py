"""Shared HelixQL helpers reused by community plugins.

This package is registered in :mod:`sys.modules` as
``community_libs.helixql`` by :func:`src.community.libs.preload_community_libs`
the first time the catalog loads. Plugins import from one of three
submodules:

- :mod:`community_libs.helixql.compiler` — :class:`HelixCompiler`,
  :class:`CompileResult`, :func:`get_compiler` (cached factory).
- :mod:`community_libs.helixql.extract` — :func:`extract_query_text`,
  :func:`extract_schema_block`, :func:`extract_schema_and_query`.
- :mod:`community_libs.helixql.semantics` — :func:`semantic_match_details`,
  :func:`hard_eval_errors`, :func:`normalize_query_text`.

The package itself re-exports the most-used names so callers that want
the top-level surface can keep imports short::

    from community_libs.helixql import semantic_match_details, get_compiler

Re-exports are wired through :pep:`562` ``__getattr__`` so that simply
loading this ``__init__.py`` does **not** eagerly import the
submodules. That matters because pytest's collection phase imports
this module before the catalog has had a chance to preload the
``community_libs`` namespace; eager re-exports would break test
discovery. The lazy form costs nothing at runtime — first access
caches the resolved object on the module so subsequent reads hit
the standard attribute fast path.

There is intentionally **no** state held at the package level — each
submodule is independently importable.
"""

from __future__ import annotations

from typing import Any

# Map of public name → ``(submodule, attr)``. Lazy resolution lets us
# expose a flat surface without the eager-import gotchas described
# above. Keep this list aligned with the per-submodule ``__all__``.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CompileResult": ("community_libs.helixql.compiler", "CompileResult"),
    "HelixCompiler": ("community_libs.helixql.compiler", "HelixCompiler"),
    "get_compiler": ("community_libs.helixql.compiler", "get_compiler"),
    "extract_query_text": ("community_libs.helixql.extract", "extract_query_text"),
    "extract_schema_and_query": (
        "community_libs.helixql.extract",
        "extract_schema_and_query",
    ),
    "extract_schema_block": ("community_libs.helixql.extract", "extract_schema_block"),
    "hard_eval_errors": ("community_libs.helixql.semantics", "hard_eval_errors"),
    "normalize_query_text": (
        "community_libs.helixql.semantics",
        "normalize_query_text",
    ),
    "semantic_match_details": (
        "community_libs.helixql.semantics",
        "semantic_match_details",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(
            f"module 'community_libs.helixql' has no attribute {name!r}; "
            f"available: {sorted(_LAZY_EXPORTS)!r}"
        )
    module_name, attr = target
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    # Cache so subsequent accesses skip __getattr__.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = sorted(_LAZY_EXPORTS)
