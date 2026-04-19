"""
BasePlugin — lightweight mixin for all plugin systems in this project.

Provides a shared set of ClassVar metadata fields that every plugin type
(validation, evaluation, reward, report) should carry, plus an optional
``MANIFEST`` dict surfacing human-readable metadata (description, category,
suggested params/thresholds) for UI consumption.
"""

from __future__ import annotations

from typing import Any, ClassVar


class BasePlugin:
    """
    Mixin that enforces a common metadata contract across all plugin ABCs.

    All plugin systems in this project share three invariants:
      - ``name``     — unique string key used by registries for lookup.
      - ``priority`` — execution order hint (lower = runs earlier).
      - ``version``  — semver string, useful for compatibility checks.

    Optional metadata for front-ends / docs:
      - ``MANIFEST`` — dict with keys:
            description        (str)
            category           (str; free-form grouping hint)
            stability          ("stable" | "beta" | "experimental")
            params_schema      ({field_name: {"type": ..., "min": ..., "max": ..., "default": ...}})
            thresholds_schema  (same shape)
            suggested_params   ({field_name: value})
            suggested_thresholds ({field_name: value})
        Every field is optional. ``get_manifest()`` returns a normalised
        dict merging MANIFEST with ClassVars and filling missing keys.

    This class intentionally carries NO abstract methods and NO ``__init__``
    so it can be inserted into any ABC/Protocol hierarchy without MRO conflicts.
    """

    name: ClassVar[str] = ""
    priority: ClassVar[int] = 50
    version: ClassVar[str] = "1.0.0"

    MANIFEST: ClassVar[dict[str, Any] | None] = None

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        """
        Return a normalised manifest dict for UI consumption.

        Always includes ``id``, ``name``, ``version``, ``priority``. Additional
        fields (description, category, stability, params_schema, ...) are
        pulled from ``cls.MANIFEST`` when present.
        """
        base: dict[str, Any] = {
            "id": cls.name,
            "name": cls.name,
            "version": cls.version,
            "priority": cls.priority,
            "description": "",
            "category": "",
            "stability": "stable",
            "params_schema": {},
            "thresholds_schema": {},
            "suggested_params": {},
            "suggested_thresholds": {},
        }
        manifest = cls.MANIFEST
        if isinstance(manifest, dict):
            for key, value in manifest.items():
                base[key] = value
        return base


__all__ = ["BasePlugin"]
