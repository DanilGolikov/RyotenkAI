"""Instance-level params/thresholds validation against a plugin manifest.

Scope: a YAML config can wire up a plugin with arbitrary
``params: {...}`` / ``thresholds: {...}`` blocks. The Configure modal
forces the user through ``params_schema`` so the values match — but
hand-edited YAML files have no such guard. Without this validator a
type/enum/range violation only surfaces when the plugin's
``__init__`` happens to assert (or worse, mid-pipeline at first use).

This module runs the same JSON Schema the UI uses
(``params_to_json_schema(manifest.params_schema)``) against the
config-supplied dict and returns *all* violations as structured rows.
The preflight gate (:mod:`src.community.preflight`) bundles these into
the same response shape it already uses for missing envs.

Why ``jsonschema`` Draft7Validator and not ``manifest.model_validate``?
Pydantic's ParamFieldSchema is the *source of truth* for the schema,
but it doesn't validate user-supplied dicts — only the schema itself.
JSON Schema validation gives us cheap, complete coverage (all errors,
not just the first) for free, using the dependency that's already
required by FastAPI for OpenAPI generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.community.manifest import PluginManifest


@dataclass(frozen=True, slots=True)
class InstanceValidationError:
    """One per-field violation of a plugin's params/thresholds schema.

    ``location`` is a dotted path like ``params.timeout_seconds`` or
    ``thresholds.min_score`` so the UI can highlight the exact field
    in the Configure modal. ``message`` is a human-readable summary
    cribbed from the underlying ``jsonschema.ValidationError``.
    """

    plugin_kind: str
    plugin_name: str
    plugin_instance_id: str
    location: str
    message: str


def validate_instance(
    manifest: PluginManifest,
    *,
    plugin_kind: str,
    plugin_name: str,
    plugin_instance_id: str,
    params: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> list[InstanceValidationError]:
    """Return every ``params`` / ``thresholds`` violation against ``manifest``.

    Empty list ⇒ the YAML config matches the manifest schema and the
    plugin's ``__init__`` won't crash on this input.

    Validates ``params`` against ``manifest.params_schema`` and
    ``thresholds`` against ``manifest.thresholds_schema``. Plugins
    without a schema (``params_schema = {}`` etc.) accept any dict —
    matches the runtime behaviour where the loader skips emitting an
    empty schema.

    Both blocks are validated independently so the caller sees all
    issues in one pass; ``Draft7Validator.iter_errors`` (rather than
    ``validate``) gives complete coverage.
    """
    from jsonschema import Draft7Validator

    from src.community.manifest import params_to_json_schema

    errors: list[InstanceValidationError] = []
    for label, value, schema_dict in (
        ("params", params or {}, manifest.params_schema),
        ("thresholds", thresholds or {}, manifest.thresholds_schema),
    ):
        if not schema_dict:
            continue
        schema = params_to_json_schema(schema_dict)
        validator = Draft7Validator(schema)
        for err in validator.iter_errors(value):
            # err.absolute_path is a deque of segments. Prefix with the
            # block label so the UI knows whether it's a params or
            # thresholds violation. Empty path = error on the dict
            # itself (e.g. additionalProperties).
            tail = ".".join(str(seg) for seg in err.absolute_path)
            location = f"{label}.{tail}" if tail else label
            errors.append(InstanceValidationError(
                plugin_kind=plugin_kind,
                plugin_name=plugin_name,
                plugin_instance_id=plugin_instance_id,
                location=location,
                message=err.message,
            ))
    return errors


__all__ = ["InstanceValidationError", "validate_instance"]
