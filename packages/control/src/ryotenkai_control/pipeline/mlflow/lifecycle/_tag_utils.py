"""MLflow tag-value sanitization helpers.

Migrated from the legacy
:mod:`ryotenkai_control.pipeline.mlflow_attempt.manager` module
(``_stringify_tag_value`` private helper) as part of Phase M7.2.

MLflow tag values are constrained to a fixed length on the server side
(see https://mlflow.org/docs/latest/python_api/mlflow.entities.html).
The limit is 5000 chars in current versions; we truncate slightly
earlier to leave room for the ``…`` continuation marker. Anything that
would overflow is replaced with a head-only excerpt rather than being
silently rejected by the backend mid-run.
"""

from __future__ import annotations

_MLFLOW_TAG_VALUE_MAX_CHARS = 4990
"""Maximum length of an MLflow tag value before head-only truncation."""


def stringify_tag_value(value: object) -> str:
    """Coerce an arbitrary value to an MLflow-safe tag string.

    :param value: Any Python value. ``None`` becomes ``"None"`` (visible
        in the MLflow UI; better than a silent drop).
    :returns: The ``str()`` coercion truncated to
        :data:`_MLFLOW_TAG_VALUE_MAX_CHARS` characters with a trailing
        ``"…"`` continuation marker if truncation was required.
    """
    s = str(value)
    if len(s) > _MLFLOW_TAG_VALUE_MAX_CHARS:
        return s[:_MLFLOW_TAG_VALUE_MAX_CHARS] + "…"
    return s


__all__ = ["_MLFLOW_TAG_VALUE_MAX_CHARS", "stringify_tag_value"]
