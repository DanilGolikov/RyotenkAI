"""Phase 14.E (V5) — bootstrap layer for MLflow exception classifier.

Pre-14.E :mod:`src.training.mlflow.resilient_transport` carried
module-level optional imports of ``requests`` and ``urllib3`` to
populate its retry-eligible exception type set. This module
relocates that library knowledge to an explicit "bootstrap" layer
so the transport module is testable / importable without those
libraries (slim CI venvs).

Production trainer init wires
:func:`make_default_classifier_for_mlflow` when constructing
:class:`~src.training.mlflow.resilient_transport.ResilientMLflowTransport`.
The transport's default (``_DefaultClassifier`` with the legacy
type set) preserves backwards compatibility for direct callers.
"""

from __future__ import annotations

from src.training.mlflow.resilient_transport import (
    ExceptionClassifier,
    _DefaultClassifier,
    _optional_exception_types,
)

__all__ = ["make_default_classifier_for_mlflow"]


def make_default_classifier_for_mlflow() -> ExceptionClassifier:
    """Build a classifier that recognises MLflow's transport stack.

    Phase 14.E (V5): explicit place where library knowledge lives.
    The function reuses :func:`_optional_exception_types` (which
    optionally imports ``requests`` and ``urllib3``) so the
    behaviour is identical to pre-14.E. Future swaps (e.g. MLflow
    moves to ``httpx``) update this single function — the transport
    module is unaffected.

    Returns:
        :class:`_DefaultClassifier` instance bound to the full
        type set including any optionally-importable libraries.
    """
    return _DefaultClassifier(types=_optional_exception_types())
