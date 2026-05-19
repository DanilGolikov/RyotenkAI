"""Concrete :class:`IPromptRegistry` implementation over ``mlflow.genai.load_prompt``.

After the wide ``IMLflowManager`` retirement, :class:`SystemPromptLoader`
depends only on the narrow :class:`IPromptRegistry` Protocol. This module
provides the production implementation used by the control orchestrator
and inference providers; the registry delegates to
``mlflow.genai.load_prompt`` with a per-call timeout so MLflow outages
never block the inference hot path indefinitely.

Supports the three URI flavours accepted by ``mlflow.genai.load_prompt``:

* name only — e.g. ``"my_prompt"``
* version pin — e.g. ``"prompts:/my_prompt/3"``
* alias pin — e.g. ``"prompts:/my_prompt@champion"``

Lookup failures (network, auth, missing prompt) return ``None`` so the
caller (:class:`SystemPromptLoader`) decides the failure mode via its
``on_mlflow_failure`` kwarg.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from ryotenkai_shared.infrastructure.mlflow.protocols import PromptArtifact

logger = get_logger(__name__)


class MlflowPromptRegistry:
    """:class:`IPromptRegistry` implementation that delegates to ``mlflow.genai.load_prompt``.

    A single instance is intended to live for the duration of a pipeline /
    evaluation run; it is stateless apart from the tracking URI.
    Construction is cheap (no network round-trip until the first
    :meth:`load` call).

    :param tracking_uri: MLflow tracking URI. Must be non-empty; an empty
        string raises ``ValueError`` because ``mlflow.genai.load_prompt``
        would silently fall back to the OSS default (``./mlruns``) and
        load the wrong prompt.
    """

    def __init__(self, tracking_uri: str) -> None:
        if not tracking_uri:
            raise ValueError("tracking_uri must be non-empty")
        self._tracking_uri = tracking_uri

    def load(self, name_or_uri: str, timeout_s: float) -> PromptArtifact | None:
        """Load a prompt by name or ``prompts:/`` URI.

        Runs the MLflow call on a background thread so the caller can
        enforce a hard timeout (``timeout_s``). Any exception during
        lookup is logged and translated into ``None`` — the caller's
        failure-mode toggle takes over from there.

        :param name_or_uri: Prompt name, ``prompts:/name/version``, or
            ``prompts:/name@alias``.
        :param timeout_s: Maximum wall-clock seconds to wait for the
            registry call.
        :returns: A :class:`PromptArtifact` on success, ``None`` on any
            failure (including timeout).
        """
        import concurrent.futures as _f

        def _do_load() -> PromptArtifact | None:
            import mlflow
            import mlflow.genai

            mlflow.set_tracking_uri(self._tracking_uri)
            return mlflow.genai.load_prompt(name_or_uri)

        try:
            with _f.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_do_load)
                return future.result(timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001 -- surface as soft failure
            logger.warning(
                f"[PROMPT_REGISTRY] load({name_or_uri!r}) failed: {exc}"
            )
            return None


__all__ = ["MlflowPromptRegistry"]
