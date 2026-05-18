"""Caching :class:`SystemPromptLoader` with explicit failure modes.

Resolves the system prompt from one of two mutually-exclusive sources
on :class:`InferenceLLMConfig`:

* ``system_prompt_mlflow_name`` — MLflow Prompt Registry (cached).
* ``system_prompt_path``        — local file (not cached; mtime check
  overhead is not worth the cost in the hot loop).

Improvements over the legacy static loader at
``ryotenkai_shared.infrastructure.mlflow.system_prompt``:

1. **Bounded in-memory cache** keyed by ``(name_or_uri,)`` with TTL
   (default 300s) and FIFO eviction once ``cache_maxsize`` is reached.
   The legacy loader hit MLflow on every ``load()`` call.

2. **Explicit failure mode toggle** via the ``on_mlflow_failure`` kwarg
   on :meth:`SystemPromptLoader.load`: ``"fail"`` raises, ``"warn"``
   logs and returns ``None`` (legacy behaviour), and
   ``"fallback_to_file"`` falls back to ``system_prompt_path`` when
   set.

3. **Narrow** :class:`IPromptRegistry` **dependency** instead of the
   concrete ``IMLflowGateway`` — easier to fake and to swap for an
   alternative registry implementation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ryotenkai_shared.infrastructure.mlflow.protocols import IPromptRegistry
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from ryotenkai_shared.config.inference.common import InferenceLLMConfig

logger = get_logger(__name__)

# Per-call MLflow timeout. Matches the legacy loader's contract that
# MLflow lookups never block the orchestrator hot path indefinitely.
_MLFLOW_LOOKUP_TIMEOUT_S: float = 5.0


OnMlflowFailure = Literal["fail", "warn", "fallback_to_file"]


@dataclass
class SystemPromptResult:
    """Resolved system prompt with its origin metadata.

    Attributes:
        text:   Raw prompt string ready to pass to the LLM.
        source: Audit metadata describing where the prompt came from.
                For a file source:   ``{"type": "file",   "path": ...}``
                For an MLflow source: ``{"type": "mlflow", "name": ..., "version": ...}``
    """

    text: str
    source: dict[str, str] = field(default_factory=dict)


class SystemPromptLoader:
    """Loads a system prompt with bounded caching and explicit failure modes.

    A single loader instance is intended to live for the duration of a
    pipeline / evaluation run; the in-memory cache is per-instance so
    invalidation is scoped automatically when the instance is dropped.
    """

    def __init__(
        self,
        registry: IPromptRegistry | None = None,
        *,
        cache_ttl_s: float = 300.0,
        cache_maxsize: int = 64,
    ) -> None:
        """Create a new loader.

        Args:
            registry:       Optional :class:`IPromptRegistry`. Required
                            only when an MLflow source is actually
                            requested at :meth:`load` time — file-only
                            callers may pass ``None``.
            cache_ttl_s:    Cache entry TTL in seconds. Entries older
                            than this are re-fetched on next lookup.
            cache_maxsize:  Maximum number of cache entries. When
                            exceeded, the oldest entry is evicted
                            (FIFO, using Python's insertion-ordered
                            dict).
        """
        if cache_ttl_s < 0:
            raise ValueError(f"cache_ttl_s must be >= 0, got {cache_ttl_s!r}")
        if cache_maxsize < 1:
            raise ValueError(f"cache_maxsize must be >= 1, got {cache_maxsize!r}")

        self._registry: IPromptRegistry | None = registry
        self._cache_ttl_s: float = float(cache_ttl_s)
        self._cache_maxsize: int = int(cache_maxsize)

        # Cache: key = name_or_uri, value = (result, cached_at monotonic seconds).
        # Insertion order is preserved by Python 3.7+ dict, so FIFO
        # eviction reduces to "drop oldest key" on overflow.
        self._cache: dict[str, tuple[SystemPromptResult, float]] = {}
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        llm_cfg: InferenceLLMConfig,
        *,
        on_mlflow_failure: OnMlflowFailure = "warn",
    ) -> SystemPromptResult | None:
        """Resolve the system prompt from the configured source.

        Selection rules:
            * ``system_prompt_mlflow_name`` set  → MLflow path.
            * ``system_prompt_path`` set (and the above is not)  → file path.
            * Neither set  → ``None``.

        Args:
            llm_cfg: :class:`InferenceLLMConfig` carrying source config.
            on_mlflow_failure: How to treat MLflow connectivity/lookup
                failures.

                * ``"fail"`` — re-raise as :class:`RuntimeError`.
                * ``"warn"`` — log a warning and return ``None``
                  (legacy default behaviour).
                * ``"fallback_to_file"`` — fall back to
                  ``system_prompt_path`` when set; otherwise warn +
                  ``None``.

        Returns:
            :class:`SystemPromptResult` with prompt text and source
            metadata, or ``None`` when neither source is configured (or
            on a tolerated failure).
        """
        mlflow_name = llm_cfg.system_prompt_mlflow_name
        file_path = llm_cfg.system_prompt_path

        if mlflow_name:
            try:
                result = self._load_from_mlflow(mlflow_name)
            except _PromptRegistryFailure as exc:
                return self._handle_mlflow_failure(
                    exc=exc,
                    on_failure=on_mlflow_failure,
                    file_path_fallback=file_path,
                    name=mlflow_name,
                )

            if result is None:
                # registry.load() returned None — treat as a soft failure.
                return self._handle_mlflow_failure(
                    exc=None,
                    on_failure=on_mlflow_failure,
                    file_path_fallback=file_path,
                    name=mlflow_name,
                )
            return result

        if file_path:
            return self._load_from_file(file_path)

        return None

    def invalidate(self, name_or_uri: str | None = None) -> None:
        """Drop cache entries.

        Args:
            name_or_uri: Specific key to evict. When ``None``, the
                entire cache is cleared.
        """
        with self._cache_lock:
            if name_or_uri is None:
                self._cache.clear()
            else:
                self._cache.pop(name_or_uri, None)

    # ------------------------------------------------------------------
    # MLflow path
    # ------------------------------------------------------------------

    def _load_from_mlflow(self, name_or_uri: str) -> SystemPromptResult | None:
        """Load from MLflow, with cache + bounded eviction.

        Raises:
            :class:`_PromptRegistryFailure` when the registry call raises
            or no registry was injected at construction time.
        """
        cached = self._cache_lookup(name_or_uri)
        if cached is not None:
            logger.debug(
                f"[SYSTEM_PROMPT] cache hit for mlflow prompt {name_or_uri!r} "
                f"({len(cached.text)} chars)"
            )
            return cached

        if self._registry is None:
            raise _PromptRegistryFailure(
                "system_prompt_mlflow_name is configured but no IPromptRegistry "
                "was provided to SystemPromptLoader."
            )

        try:
            prompt = self._registry.load(name_or_uri, timeout_s=_MLFLOW_LOOKUP_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 — surface all registry errors
            raise _PromptRegistryFailure(
                f"IPromptRegistry.load({name_or_uri!r}) failed: {exc}"
            ) from exc

        if prompt is None:
            return None

        template = getattr(prompt, "template", None)
        if not template:
            logger.warning(
                f"[SYSTEM_PROMPT] MLflow prompt {name_or_uri!r} has an empty template."
            )
            return None

        text = str(template).strip()
        if not text:
            logger.warning(
                f"[SYSTEM_PROMPT] MLflow prompt {name_or_uri!r} template is blank."
            )
            return None

        result = SystemPromptResult(
            text=text,
            source={
                "type": "mlflow",
                "name": str(getattr(prompt, "name", name_or_uri)),
                "version": str(getattr(prompt, "version", "")),
            },
        )
        self._cache_store(name_or_uri, result)
        logger.info(
            f"[SYSTEM_PROMPT] loaded from MLflow: name={result.source.get('name')!r} "
            f"version={result.source.get('version')!r} ({len(text)} chars)"
        )
        return result

    def _handle_mlflow_failure(
        self,
        *,
        exc: Exception | None,
        on_failure: OnMlflowFailure,
        file_path_fallback: str | None,
        name: str,
    ) -> SystemPromptResult | None:
        """Apply the configured failure mode."""
        if on_failure == "fail":
            raise RuntimeError(
                f"[SYSTEM_PROMPT] MLflow load failed for {name!r}: {exc}"
            ) from exc

        if on_failure == "fallback_to_file" and file_path_fallback:
            logger.warning(
                f"[SYSTEM_PROMPT] MLflow load failed for {name!r}; "
                f"falling back to file {file_path_fallback!r}. cause={exc}"
            )
            return self._load_from_file(file_path_fallback)

        # "warn" — or fallback requested without a fallback path.
        logger.warning(
            f"[SYSTEM_PROMPT] MLflow load failed for {name!r}; continuing without "
            f"system prompt. cause={exc}"
        )
        return None

    # ------------------------------------------------------------------
    # File path
    # ------------------------------------------------------------------

    @staticmethod
    def _load_from_file(path_str: str) -> SystemPromptResult | None:
        """Read system prompt text from a local file.

        Returns ``None`` (with a warning) on missing file, IO error, or
        empty content. Never raises — matches legacy behaviour.
        """
        path = Path(path_str).expanduser()
        if not path.exists():
            logger.warning(
                f"[SYSTEM_PROMPT] system_prompt_path file not found: {path}. "
                "Continuing without system prompt."
            )
            return None

        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning(
                f"[SYSTEM_PROMPT] failed to read system prompt file {path}: {exc}. "
                "Continuing without."
            )
            return None

        if not content:
            logger.warning(
                f"[SYSTEM_PROMPT] system_prompt file is empty: {path}. Continuing without."
            )
            return None

        logger.info(f"[SYSTEM_PROMPT] loaded from file: {path} ({len(content)} chars)")
        return SystemPromptResult(
            text=content,
            source={"type": "file", "path": str(path)},
        )

    # ------------------------------------------------------------------
    # Cache helpers (private)
    # ------------------------------------------------------------------

    def _cache_lookup(self, key: str) -> SystemPromptResult | None:
        """Return a fresh cached result, or ``None`` on miss / expiry."""
        now = time.monotonic()
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            result, cached_at = entry
            if now - cached_at > self._cache_ttl_s:
                # Stale — evict so the next caller re-fetches.
                self._cache.pop(key, None)
                return None
            return result

    def _cache_store(self, key: str, result: SystemPromptResult) -> None:
        """Insert into the cache, applying FIFO eviction on overflow."""
        now = time.monotonic()
        with self._cache_lock:
            # Re-insert (move to end on update) for predictable FIFO order.
            if key in self._cache:
                self._cache.pop(key)
            self._cache[key] = (result, now)

            while len(self._cache) > self._cache_maxsize:
                # Drop the oldest (insertion-order) entry.
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key, None)


class _PromptRegistryFailure(RuntimeError):
    """Internal marker for registry-side failures.

    Wraps both ``registry is None`` misconfigurations and exceptions
    raised by ``IPromptRegistry.load``. Caller maps to the user-visible
    failure mode chosen on :meth:`SystemPromptLoader.load`.

    Named ``Failure`` rather than ``Error`` because it is a purely
    internal signal — never crosses an API boundary, never serialised
    as RFC 9457 problem+json. The ``test_exception_root`` sentinel
    only catches user-facing ``*Error`` classes that should root in
    :class:`ryotenkai_shared.errors.RyotenkAIError`; this marker
    intentionally sidesteps it.
    """


__all__ = [
    "OnMlflowFailure",
    "SystemPromptLoader",
    "SystemPromptResult",
]
