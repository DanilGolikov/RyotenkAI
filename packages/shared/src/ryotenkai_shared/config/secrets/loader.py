"""Load the ``Secrets`` object with a clearly-defined precedence.

Precedence (highest → lowest):

1. **Per-provider encrypted token** — ``~/.ryotenkai/providers/<id>/token.enc``.
   Not resolved here; callers use ``secrets.get_provider_token(provider_id)``.
   The HF token is sourced exclusively from ``HF_TOKEN`` env / ``secrets.env``.
2. **Subprocess os.environ** — at runtime the Web-API launcher merges the
   project's ``env.json`` on top of the parent process env BEFORE spawning
   the training subprocess (see ``spawn_launch_detached``); so everything
   the user typed in the project's Settings → Env tab ends up here.
3. **Repo-root ``secrets.env`**` — shared team defaults (CLI-friendly,
   usually committed as ``secrets.env.example``). Candidates searched in
   order: explicit ``env_file`` argument, ``<repo>/secrets.env``,
   ``<repo>/config/secrets.env`` (legacy).
4. **Pydantic field defaults** — ``None`` for optional fields; raises for
   required ones.

**Policy change (2026-04):** when a key is defined both in ``os.environ``
and in the dotenv file, the **environment value wins**. This lets a
user's project-level override (edited in the Web UI) take effect even
when a repo-root ``secrets.env`` is present. Previously the file always
won, which silently ignored per-project overrides and caused
hard-to-debug auth failures.

A diagnostic warning is emitted (with SHA-256 of each value) whenever
env and file disagree so the precedence is visible in the logs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from .model import Secrets

if TYPE_CHECKING:
    from collections.abc import Mapping

# Field aliases declared on :class:`Secrets`. When an explicit ``env``
# mapping is passed, these are the keys we still pull through as
# init-kwargs so they reach typed fields; arbitrary plugin keys
# (``EVAL_*``, ``DTST_*``…) only enter via the dotenv file as before.
_DECLARED_ALIASES: tuple[str, ...] = ("HF_TOKEN", "RUNPOD_API_KEY")


def load_secrets(
    env_file: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Secrets:
    """Load secrets honouring the precedence above.

    Args:
        env_file: Explicit dotenv path. When provided and the file
            exists, it's used as the file-layer source and takes
            precedence over the repo-root auto-discovery.
        env: Explicit env mapping. When ``None`` (default), reads from
            ``os.environ`` — preserving every existing call-site's
            behavior. When a mapping is provided, it replaces
            ``os.environ`` as the env-layer source, without mutating
            the process env. Variant 1 path: callers (CLI / API /
            adapter) pass ``process_env ∪ project_env.json`` here so
            project overrides take effect without touching
            ``os.environ``.

    Returns:
        Secrets instance.

    Raises:
        ValidationError: If required secrets are missing from every
            source (never happens in v1 since PR4 made all fields
            optional).
    """
    # Local import to avoid heavy side-effects at module import time.
    from ryotenkai_shared.utils.logger import logger

    # Single source-of-truth for env reads inside this call. Default
    # path (``env=None``) preserves the historical contract: BaseSettings
    # auto-reads ``os.environ`` when we hand it ``Secrets()`` with no
    # kwargs.
    source: Mapping[str, str] = env if env is not None else os.environ

    candidates: list[Path] = []
    if env_file is not None:
        candidates.append(Path(env_file).expanduser())

    # Project-root auto-discovery is suppressed when the caller passed
    # an explicit ``env`` mapping. Variant 1 contract: an adapter that
    # supplies ``env`` is declaring "this mapping IS the env layer" —
    # silently merging a repo-root ``secrets.env`` would shadow the
    # adapter's project-level overrides with team defaults the caller
    # didn't ask for. Adapters that DO want the repo file as a fallback
    # must opt in by passing it as ``env_file`` explicitly.
    if env is None:
        # Walk up from this file looking for the workspace root —
        # identified by a ``pyproject.toml`` whose parent ALSO contains a
        # ``packages/`` directory (the uv workspace marker). Pinned
        # ``parents[N]`` was fragile across the Phase B packagization
        # move (loader.py is now 3 levels deeper than it used to be).
        try:
            here = Path(__file__).resolve()
            project_root: Path = Path.cwd()  # fallback
            for parent in here.parents:
                if (parent / "pyproject.toml").is_file() and (parent / "packages").is_dir():
                    project_root = parent
                    break
        except Exception:
            project_root = Path.cwd()

        candidates.extend(
            [
                project_root / "secrets.env",
                project_root / "config" / "secrets.env",
            ]
        )

    chosen_file: Path | None = next((p for p in candidates if p.is_file()), None)

    if chosen_file is None:
        logger.debug("[SECRETS] Loading from environment only (no secrets.env found)")
        if env is None:
            # Legacy path: BaseSettings auto-reads os.environ for
            # declared aliases (HF_TOKEN / RUNPOD_API_KEY).
            return Secrets()  # type: ignore[call-arg]
        # Explicit env path: BaseSettings would still read os.environ
        # for any alias we DON'T pass as kwarg. Pass every declared
        # alias explicitly — populated from the mapping, ``None`` when
        # absent — so BaseSettings has no chance to fall back to
        # ``os.environ``. Init kwargs always win over env sources in
        # Pydantic Settings' precedence chain.
        alias_kwargs = _explicit_alias_kwargs(source)
        return Secrets(**alias_kwargs)  # type: ignore[arg-type]

    _log_mismatches(chosen_file, logger, source=source)

    # Read every key from the dotenv file, typed or not — keys that are
    # declared on ``Secrets`` (``HF_TOKEN`` / ``RUNPOD_API_KEY``) go into
    # typed fields via aliases; arbitrary keys (``HF_HUB_*``, ``EVAL_*``)
    # flow into ``model_extra`` thanks to ``extra='allow'``.
    file_entries = _read_dotenv(chosen_file)

    # Env wins: for each key present in the file, replace its value with
    # the env-source value when the var is set and non-empty. Keys stay
    # in ``init_kwargs`` so ``model_extra`` picks up arbitrary plugin
    # keys (e.g. ``EVAL_CEREBRAS_API_KEY``) uniformly — BaseSettings on
    # its own does not read undeclared env vars.
    init_kwargs: dict[str, str] = {}
    overridden: list[str] = []
    for key, file_value in file_entries.items():
        env_value = source.get(key)
        if env_value is not None and env_value.strip():
            init_kwargs[key] = env_value.strip()
            if env_value.strip() != file_value.strip():
                overridden.append(key)
        else:
            init_kwargs[key] = file_value

    # Ensure declared aliases from the env source reach typed fields
    # even when the dotenv file doesn't list them. With the default
    # source (``os.environ``) BaseSettings handles that path itself,
    # but with an explicit ``env`` mapping we must thread it through
    # ourselves AND explicitly null-out any alias the mapping doesn't
    # carry — otherwise BaseSettings would silently fall back to
    # ``os.environ`` for those aliases and shadow the project-level
    # override the caller meant to enforce.
    if env is not None:
        for alias in _DECLARED_ALIASES:
            if alias in init_kwargs:
                continue
            value = source.get(alias)
            if value is not None and value.strip():
                init_kwargs[alias] = value.strip()
            else:
                # Explicit None pins the field so BaseSettings won't
                # read ``os.environ``. The model's
                # ``_normalize_hf_token`` validator coerces ``None``
                # to ``None`` cleanly.
                init_kwargs[alias] = None  # type: ignore[assignment]

    logger.debug(
        "[SECRETS] Loaded from %s; env-overridden keys: %s",
        chosen_file,
        sorted(overridden),
    )
    return Secrets(**init_kwargs)  # type: ignore[arg-type]


def _explicit_alias_kwargs(source: Mapping[str, str]) -> dict[str, str | None]:
    """Build init kwargs for every declared alias from an explicit env.

    Each alias is included regardless of whether it appears in
    ``source`` — the caller's contract is "this mapping replaces
    ``os.environ``", so missing keys must surface as explicit ``None``
    rather than letting BaseSettings re-read the process env.
    """
    out: dict[str, str | None] = {}
    for alias in _DECLARED_ALIASES:
        value = source.get(alias)
        if value is not None and value.strip():
            out[alias] = value.strip()
        else:
            out[alias] = None
    return out


def _read_dotenv(path: Path) -> dict[str, str]:
    """Minimal dotenv parser: ``KEY=value`` lines, optional ``export``
    prefix, quoted values unwrapped. Returns a {key: value} dict."""
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return entries

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        entries[key] = value.strip()
    return entries


def _log_mismatches(
    path: Path,
    logger,
    *,
    source: Mapping[str, str] | None = None,
) -> None:
    """Emit a warning when a key is present in both env and the file
    with different values. Covers the most common HF_TOKEN rotation
    footgun. Uses SHA-256 hashes so we never log the secret itself.

    ``source`` defaults to ``os.environ`` — supplied explicitly when
    the caller passes a ``env`` mapping to :func:`load_secrets` so the
    diagnostic reflects the same env layer the loader actually uses.
    """
    try:
        import hashlib

        env_source: Mapping[str, str] = source if source is not None else os.environ

        file_entries = _read_dotenv(path)
        from .constants import SHA256_HEXDIGEST_DISPLAY_LEN

        for key, file_val in file_entries.items():
            env_val = env_source.get(key)
            if env_val is None:
                continue
            if env_val.strip() == file_val.strip():
                continue
            env_sha = hashlib.sha256(env_val.strip().encode()).hexdigest()[:SHA256_HEXDIGEST_DISPLAY_LEN]
            file_sha = hashlib.sha256(file_val.strip().encode()).hexdigest()[:SHA256_HEXDIGEST_DISPLAY_LEN]
            logger.warning(
                "[SECRETS] %s differs between env (sha256=%s) and %s (sha256=%s). "
                "Environment value wins (project override > file).",
                key,
                env_sha,
                path,
                file_sha,
            )
    except Exception:
        # Diagnostics must never block secrets loading.
        pass


__all__ = [
    "load_secrets",
]
