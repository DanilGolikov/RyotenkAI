"""Load the ``Secrets`` object with a clearly-defined precedence.

Precedence (highest → lowest):

1. **Per-resource encrypted token** — ``~/.ryotenkai/{providers,integrations}/<id>/token.enc``.
   Not resolved here; callers use ``secrets.get_hf_token(integration_id)`` or
   ``secrets.get_provider_token(provider_id)``.
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

from pathlib import Path

from .model import Secrets


def load_secrets(env_file: str | Path | None = None) -> Secrets:
    """Load secrets honouring the precedence above.

    Args:
        env_file: Explicit dotenv path. When provided and the file
            exists, it's used as the file-layer source and takes
            precedence over the repo-root auto-discovery.

    Returns:
        Secrets instance.

    Raises:
        ValidationError: If required secrets are missing from every
            source (never happens in v1 since PR4 made all fields
            optional).
    """
    # Local import to avoid heavy side-effects at module import time.
    from src.utils.logger import logger

    candidates: list[Path] = []
    if env_file is not None:
        candidates.append(Path(env_file).expanduser())

    try:
        project_root = Path(__file__).resolve().parents[3]
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
        return Secrets()  # type: ignore[call-arg]  # BaseSettings loads from env

    _log_mismatches(chosen_file, logger)

    # Read every key from the dotenv file, typed or not — keys that are
    # declared on ``Secrets`` (``HF_TOKEN`` / ``RUNPOD_API_KEY``) go into
    # typed fields via aliases; arbitrary keys (``HF_HUB_*``, ``EVAL_*``)
    # flow into ``model_extra`` thanks to ``extra='allow'``.
    file_entries = _read_dotenv(chosen_file)

    # Env wins: for each key present in the file, replace its value with
    # the process-env value when the env var is set and non-empty. Keys
    # stay in ``init_kwargs`` so ``model_extra`` picks up arbitrary plugin
    # keys (e.g. ``EVAL_CEREBRAS_API_KEY``) uniformly — BaseSettings on
    # its own does not read undeclared env vars.
    import os as _os

    init_kwargs: dict[str, str] = {}
    overridden: list[str] = []
    for key, file_value in file_entries.items():
        env_value = _os.environ.get(key)
        if env_value is not None and env_value.strip():
            init_kwargs[key] = env_value.strip()
            if env_value.strip() != file_value.strip():
                overridden.append(key)
        else:
            init_kwargs[key] = file_value

    logger.debug(
        "[SECRETS] Loaded from %s; env-overridden keys: %s",
        chosen_file,
        sorted(overridden),
    )
    return Secrets(**init_kwargs)  # type: ignore[arg-type]


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
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        entries[key] = value.strip()
    return entries


def _log_mismatches(path: Path, logger) -> None:
    """Emit a warning when a key is present in both env and the file
    with different values. Covers the most common HF_TOKEN rotation
    footgun. Uses SHA-256 hashes so we never log the secret itself."""
    try:
        import hashlib
        import os

        file_entries = _read_dotenv(path)
        from .constants import SHA256_HEXDIGEST_DISPLAY_LEN

        for key, file_val in file_entries.items():
            env_val = os.environ.get(key)
            if env_val is None:
                continue
            if env_val.strip() == file_val.strip():
                continue
            env_sha = hashlib.sha256(env_val.strip().encode()).hexdigest()[
                :SHA256_HEXDIGEST_DISPLAY_LEN
            ]
            file_sha = hashlib.sha256(file_val.strip().encode()).hexdigest()[
                :SHA256_HEXDIGEST_DISPLAY_LEN
            ]
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
