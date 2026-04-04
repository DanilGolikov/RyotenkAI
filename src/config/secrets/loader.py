from __future__ import annotations

from pathlib import Path

from .model import Secrets


def load_secrets(env_file: str | Path | None = None) -> Secrets:
    """
    Load secrets from a single "source of truth".

    Policy (project convention):
    - If a secrets dotenv file exists (explicit `env_file` or auto-detected), prefer it.
    - Otherwise, fall back to real OS environment variables.

    Returns:
        Secrets: Secrets object

    Raises:
        ValidationError: If required secrets are missing
    """
    # Local import to avoid heavy side-effects at module import time.
    from src.utils.logger import logger

    # NOTE:
    # Historically we used `config/secrets.env`, but in practice many setups keep a single
    # repo-root `secrets.env`.
    #
    # Priority (file-first, then env):
    # 1) explicit env_file argument (if provided and exists)
    # 2) <project_root>/secrets.env
    # 3) <project_root>/config/secrets.env (legacy)
    # 4) fallback: environment variables only

    candidates: list[Path] = []
    if env_file is not None:
        candidates.append(Path(env_file).expanduser())

    try:
        project_root = Path(__file__).resolve().parents[3]  # .../src/config/secrets/loader.py -> repo root
    except Exception:
        project_root = Path.cwd()

    candidates.extend(
        [
            project_root / "secrets.env",
            project_root / "config" / "secrets.env",
        ]
    )

    def _dotenv_get(path: Path, key: str) -> str | None:
        """
        Minimal .env parser (key=value), used for file-first secrets loading.

        We intentionally avoid printing secret values; call-sites should log hashes only.
        """
        try:
            for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                # Allow both "KEY=..." and "export KEY=..."
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                k, v = line.split("=", 1)
                if k.strip() != key:
                    continue
                value = v.strip()
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                return value.strip()
        except Exception:
            return None
        return None

    for p in candidates:
        if p.exists():
            # Helpful diagnostics: dotenv is preferred, so env differences are ignored.
            # This is a very common source of confusion when HF_TOKEN is rotated.
            try:
                import hashlib
                import os

                env_tok = os.environ.get("HF_TOKEN")
                file_tok = _dotenv_get(p, "HF_TOKEN")
                if env_tok and file_tok and env_tok.strip() != file_tok.strip():
                    from .constants import SHA256_HEXDIGEST_DISPLAY_LEN

                    env_sha = hashlib.sha256(env_tok.strip().encode("utf-8")).hexdigest()[:SHA256_HEXDIGEST_DISPLAY_LEN]
                    file_sha = hashlib.sha256(file_tok.strip().encode("utf-8")).hexdigest()[
                        :SHA256_HEXDIGEST_DISPLAY_LEN
                    ]
                    logger.warning(
                        "[SECRETS] HF_TOKEN is set in environment and differs from %s. "
                        "This project prefers the dotenv token for local runs; environment HF_TOKEN will be ignored. "
                        "env_sha256=%s file_sha256=%s",
                        str(p),
                        env_sha,
                        file_sha,
                    )
            except Exception:
                # Never fail secrets loading due to diagnostics
                pass

            # File-first source of truth: if the file exists but doesn't define HF_TOKEN, fail fast.
            # (Otherwise we'd silently fall back to environment, which defeats the purpose of a single source.)
            if _dotenv_get(p, "HF_TOKEN") is None:
                raise ValueError(f"Secrets file exists but HF_TOKEN is missing: {p}")

            # Build init kwargs from file using field aliases (auto-support new secrets fields).
            # IMPORTANT: our Secrets model uses aliases like HF_TOKEN; by default pydantic requires
            # population via aliases (not field names), so we pass aliases here.
            init_kwargs: dict[str, str] = {}
            for field_name, field_info in Secrets.model_fields.items():
                alias = field_info.alias or field_name
                v = _dotenv_get(p, str(alias))
                if v is not None:
                    init_kwargs[str(alias)] = v

            # Also collect arbitrary plugin / env-forward keys from the file so they land in
            # Secrets.model_extra (extra="allow"). This is needed for keys like HF_HUB_DISABLE_XET
            # that are not declared as model fields but should be accessible via model_extra.
            # We avoid double-adding already-captured field aliases.
            known_aliases = {str(fi.alias or fn) for fn, fi in Secrets.model_fields.items()}
            try:
                for raw_line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):].strip()
                    k, v_raw = line.split("=", 1)
                    key = k.strip()
                    if key in known_aliases or key in init_kwargs:
                        continue
                    val = v_raw.strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    val = val.strip()
                    if val:
                        init_kwargs[key] = val
            except Exception:
                pass  # extra keys are best-effort; never fail secrets loading

            logger.debug(f"[SECRETS] Loading secrets from env_file (preferred): {p}")
            return Secrets(**init_kwargs)  # type: ignore[call-arg]

    logger.debug("[SECRETS] Loading secrets from environment variables (no env_file found)")
    return Secrets()  # type: ignore[call-arg]  # BaseSettings loads from env vars


__all__ = [
    "load_secrets",
]
