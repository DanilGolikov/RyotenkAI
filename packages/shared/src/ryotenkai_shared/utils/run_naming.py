from __future__ import annotations

import secrets
import string
from datetime import datetime, timezone
from pathlib import Path

_RUN_ID_ALPHABET = string.ascii_lowercase + string.digits


def generate_run_name(*, now_utc: datetime | None = None, id_length: int = 5) -> tuple[str, datetime]:
    """Generate a canonical run name in UTC."""
    if id_length <= 0:
        raise ValueError("id_length must be > 0")

    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware (UTC)")

    ts = now.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = "".join(secrets.choice(_RUN_ID_ALPHABET) for _ in range(id_length))
    return f"run_{ts}_{run_id}", now.astimezone(timezone.utc)


def build_run_directory(
    *, base_dir: Path | None = None, now_utc: datetime | None = None, id_length: int = 5
) -> tuple[Path, datetime]:
    """Build a canonical run directory path from the shared run name generator."""
    run_name, created_at = generate_run_name(now_utc=now_utc, id_length=id_length)
    return (base_dir or Path("runs")) / run_name, created_at
