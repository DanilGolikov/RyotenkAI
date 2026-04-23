from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import load_secrets


def _write_env(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


class TestLoadSecretsEnvFileResolution:
    def test_explicit_env_file_is_used(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)

        env_path = tmp_path / "my.env"
        _write_env(
            env_path,
            """
HF_TOKEN=hf_file_token
RUNPOD_API_KEY=rk_file_token
""",
        )

        s = load_secrets(env_file=env_path)
        assert s.hf_token == "hf_file_token"
        assert s.runpod_api_key == "rk_file_token"

    def test_environment_overrides_env_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Precedence (2026-04): env > file.

        Rationale: the Web-launch flow merges a project's ``env.json``
        into the subprocess environment before ``load_secrets`` runs.
        When the user edits ``HF_TOKEN`` in Settings → Env for a
        specific project, that value MUST win even if a repo-root
        ``secrets.env`` is present with a shared team default.

        Covers the opposite of the old contract — see the git log for
        the rationale of the flip.
        """
        monkeypatch.setenv("HF_TOKEN", "hf_env_token")

        env_path = tmp_path / "my.env"
        _write_env(
            env_path,
            """
HF_TOKEN=hf_file_token
""",
        )

        s = load_secrets(env_file=env_path)
        assert s.hf_token == "hf_env_token"

    def test_file_fills_keys_absent_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env wins only when the env variable is actually set — keys
        present only in the file are still picked up."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

        env_path = tmp_path / "my.env"
        _write_env(
            env_path,
            """
HF_TOKEN=hf_file_token
RUNPOD_API_KEY=rp_file_token
""",
        )

        s = load_secrets(env_file=env_path)
        assert s.hf_token == "hf_file_token"
        assert s.runpod_api_key == "rp_file_token"

    def test_empty_env_value_does_not_shadow_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty-string env var (common artefact of shell bugs) must
        NOT silently wipe a real value from the file."""
        monkeypatch.setenv("HF_TOKEN", "")
        env_path = tmp_path / "my.env"
        _write_env(env_path, "HF_TOKEN=hf_file_token\n")

        s = load_secrets(env_file=env_path)
        assert s.hf_token == "hf_file_token"


