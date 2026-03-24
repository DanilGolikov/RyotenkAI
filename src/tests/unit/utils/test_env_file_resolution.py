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

    def test_env_file_overrides_environment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_env_token")

        env_path = tmp_path / "my.env"
        _write_env(
            env_path,
            """
HF_TOKEN=hf_file_token
""",
        )

        s = load_secrets(env_file=env_path)
        assert s.hf_token == "hf_file_token"


