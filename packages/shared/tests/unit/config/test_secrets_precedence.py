"""Precedence contract for env vs file sources (2026-04 flip).

Context: the Web-launch flow merges each project's ``env.json`` into
the subprocess environment before ``load_secrets`` runs in the child
(see ``launch_service._project_env_for_run_dir`` +
``spawn_launch_detached``). For that override to actually take effect
when a repo-root ``secrets.env`` is also present, env must win over
the file. This test set pins that contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.secrets.loader import load_secrets


def _write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


class TestProjectOverrideScenario:
    """End-to-end: what happens when launcher merges ``env.json`` and
    a team ``secrets.env`` coexists.
    """

    def test_project_env_override_wins_over_repo_secrets_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Repo has a team default token.
        repo_env = tmp_path / "secrets.env"
        _write(repo_env, "HF_TOKEN=team_default_token\nRUNPOD_API_KEY=team_rp\n")

        # Simulate launcher merging per-project env.json into the
        # subprocess env — user typed these in Settings → Env.
        monkeypatch.setenv("HF_TOKEN", "project_override_token")
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

        s = load_secrets(env_file=repo_env)
        # HF_TOKEN was present in both → env wins (project override).
        assert s.hf_token == "project_override_token"
        # RUNPOD_API_KEY was only in the file → file fills the gap.
        assert s.runpod_api_key == "team_rp"

    def test_extra_file_keys_still_flow_to_model_extra(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Arbitrary plugin keys (``HF_HUB_*``, ``EVAL_*``) that aren't
        declared as fields must still land in ``model_extra`` so
        ``SecretsResolver`` can find them."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
        repo_env = tmp_path / "secrets.env"
        _write(repo_env, "HF_HUB_DISABLE_XET=1\nEVAL_CEREBRAS_API_KEY=sk-test\n")

        s = load_secrets(env_file=repo_env)
        extra = s.model_extra or {}
        assert extra.get("HF_HUB_DISABLE_XET") == "1"
        assert extra.get("EVAL_CEREBRAS_API_KEY") == "sk-test"

    def test_env_override_applies_to_extra_keys_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Precedence applies uniformly — a plugin key set in env
        (e.g. via project env.json) overrides the file copy."""
        repo_env = tmp_path / "secrets.env"
        _write(repo_env, "EVAL_CEREBRAS_API_KEY=team_key\n")
        monkeypatch.setenv("EVAL_CEREBRAS_API_KEY", "per_project_key")

        s = load_secrets(env_file=repo_env)
        extra = s.model_extra or {}
        assert extra.get("EVAL_CEREBRAS_API_KEY") == "per_project_key"

    def test_no_file_no_env_yields_all_optional_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PR4 made typed fields optional. With no file AND no env the
        loader still returns a valid ``Secrets`` (all ``None``)."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        # Ensure auto-discovery doesn't pick up a real repo-root
        # ``secrets.env`` on the dev machine.
        s = load_secrets(env_file=tmp_path / "nonexistent.env")
        assert s.hf_token is None
        assert s.runpod_api_key is None
