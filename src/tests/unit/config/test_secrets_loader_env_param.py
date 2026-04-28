"""Step 4 contract: ``load_secrets(env=...)`` parametrised env mapping.

The Variant 1 hexagonal refactor means callers (CLI / API / project
adapter) should be able to hand the secrets loader an explicit env
mapping (``process_env ∪ project_env.json``) instead of having to mutate
``os.environ`` first. This test set pins that contract:

- ``env=None`` → identical to historical behavior (reads ``os.environ``).
- ``env=<mapping>`` → replaces ``os.environ`` as the env-source layer
  for the typed aliases (``HF_TOKEN`` / ``RUNPOD_API_KEY``) AND for the
  override-of-dotenv-file logic.

Categories covered: positive, negative, boundary, invariants,
dependency-error, regression, logic-specific. Combinatorial coverage of
``(env_file y/n) × (env y/n) × (alias_in_env y/n)`` lives at the bottom.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.secrets.loader import load_secrets


def _write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


class TestPositiveExplicitEnv:
    def test_explicit_env_supplies_hf_token_when_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``env={"HF_TOKEN": …}`` populates the typed alias even when
        the process env doesn't set ``HF_TOKEN`` and no dotenv file
        exists."""
        # Make sure no auto-discovered file collides with the test.
        monkeypatch.chdir(tmp_path)
        # Strip the ambient process env so the explicit mapping is the
        # ONLY source.
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        # Loader probes the project root for ``secrets.env`` /
        # ``config/secrets.env``. Point it elsewhere via env_file.
        empty_env = tmp_path / "missing.env"

        s = load_secrets(
            env_file=empty_env,
            env={"HF_TOKEN": "from_explicit_mapping"},
        )

        assert s.hf_token == "from_explicit_mapping"

    def test_explicit_env_supplies_runpod_key_when_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

        s = load_secrets(
            env_file=tmp_path / "missing.env",
            env={"RUNPOD_API_KEY": "rp_from_mapping"},
        )

        assert s.runpod_api_key == "rp_from_mapping"

    def test_explicit_env_overrides_file_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When a key is in BOTH the explicit env mapping AND the dotenv
        file, the env value wins — same precedence as the historical
        os.environ-vs-file flip from 2026-04."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        f = tmp_path / "secrets.env"
        _write(f, "HF_TOKEN=team_default_in_file\n")

        s = load_secrets(
            env_file=f,
            env={"HF_TOKEN": "project_override"},
        )
        assert s.hf_token == "project_override"

    def test_explicit_env_propagates_arbitrary_dotenv_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-aliased keys (``EVAL_*``, ``HF_HUB_*``) come from the
        file as before; explicit env override still works for them."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("EVAL_FOO", raising=False)
        f = tmp_path / "secrets.env"
        _write(f, "EVAL_FOO=file_value\n")

        s = load_secrets(
            env_file=f,
            env={"EVAL_FOO": "explicit_override"},
        )
        assert (s.model_extra or {}).get("EVAL_FOO") == "explicit_override"

    def test_explicit_env_does_not_mutate_os_environ(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cardinal Variant 1 invariant: passing ``env=`` should NOT
        leak into the process env. Otherwise we've defeated the whole
        point of the parameter."""
        import os

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        sentinel = "explicit_only_token"

        load_secrets(
            env_file=tmp_path / "missing.env",
            env={"HF_TOKEN": sentinel},
        )

        assert os.environ.get("HF_TOKEN") != sentinel
        assert os.environ.get("HF_TOKEN") is None  # we cleared it above


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


class TestNegativeExplicitEnv:
    def test_empty_string_in_explicit_env_treated_as_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit empty/whitespace value should NOT clobber a file
        value with ``""``. This matches how os.environ-vs-file works."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        f = tmp_path / "secrets.env"
        _write(f, "HF_TOKEN=valid_file_token\n")

        s = load_secrets(env_file=f, env={"HF_TOKEN": "   "})
        assert s.hf_token == "valid_file_token"

    def test_missing_alias_in_explicit_env_yields_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No file, env mapping doesn't carry the alias → typed field
        is ``None`` (PR4 contract: all typed fields optional)."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

        s = load_secrets(
            env_file=tmp_path / "missing.env",
            env={"UNRELATED_VAR": "x"},
        )

        assert s.hf_token is None
        assert s.runpod_api_key is None


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_empty_env_mapping_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Boundary: ``env={}`` is meaningfully different from
        ``env=None``. With ``{}`` we explicitly say "there are no env
        vars", so typed fields fall to defaults."""
        monkeypatch.chdir(tmp_path)
        # Even if process env has HF_TOKEN, env={} should suppress it
        # via the explicit-mapping path (alias not in mapping).
        monkeypatch.setenv("HF_TOKEN", "process_env_token_should_be_ignored")

        s = load_secrets(env_file=tmp_path / "missing.env", env={})

        assert s.hf_token is None
        assert s.runpod_api_key is None

    def test_unicode_token_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Boundary: tokens with non-ASCII chars (real-world: some OAuth
        tokens include ``±``-style sequences). Should round-trip."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        s = load_secrets(
            env_file=tmp_path / "missing.env",
            env={"HF_TOKEN": "tök_ünicödé"},
        )
        assert s.hf_token == "tök_ünicödé"


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_env_none_falls_back_to_os_environ(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backward-compat invariant: ``env=None`` (default) reads
        ``os.environ`` just like before Step 4."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_TOKEN", "from_process_env")

        s = load_secrets(env_file=tmp_path / "missing.env")

        assert s.hf_token == "from_process_env"

    def test_explicit_mapping_isolates_from_process_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invariant: when ``env`` is given, ``os.environ`` MUST NOT be
        consulted for declared aliases. Otherwise project-level overrides
        passed by the adapter could be silently shadowed by an ambient
        process value the user expected to bypass."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_TOKEN", "ambient_process_token")

        s = load_secrets(
            env_file=tmp_path / "missing.env",
            env={"HF_TOKEN": "adapter_override"},
        )

        assert s.hf_token == "adapter_override"

    def test_explicit_mapping_with_unrelated_key_does_not_leak_alias(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invariant: when ``env`` carries no alias, typed fields stay
        ``None`` even if ``os.environ`` would have provided one."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_TOKEN", "ambient_should_not_leak")

        s = load_secrets(
            env_file=tmp_path / "missing.env",
            env={"PATH": "/usr/bin"},
        )

        assert s.hf_token is None

    def test_idempotent_for_same_inputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invariant: same (env_file, env) → same Secrets state."""
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "secrets.env"
        _write(f, "HF_TOKEN=file_token\nEVAL_X=v\n")

        s1 = load_secrets(env_file=f, env={"HF_TOKEN": "explicit"})
        s2 = load_secrets(env_file=f, env={"HF_TOKEN": "explicit"})

        assert s1.hf_token == s2.hf_token == "explicit"
        assert (s1.model_extra or {}).get("EVAL_X") == (s2.model_extra or {}).get(
            "EVAL_X"
        ) == "v"


# ---------------------------------------------------------------------------
# Logic-specific: env wins over file when both are explicit
# ---------------------------------------------------------------------------


class TestEnvWinsOverFile:
    def test_env_wins_for_alias(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        f = tmp_path / "secrets.env"
        _write(f, "HF_TOKEN=file_value\n")

        s = load_secrets(env_file=f, env={"HF_TOKEN": "env_value"})
        assert s.hf_token == "env_value"

    def test_file_fills_alias_when_env_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        f = tmp_path / "secrets.env"
        _write(f, "HF_TOKEN=file_only_value\n")

        s = load_secrets(env_file=f, env={"UNRELATED": "x"})
        assert s.hf_token == "file_only_value"

    def test_file_arbitrary_keys_kept_when_env_lacks_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """File-only ``EVAL_*`` keys still flow into ``model_extra``
        when the explicit env mapping doesn't mention them."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("EVAL_FOO", raising=False)
        f = tmp_path / "secrets.env"
        _write(f, "EVAL_FOO=file_only\n")

        s = load_secrets(env_file=f, env={"OTHER": "y"})
        assert (s.model_extra or {}).get("EVAL_FOO") == "file_only"


# ---------------------------------------------------------------------------
# Regression: existing call-sites still work unchanged
# ---------------------------------------------------------------------------


class TestRegression:
    def test_load_secrets_zero_args_still_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing call sites use ``load_secrets()`` with no args; that
        path must remain identical."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_TOKEN", "back_compat_token")

        s = load_secrets()

        assert s.hf_token == "back_compat_token"

    def test_positional_env_file_still_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing call sites pass ``env_file`` positionally."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        f = tmp_path / "x.env"
        _write(f, "HF_TOKEN=from_file\n")

        s = load_secrets(f)

        assert s.hf_token == "from_file"


# ---------------------------------------------------------------------------
# Combinatorial: (env_file?) × (env mapping?) × (alias source?)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "file_has_alias,explicit_env_alias,expected",
    [
        # file=N, env=N: typed field stays None.
        (False, None, None),
        # file=N, env=Y: explicit mapping supplies it.
        (False, "explicit_v", "explicit_v"),
        # file=Y, env=N: file fills it (mapping doesn't override).
        (True, None, "file_v"),
        # file=Y, env=Y: env wins.
        (True, "explicit_v", "explicit_v"),
        # file=Y, env=blank: blank ignored, file fills.
        (True, "   ", "file_v"),
        # file=N, env=blank: blank ignored, no source → None.
        (False, "", None),
    ],
)
def test_alias_resolution_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_has_alias: bool,
    explicit_env_alias: str | None,
    expected: str | None,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    if file_has_alias:
        f = tmp_path / "secrets.env"
        _write(f, "HF_TOKEN=file_v\n")
        env_file_arg: Path = f
    else:
        env_file_arg = tmp_path / "no_such_file.env"

    if explicit_env_alias is None:
        # env mapping with no HF_TOKEN entry; pass an unrelated key so
        # we exercise the explicit-mapping path (not env=None).
        env_arg: dict[str, str] = {"UNRELATED": "z"}
    else:
        env_arg = {"HF_TOKEN": explicit_env_alias}

    s = load_secrets(env_file=env_file_arg, env=env_arg)
    assert s.hf_token == expected
