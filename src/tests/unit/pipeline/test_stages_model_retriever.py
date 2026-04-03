"""
Unit tests for Model Retriever stage.

Regression coverage:
- HF upload previously failed while generating model card due to legacy access to `phase.epochs`.
- HF privacy flag should be enforced even when repo already exists (create_repo + update_repo_settings).
"""

from __future__ import annotations

import json
import types
from unittest.mock import MagicMock

import pytest

from src.pipeline.stages.model_retriever import ModelCardContext, ModelRetriever, PhaseMetricsResult
from src.utils.config import HuggingFaceHubConfig, PhaseHyperparametersConfig, StrategyPhaseConfig
from src.utils.result import Err, Ok


@pytest.fixture
def mock_secrets() -> MagicMock:
    secrets = MagicMock()
    secrets.hf_token = "hf_test_token"
    return secrets


@pytest.fixture
def mock_config_with_hf() -> MagicMock:
    cfg = MagicMock()

    # Provider metadata (used in model card)
    cfg.get_active_provider_name.return_value = "single_node"
    cfg.get_provider_config.return_value = {"gpu_type": "NVIDIA_TEST", "mock_mode": False}

    # Experiment tracking (HF config)
    cfg.experiment_tracking.huggingface = HuggingFaceHubConfig(
        enabled=True,
        repo_id="org/test-model",
        private=True,
    )

    # Model + training fields used by _generate_model_card
    cfg.model.name = "Qwen/Qwen2.5-0.5B-Instruct"
    cfg.training.type = "qlora"
    cfg.training.hyperparams = PhaseHyperparametersConfig(
        epochs=3,
        per_device_train_batch_size=8,
    )
    cfg.training.get_strategy_chain.return_value = [
        StrategyPhaseConfig(strategy_type="cpt", dataset="ds_cpt", hyperparams=PhaseHyperparametersConfig(epochs=1)),
        StrategyPhaseConfig(strategy_type="sft", dataset="ds_sft", hyperparams=PhaseHyperparametersConfig()),
    ]

    # Make adapter lookup fail so _get_lora_param returns "N/A" deterministically
    cfg.get_adapter_config.side_effect = ValueError("adapter config not available in this unit test")

    return cfg


class TestModelCardHyperparams:
    def test_generate_model_card_uses_phase_hyperparams_not_legacy_epochs(self, mock_config_with_hf, mock_secrets):
        """
        Regression: `ModelRetriever._format_strategies()` used to access `phase.epochs` (removed after
        hyperparams unification) and crashed HF upload at model card generation time.
        """
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)

        # Sanity: StrategyPhaseConfig should NOT have legacy fields
        phase = mock_config_with_hf.training.get_strategy_chain.return_value[0]
        assert not hasattr(phase, "epochs")

        card = retriever._generate_model_card()
        assert "CPT (1ep)" in card
        assert "SFT (3ep)" in card  # fallback to global epochs
        assert "| **Batch Size** | 8 |" in card


class TestModelRetrieverHelpers:
    def test_format_strategies_default_when_no_chain(self, mock_config_with_hf, mock_secrets):
        mock_config_with_hf.training.get_strategy_chain.return_value = []
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        assert retriever._format_strategies(mock_config_with_hf) == "SFT (default)"

    def test_get_lora_param_reads_adapter_config_when_available(self, mock_secrets):
        cfg = MagicMock()
        cfg.get_active_provider_name.return_value = "single_node"
        cfg.get_provider_config.return_value = {"gpu_type": "NVIDIA_TEST", "mock_mode": False}
        cfg.experiment_tracking.huggingface = HuggingFaceHubConfig(enabled=True, repo_id="org/test-model", private=True)
        cfg.model.name = "base"
        cfg.training.type = "qlora"
        cfg.training.hyperparams = PhaseHyperparametersConfig(epochs=1, per_device_train_batch_size=1)
        cfg.training.get_strategy_chain.return_value = []
        cfg.get_adapter_config.return_value = types.SimpleNamespace(r=8, lora_alpha=16)

        retriever = ModelRetriever(cfg, mock_secrets)
        assert retriever._get_lora_param(cfg, "r") == "8"


class TestModelCardDatasetsExtraction:
    @pytest.mark.parametrize(
        "train_path,eval_path,expected",
        [
            # positive: normal paths
            ("/abs/path/train.jsonl", None, ["train.jsonl"]),
            ("/abs/path/train.jsonl", "/abs/path/eval.jsonl", ["train.jsonl", "eval.jsonl"]),
            (" train.jsonl ", " eval.jsonl ", ["train.jsonl", "eval.jsonl"]),
            ("relative/dir/train.jsonl", "relative/dir/train.jsonl", ["train.jsonl"]),  # duplicate removal
            # boundary: dirs / trailing slashes
            ("/abs/path/dir/", None, ["dir"]),
            ("/", None, []),  # no filename → should not leak path
            # boundary: windows-style paths (\\)
            (r"C:\secret\train.jsonl", None, ["train.jsonl"]),
            (r"C:\secret\train.jsonl", r"C:\secret\eval.jsonl", ["train.jsonl", "eval.jsonl"]),
            (r"C:\secret\dir\\", None, ["dir"]),
            # negative: empty/invalid values
            ("", "/abs/eval.jsonl", ["eval.jsonl"]),
            ("   ", "   ", []),
            (None, None, []),  # type: ignore[arg-type]
            (123, "eval.jsonl", ["eval.jsonl"]),  # type: ignore[arg-type]
        ],
    )
    def test_extract_datasets_for_readme_local_returns_basenames(
        self,
        mock_config_with_hf,
        mock_secrets,
        train_path,
        eval_path,
        expected,
    ):
        # Arrange: local dataset
        ds = types.SimpleNamespace()
        ds.get_source_type = lambda: "local"
        ds.source_hf = None
        ds.source_local = types.SimpleNamespace(local_paths=types.SimpleNamespace(train=train_path, eval=eval_path))
        mock_config_with_hf.get_primary_dataset.return_value = ds

        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)

        # Act
        out = retriever._extract_datasets_for_readme()

        # Assert
        assert out == expected
        # Invariant: local dataset must not leak path (no / or \\)
        for name in out:
            assert "/" not in name
            assert "\\" not in name

    @pytest.mark.parametrize(
        "train_id,eval_id,expected",
        [
            ("org/ds", None, ["org/ds"]),
            ("org/ds", "org/ds-eval", ["org/ds", "org/ds-eval"]),
            (" org/ds ", " org/ds-eval ", ["org/ds", "org/ds-eval"]),  # trimming
            ("org/ds", "org/ds", ["org/ds"]),  # duplicate removal
        ],
    )
    def test_extract_datasets_for_readme_hf_returns_ids(
        self, mock_config_with_hf, mock_secrets, train_id, eval_id, expected
    ):
        ds = types.SimpleNamespace()
        ds.get_source_type = lambda: "huggingface"
        ds.source_local = None
        ds.source_hf = types.SimpleNamespace(train_id=train_id, eval_id=eval_id)
        mock_config_with_hf.get_primary_dataset.return_value = ds

        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        out = retriever._extract_datasets_for_readme()
        assert out == expected

    @pytest.mark.parametrize(
        "source_type,expected",
        [
            ("local", "local"),
            ("huggingface", "huggingface"),
            ("custom", "custom"),  # future-proof: propagate non-empty strings
        ],
    )
    def test_extract_dataset_source_type_for_readme_returns_type(self, mock_config_with_hf, mock_secrets, source_type, expected):
        ds = types.SimpleNamespace()
        ds.get_source_type = lambda: source_type
        ds.source_local = None
        ds.source_hf = None
        mock_config_with_hf.get_primary_dataset.return_value = ds
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        assert retriever._extract_dataset_source_type_for_readme() == expected

    def test_extract_dataset_source_type_for_readme_handles_get_primary_dataset_error(self, mock_config_with_hf, mock_secrets):
        mock_config_with_hf.get_primary_dataset.side_effect = AttributeError("boom")
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        assert retriever._extract_dataset_source_type_for_readme() is None

    def test_extract_dataset_source_type_for_readme_handles_get_source_type_error(self, mock_config_with_hf, mock_secrets):
        ds = MagicMock()
        ds.get_source_type.side_effect = AttributeError("boom")
        mock_config_with_hf.get_primary_dataset.return_value = ds
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        assert retriever._extract_dataset_source_type_for_readme() is None

    def test_extract_datasets_for_readme_handles_get_primary_dataset_error(self, mock_config_with_hf, mock_secrets):
        # Dependency failure: config.get_primary_dataset() raises
        mock_config_with_hf.get_primary_dataset.side_effect = AttributeError("boom")
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        assert retriever._extract_datasets_for_readme() == []

    def test_extract_datasets_for_readme_handles_dataset_missing(self, mock_config_with_hf, mock_secrets):
        # Dependency failure: dataset missing / None
        mock_config_with_hf.get_primary_dataset.return_value = None
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        assert retriever._extract_datasets_for_readme() == []

    def test_extract_datasets_for_readme_handles_get_source_type_error(self, mock_config_with_hf, mock_secrets):
        # Negative: dataset.get_source_type() raises
        ds = MagicMock()
        ds.get_source_type.side_effect = AttributeError("boom")
        mock_config_with_hf.get_primary_dataset.return_value = ds
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        assert retriever._extract_datasets_for_readme() == []

    def test_extract_datasets_for_readme_missing_source_local_attr_does_not_crash(self, mock_config_with_hf, mock_secrets):
        # Negative: local claimed but source_local missing (AttributeError)
        ds = types.SimpleNamespace()
        ds.get_source_type = lambda: "local"
        ds.source_hf = None
        # no ds.source_local
        mock_config_with_hf.get_primary_dataset.return_value = ds
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        assert retriever._extract_datasets_for_readme() == []

    def test_extract_datasets_for_readme_missing_source_hf_attr_does_not_crash(self, mock_config_with_hf, mock_secrets):
        # Negative: huggingface claimed but source_hf missing (AttributeError)
        ds = types.SimpleNamespace()
        ds.get_source_type = lambda: "huggingface"
        ds.source_local = None
        # no ds.source_hf
        mock_config_with_hf.get_primary_dataset.return_value = ds
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        assert retriever._extract_datasets_for_readme() == []

    @pytest.mark.parametrize(
        "train_path,eval_path,secret_snippet,expected_names",
        [
            ("/very/secret/train.jsonl", "/very/secret/eval.jsonl", "/very/secret", ["train.jsonl", "eval.jsonl"]),
            (r"C:\secret\train.jsonl", None, r"C:\secret", ["train.jsonl"]),
        ],
    )
    def test_generate_model_card_local_datasets_no_path_leak_regression(
        self,
        mock_config_with_hf,
        mock_secrets,
        train_path,
        eval_path,
        secret_snippet,
        expected_names,
    ):
        # Regression: README must not contain full local dataset path
        ds = types.SimpleNamespace()
        ds.get_source_type = lambda: "local"
        ds.source_hf = None
        ds.source_local = types.SimpleNamespace(local_paths=types.SimpleNamespace(train=train_path, eval=eval_path))
        mock_config_with_hf.get_primary_dataset.return_value = ds

        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        datasets = retriever._extract_datasets_for_readme()
        assert datasets == expected_names
        ctx = ModelCardContext(phase_metrics=[], datasets=datasets, dataset_source_type="local")
        card = retriever._generate_model_card(ctx)

        assert secret_snippet not in card
        assert "| **Dataset source** | `local` |" in card
        # datasets must appear in YAML frontmatter
        assert "datasets:" in card
        for name in expected_names:
            assert f"  - {name}" in card

    def test_generate_model_card_hf_dataset_source_type_is_rendered(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        ctx = ModelCardContext(
            phase_metrics=[],
            datasets=["org/ds", "org/ds-eval"],
            dataset_source_type="huggingface",
        )
        card = retriever._generate_model_card(ctx)
        assert "| **Dataset source** | `huggingface` |" in card


class TestHuggingFaceRepoPrivacy:
    def test_ensure_hf_repo_ready_creates_repo_and_enforces_visibility(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()
        # Simulate missing repo so create_repo path is exercised
        import types as _types

        class _NotFound(Exception):
            def __init__(self):
                self.response = _types.SimpleNamespace(status_code=404)

        retriever.hf_api.repo_info.side_effect = _NotFound()

        result = retriever._ensure_hf_repo_ready()

        assert result.is_ok()
        retriever.hf_api.repo_info.assert_called_once_with(repo_id="org/test-model", repo_type="model")
        retriever.hf_api.create_repo.assert_called_once_with(
            repo_id="org/test-model",
            private=True,
            exist_ok=True,
            repo_type="model",
        )
        retriever.hf_api.update_repo_settings.assert_called_once_with(repo_id="org/test-model", private=True)

    def test_ensure_hf_repo_ready_existing_repo_does_not_require_create_permission(self, mock_config_with_hf, mock_secrets):
        """
        Some orgs disallow creating repos but allow pushing to an existing repo.
        We should not call create_repo when repo already exists.
        """
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()
        retriever.hf_api.repo_info.return_value = MagicMock()  # repo exists

        result = retriever._ensure_hf_repo_ready()

        assert result.is_ok()
        retriever.hf_api.repo_info.assert_called_once_with(repo_id="org/test-model", repo_type="model")
        retriever.hf_api.create_repo.assert_not_called()
        retriever.hf_api.update_repo_settings.assert_called_once_with(repo_id="org/test-model", private=True)

    def test_ensure_hf_repo_ready_does_not_fail_when_update_settings_forbidden(
        self, mock_config_with_hf, mock_secrets
    ):
        """
        Updating repo settings may require admin rights; upload should continue anyway.
        """
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()
        retriever.hf_api.repo_info.return_value = MagicMock()
        retriever.hf_api.update_repo_settings.side_effect = RuntimeError("forbidden")

        result = retriever._ensure_hf_repo_ready()

        assert result.is_ok()
        retriever.hf_api.create_repo.assert_not_called()

    def test_ensure_hf_repo_ready_returns_err_when_disabled(self, mock_secrets):
        cfg = MagicMock()
        cfg.get_active_provider_name.return_value = "single_node"
        cfg.get_provider_config.return_value = {"gpu_type": "NVIDIA_TEST", "mock_mode": False}
        cfg.experiment_tracking.huggingface = None
        cfg.model.name = "base"
        cfg.training.type = "qlora"
        cfg.training.hyperparams = PhaseHyperparametersConfig(epochs=1, per_device_train_batch_size=1)
        cfg.training.get_strategy_chain.return_value = []

        retriever = ModelRetriever(cfg, mock_secrets)
        retriever.hf_api = MagicMock()
        res = retriever._ensure_hf_repo_ready()
        assert res.is_failure()
        assert "disabled" in str(res.unwrap_err()).lower()

    def test_ensure_hf_repo_ready_returns_err_when_repo_id_missing(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()
        retriever.hf_repo_id = None
        res = retriever._ensure_hf_repo_ready()
        assert res.is_failure()
        assert "repo_id" in str(res.unwrap_err()).lower()

    def test_ensure_hf_repo_ready_handles_exception(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()
        retriever.hf_api.create_repo.side_effect = AttributeError("boom")
        # Force create_repo path (missing repo)
        import types as _types

        class _NotFound(Exception):
            def __init__(self):
                self.response = _types.SimpleNamespace(status_code=404)

        retriever.hf_api.repo_info.side_effect = _NotFound()
        res = retriever._ensure_hf_repo_ready()
        assert res.is_failure()
        assert "failed to prepare" in str(res.unwrap_err()).lower()


class TestModelRetrieverExecute:
    def test_execute_errors_when_no_ssh_info(self, mock_config_with_hf, mock_secrets) -> None:
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()

        res = retriever.execute(context={})
        assert res.is_failure()
        assert "No SSH connection info" in str(res.unwrap_err())

    def test_execute_mock_mode_short_circuits(
        self, mock_config_with_hf, mock_secrets, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # Isolate filesystem side-effects (prevents ./outputs in repo root)
        monkeypatch.chdir(tmp_path)

        # Enable mock mode at provider config level
        mock_config_with_hf.get_provider_config.return_value = {"mock_mode": True}

        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()

        # Avoid slow sleeps
        monkeypatch.setattr("time.sleep", lambda s: None)

        ctx = {"GPU Deployer": {"ssh_host": "pc", "provider_info": {"mock": True}, "resource_id": "x"}}
        res = retriever.execute(ctx)
        assert res.is_success()
        out = res.unwrap()
        stage_ctx = out["Model Retriever"]
        assert stage_ctx["mock"] is True
        assert stage_ctx["provider_name"] == "single_node"
        assert (tmp_path / "outputs/models/mock-model-checkpoint").exists()

    def test_execute_hf_upload_success_skips_download_and_calls_callbacks(
        self, mock_config_with_hf, mock_secrets, monkeypatch
    ):
        events: list[str] = []

        callbacks = MagicMock()
        callbacks.on_hf_upload_started = lambda repo_id: events.append(f"start:{repo_id}")
        callbacks.on_hf_upload_completed = lambda repo_id, dur: events.append(f"done:{repo_id}")
        callbacks.on_retrieval_completed = lambda hf, path: events.append(f"final:{hf}:{path}")

        retriever = ModelRetriever(mock_config_with_hf, mock_secrets, callbacks=callbacks)
        retriever.hf_api = MagicMock()

        monkeypatch.setattr(retriever, "_get_model_size", lambda: Ok(10.0))
        monkeypatch.setattr(retriever, "_upload_to_hf_from_remote", lambda context=None: Ok(None))
        monkeypatch.setattr(
            retriever, "_download_model", lambda: (_ for _ in ()).throw(AssertionError("should not download"))
        )

        # deterministic duration
        times = iter([100.0, 101.5])
        monkeypatch.setattr("time.time", lambda: next(times))

        # Avoid real SSH construction
        monkeypatch.setattr("src.pipeline.stages.model_retriever.SSHClient", MagicMock())

        ctx = {"GPU Deployer": {"ssh_host": "1.2.3.4", "ssh_port": 22, "ssh_user": "root", "workspace_path": "/w"}}
        res = retriever.execute(ctx)
        assert res.is_success()
        out = res.unwrap()["Model Retriever"]
        assert out["hf_uploaded"] is True
        assert out["hf_repo_id"] == "org/test-model"
        assert "upload_duration_seconds" in out
        assert events[:2] == ["start:org/test-model", "done:org/test-model"]

    def test_execute_hf_upload_fails_then_downloads_small_model(
        self, mock_config_with_hf, mock_secrets, monkeypatch, tmp_path
    ):
        events: list[str] = []
        callbacks = MagicMock()
        callbacks.on_local_download_started = lambda size: events.append(f"dl_start:{size}")
        callbacks.on_local_download_completed = lambda path: events.append(f"dl_done:{path}")

        retriever = ModelRetriever(mock_config_with_hf, mock_secrets, callbacks=callbacks)
        retriever.hf_api = MagicMock()

        monkeypatch.setattr(retriever, "_get_model_size", lambda: Ok(100.0))
        monkeypatch.setattr(retriever, "_upload_to_hf_from_remote", lambda context=None: Err("fail"))
        monkeypatch.setattr(retriever, "_download_model", lambda: Ok(tmp_path))
        monkeypatch.setattr("src.pipeline.stages.model_retriever.SSHClient", MagicMock())

        ctx = {"GPU Deployer": {"ssh_host": "1.2.3.4", "ssh_port": 22, "ssh_user": "root", "workspace_path": "/w"}}
        res = retriever.execute(ctx)
        assert res.is_success()
        out = res.unwrap()["Model Retriever"]
        assert out["hf_uploaded"] is False
        assert out["local_model_path"] == str(tmp_path)
        assert events and events[0].startswith("dl_start:")

    def test_execute_download_failure_calls_failed_callback_and_returns_err(
        self, mock_config_with_hf, mock_secrets, monkeypatch
    ) -> None:
        events: list[str] = []
        callbacks = MagicMock()
        callbacks.on_local_download_failed = lambda msg: events.append(f"dl_fail:{msg}")

        retriever = ModelRetriever(mock_config_with_hf, mock_secrets, callbacks=callbacks)
        retriever.hf_api = MagicMock()

        monkeypatch.setattr(retriever, "_get_model_size", lambda: Ok(100.0))
        monkeypatch.setattr(retriever, "_upload_to_hf_from_remote", lambda context=None: Err("hf fail"))
        monkeypatch.setattr(retriever, "_download_model", lambda: Err("dl fail"))
        monkeypatch.setattr("src.pipeline.stages.model_retriever.SSHClient", MagicMock())

        ctx = {"GPU Deployer": {"ssh_host": "1.2.3.4", "ssh_port": 22, "ssh_user": "root", "workspace_path": "/w"}}
        res = retriever.execute(ctx)
        assert res.is_failure()
        assert "local download failed" in str(res.unwrap_err()).lower()
        assert events and events[0].startswith("dl_fail:")

    def test_execute_hf_disabled_downloads(self, mock_config_with_hf, mock_secrets, monkeypatch, tmp_path) -> None:
        # Make HF "disabled" but keep repo_id present to hit the `elif not self.hf_enabled` branch
        mock_config_with_hf.experiment_tracking.huggingface = MagicMock(
            enabled=False, repo_id="org/test-model", private=True
        )
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()

        monkeypatch.setattr(retriever, "_get_model_size", lambda: Ok(10.0))
        monkeypatch.setattr(retriever, "_download_model", lambda: Ok(tmp_path))
        monkeypatch.setattr("src.pipeline.stages.model_retriever.SSHClient", MagicMock())

        ctx = {"GPU Deployer": {"ssh_host": "1.2.3.4", "ssh_port": 22, "ssh_user": "root", "workspace_path": "/w"}}
        res = retriever.execute(ctx)
        assert res.is_success()
        out = res.unwrap()["Model Retriever"]
        assert out["hf_uploaded"] is False
        assert out["local_model_path"] == str(tmp_path)

    def test_execute_hf_repo_id_missing_skips_upload_and_downloads(
        self, mock_config_with_hf, mock_secrets, monkeypatch, tmp_path
    ) -> None:
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()
        retriever.hf_enabled = True
        retriever.hf_repo_id = None

        monkeypatch.setattr(retriever, "_get_model_size", lambda: Ok(10.0))
        monkeypatch.setattr(retriever, "_download_model", lambda: Ok(tmp_path))
        monkeypatch.setattr("src.pipeline.stages.model_retriever.SSHClient", MagicMock())

        ctx = {"GPU Deployer": {"ssh_host": "1.2.3.4", "ssh_port": 22, "ssh_user": "root", "workspace_path": "/w"}}
        res = retriever.execute(ctx)
        assert res.is_success()
        out = res.unwrap()["Model Retriever"]
        assert out["hf_uploaded"] is False
        assert out["local_model_path"] == str(tmp_path)

    def test_execute_hf_upload_fails_and_model_too_large_errors(self, mock_config_with_hf, mock_secrets, monkeypatch):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()

        monkeypatch.setattr(retriever, "_get_model_size", lambda: Ok(2048.0))
        monkeypatch.setattr(retriever, "_upload_to_hf_from_remote", lambda context=None: Err("fail"))
        monkeypatch.setattr("src.pipeline.stages.model_retriever.SSHClient", MagicMock())

        ctx = {"GPU Deployer": {"ssh_host": "1.2.3.4", "ssh_port": 22, "ssh_user": "root", "workspace_path": "/w"}}
        res = retriever.execute(ctx)
        assert res.is_failure()
        assert "too large" in str(res.unwrap_err()).lower()


class _SSHStub:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def exec_command(self, *, command: str, background: bool = False, timeout: int | None = None, **kwargs):
        self.commands.append(command)

        if command.startswith("find ") and "checkpoint-final" in command:
            return True, "/w/output/checkpoint-final\n", ""
        if command.startswith("find ") and "checkpoint-*" in command:
            return True, "/w/output/checkpoint-9\n", ""
        if command.startswith("ls -la "):
            return True, "ok", ""
        if "base64 -d >" in command:
            return True, "", ""
        if "huggingface-cli upload" in command:
            return True, "", ""
        return True, "", ""


class TestModelRetrieverInternalRemoteUpload:
    def test_upload_to_hf_from_remote_success(self, mock_config_with_hf, mock_secrets, monkeypatch) -> None:
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()
        retriever._workspace_path = "/w"
        retriever._ssh_client = _SSHStub()

        # Keep model card small to avoid huge base64 cmd in debugging
        monkeypatch.setattr(retriever, "_generate_model_card", lambda ctx=None: "CARD")
        monkeypatch.setattr(retriever, "_ensure_hf_repo_ready", lambda: Ok(None))

        res = retriever._upload_to_hf_from_remote()
        assert res.is_success()
        assert any("huggingface-cli upload" in c for c in retriever._ssh_client.commands)

    def test_upload_to_hf_from_remote_returns_err_on_upload_cmd_failure(
        self, mock_config_with_hf, mock_secrets, monkeypatch
    ):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()
        retriever._workspace_path = "/w"

        class _FailUploadSSH(_SSHStub):
            def exec_command(self, *, command: str, background: bool = False, timeout: int | None = None, **kwargs):
                if "huggingface-cli upload" in command:
                    return False, "", "boom"
                return super().exec_command(command=command, background=background, timeout=timeout, **kwargs)

        retriever._ssh_client = _FailUploadSSH()
        monkeypatch.setattr(retriever, "_generate_model_card", lambda ctx=None: "CARD")
        monkeypatch.setattr(retriever, "_ensure_hf_repo_ready", lambda: Ok(None))

        res = retriever._upload_to_hf_from_remote()
        assert res.is_failure()
        assert "upload command failed" in str(res.unwrap_err()).lower()

    def test_upload_to_hf_from_remote_returns_err_when_ssh_missing(
        self, mock_config_with_hf, mock_secrets, monkeypatch
    ):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()
        retriever._ssh_client = None
        monkeypatch.setattr(retriever, "_ensure_hf_repo_ready", lambda: Ok(None))
        res = retriever._upload_to_hf_from_remote()
        assert res.is_failure()
        assert "ssh client" in str(res.unwrap_err()).lower()

    def test_upload_to_hf_from_remote_propagates_repo_prepare_error(
        self, mock_config_with_hf, mock_secrets, monkeypatch
    ):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever._ssh_client = _SSHStub()
        monkeypatch.setattr(retriever, "_ensure_hf_repo_ready", lambda: Err("nope"))
        res = retriever._upload_to_hf_from_remote()
        assert res.is_failure()
        assert "nope" in str(res.unwrap_err())


class TestModelRetrieverModelSizeAndDownload:
    def test_get_model_size_returns_err_when_no_ssh(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever._ssh_client = None
        res = retriever._get_model_size()
        assert res.is_failure()
        assert "ssh client" in str(res.unwrap_err()).lower()

    def test_get_model_size_success_and_failures(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever._workspace_path = "/w"

        ssh = MagicMock()
        # success: 1MB in bytes
        ssh.exec_command.return_value = (True, "1048576\n", "")
        retriever._ssh_client = ssh
        ok = retriever._get_model_size()
        assert ok.is_success()
        assert ok.unwrap() == pytest.approx(1.0, rel=1e-6)

        # command failure
        ssh.exec_command.return_value = (False, "", "err")
        bad = retriever._get_model_size()
        assert bad.is_failure()
        assert "size command failed" in str(bad.unwrap_err()).lower()

        # parsing exception
        ssh.exec_command.return_value = (True, "not-int\n", "")
        boom = retriever._get_model_size()
        assert boom.is_failure()
        assert "failed to get model size" in str(boom.unwrap_err()).lower()

    def test_download_model_download_directory_failure(self, mock_config_with_hf, mock_secrets, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever._workspace_path = "/w"

        ssh = MagicMock()
        ssh.exec_command.return_value = (True, "/w/output/checkpoint-final\n", "")
        ssh.download_directory.return_value = Err("x")
        retriever._ssh_client = ssh

        res = retriever._download_model()
        assert res.is_failure()
        assert "failed to download model" in str(res.unwrap_err()).lower()

    def test_download_model_ok(self, mock_config_with_hf, mock_secrets, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("time.time", lambda: 100.0)

        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever._workspace_path = "/w"

        ssh = MagicMock()
        ssh.exec_command.return_value = (True, "/w/output/checkpoint-final\n", "")
        ssh.download_directory.return_value = Ok(None)
        retriever._ssh_client = ssh

        res = retriever._download_model()
        assert res.is_success()
        p = res.unwrap()
        assert p.exists()
        assert "models" in str(p)


# =============================================================================
# NEW COVERAGE TESTS
# =============================================================================


class TestModelRetrieverInit:
    def test_init_get_provider_training_config_raises_falls_back_to_provider_config(self, mock_secrets):
        cfg = MagicMock()
        cfg.get_active_provider_name.return_value = "single_node"
        cfg.get_provider_config.return_value = {"gpu_type": "A100", "mock_mode": False}
        cfg.get_provider_training_config.side_effect = RuntimeError("unavailable")
        cfg.experiment_tracking.huggingface = HuggingFaceHubConfig(enabled=True, repo_id="org/m", private=True)
        cfg.model.name = "base"
        cfg.training.type = "qlora"
        cfg.training.hyperparams = PhaseHyperparametersConfig(epochs=1, per_device_train_batch_size=1)
        cfg.training.get_strategy_chain.return_value = []
        cfg.get_adapter_config.side_effect = ValueError("n/a")

        retriever = ModelRetriever(cfg, mock_secrets)
        # Falls back to provider config when get_provider_training_config raises
        assert retriever._provider_training_cfg == {"gpu_type": "A100", "mock_mode": False}

    def test_init_get_provider_training_config_non_dict_falls_back(self, mock_secrets):
        cfg = MagicMock()
        cfg.get_active_provider_name.return_value = "single_node"
        cfg.get_provider_config.return_value = {"gpu_type": "A100", "mock_mode": False}
        cfg.get_provider_training_config.return_value = "not_a_dict"  # non-dict return
        cfg.experiment_tracking.huggingface = HuggingFaceHubConfig(enabled=True, repo_id="org/m", private=True)
        cfg.model.name = "base"
        cfg.training.type = "qlora"
        cfg.training.hyperparams = PhaseHyperparametersConfig(epochs=1, per_device_train_batch_size=1)
        cfg.training.get_strategy_chain.return_value = []
        cfg.get_adapter_config.side_effect = ValueError("n/a")

        retriever = ModelRetriever(cfg, mock_secrets)
        assert retriever._provider_training_cfg == {"gpu_type": "A100", "mock_mode": False}


class TestBasename:
    def test_basename_empty_string_returns_empty(self):
        assert ModelRetriever._basename("") == ""

    def test_basename_whitespace_only_returns_empty(self):
        assert ModelRetriever._basename("   ") == ""


class TestExtractPhaseMetrics:
    def test_from_context_phase_metrics_key(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        ctx = {"phase_metrics": [{"phase_idx": 0, "train_loss": 1.5}]}
        result = retriever._extract_phase_metrics(context=ctx, remote_output_dir="/w/output")
        assert result.phase_metrics == [{"phase_idx": 0, "train_loss": 1.5}]
        assert result.training_started_at is None
        assert result.training_completed_at is None

    def test_from_context_training_phase_metrics_key(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        ctx = {"training_phase_metrics": [{"phase_idx": 1, "eval_loss": 0.8}]}
        result = retriever._extract_phase_metrics(context=ctx, remote_output_dir="/w/output")
        assert result.phase_metrics == [{"phase_idx": 1, "eval_loss": 0.8}]

    def test_from_context_ignores_non_list(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever._ssh_client = None
        ctx = {"phase_metrics": "not_a_list"}
        result = retriever._extract_phase_metrics(context=ctx, remote_output_dir="/w/output")
        assert result.phase_metrics == []

    def test_from_ssh_valid_json_full_phases(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        state = {
            "started_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T01:00:00",
            "phases": [
                {
                    "phase_idx": 1,
                    "strategy_type": "sft",
                    "status": "completed",
                    "started_at": "2024-01-01T00:00:00",
                    "completed_at": "2024-01-01T00:30:00",
                    "metrics": {"train_loss": 0.5, "global_step": 100},
                },
                {
                    "phase_idx": 0,
                    "strategy_type": "cpt",
                    "status": "completed",
                    "metrics": {"train_loss": 1.2},
                },
            ],
        }
        ssh = MagicMock()
        ssh.exec_command.return_value = (True, json.dumps(state), "")
        retriever._ssh_client = ssh

        result = retriever._extract_phase_metrics(context={}, remote_output_dir="/w/output")
        assert len(result.phase_metrics) == 2
        # Should be sorted by phase_idx
        assert result.phase_metrics[0]["phase_idx"] == 0
        assert result.phase_metrics[1]["train_loss"] == 0.5
        assert result.training_started_at == "2024-01-01T00:00:00"
        assert result.training_completed_at == "2024-01-01T01:00:00"

    def test_from_ssh_invalid_json_returns_empty(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        ssh = MagicMock()
        ssh.exec_command.return_value = (True, "NOT VALID JSON {{{", "")
        retriever._ssh_client = ssh

        result = retriever._extract_phase_metrics(context={}, remote_output_dir="/w/output")
        assert result.phase_metrics == []

    def test_from_ssh_empty_stdout_returns_empty(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        ssh = MagicMock()
        ssh.exec_command.return_value = (True, "   ", "")
        retriever._ssh_client = ssh

        result = retriever._extract_phase_metrics(context={}, remote_output_dir="/w/output")
        assert result.phase_metrics == []

    def test_from_ssh_command_fails_returns_empty(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        ssh = MagicMock()
        ssh.exec_command.return_value = (False, "", "error")
        retriever._ssh_client = ssh

        result = retriever._extract_phase_metrics(context={}, remote_output_dir="/w/output")
        assert result.phase_metrics == []

    def test_from_ssh_json_without_phases_key(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        state = {"started_at": "2024-01-01T00:00:00", "completed_at": "2024-01-01T01:00:00"}
        ssh = MagicMock()
        ssh.exec_command.return_value = (True, json.dumps(state), "")
        retriever._ssh_client = ssh

        result = retriever._extract_phase_metrics(context={}, remote_output_dir="/w/output")
        assert result.phase_metrics == []
        assert result.training_started_at == "2024-01-01T00:00:00"

    def test_no_ssh_client_returns_empty(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever._ssh_client = None

        result = retriever._extract_phase_metrics(context={}, remote_output_dir="/w/output")
        assert result.phase_metrics == []
        assert result.training_started_at is None


class TestResolveCheckpoint:
    def test_no_ssh_returns_output_dir(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever._ssh_client = None
        assert retriever._resolve_checkpoint("/w/output") == "/w/output"

    def test_finds_checkpoint_final(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        ssh = MagicMock()
        ssh.exec_command.side_effect = [
            (True, "/w/output/checkpoint-final\n", ""),
        ]
        retriever._ssh_client = ssh
        result = retriever._resolve_checkpoint("/w/output")
        assert result == "/w/output/checkpoint-final"

    def test_fallback_to_latest_checkpoint_version(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)

        class _NoFinalSSH:
            def exec_command(self, *, command, background=False, timeout=None, **kwargs):
                if "checkpoint-final" in command:
                    return True, "", ""  # nothing found
                if "checkpoint-*" in command:
                    return True, "/w/output/checkpoint-9\n", ""
                return True, "", ""

        retriever._ssh_client = _NoFinalSSH()
        result = retriever._resolve_checkpoint("/w/output")
        assert result == "/w/output/checkpoint-9"

    def test_fallback_to_output_dir_when_no_checkpoints(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)

        class _NoneSSH:
            def exec_command(self, *, command, background=False, timeout=None, **kwargs):
                return True, "", ""  # nothing found for both find commands

        retriever._ssh_client = _NoneSSH()
        result = retriever._resolve_checkpoint("/w/output")
        assert result == "/w/output"


class TestEnsureHfRepoReadyEdgeCases:
    def test_non_404_exception_from_repo_info_returns_err(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()

        class _ServerError(Exception):
            def __init__(self):
                self.response = types.SimpleNamespace(status_code=500)

        retriever.hf_api.repo_info.side_effect = _ServerError()
        res = retriever._ensure_hf_repo_ready()
        assert res.is_failure()
        assert "failed to prepare" in str(res.unwrap_err()).lower()

    def test_401_in_update_repo_settings_logs_hint_and_continues(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()
        retriever.hf_api.repo_info.return_value = MagicMock()  # repo exists

        class _Unauthorized(Exception):
            def __init__(self):
                self.response = types.SimpleNamespace(status_code=401)

        retriever.hf_api.update_repo_settings.side_effect = _Unauthorized()
        res = retriever._ensure_hf_repo_ready()
        # Must not fail – just log warning
        assert res.is_ok()

    def test_401_in_repo_prepare_includes_whoami_in_error(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()

        class _Unauthorized(Exception):
            def __init__(self):
                self.response = types.SimpleNamespace(status_code=401)

        # repo_info raises 401 → not 404 → re-raises → outer except catches
        retriever.hf_api.repo_info.side_effect = _Unauthorized()
        retriever.hf_api.whoami.return_value = {"name": "user1", "orgs": ["org1"]}

        res = retriever._ensure_hf_repo_ready()
        assert res.is_failure()
        err_msg = str(res.unwrap_err())
        assert "401" in err_msg or "unauthorized" in err_msg.lower()

    def test_403_in_repo_prepare_includes_forbidden_hint(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()

        class _Forbidden(Exception):
            def __init__(self):
                self.response = types.SimpleNamespace(status_code=403)

        retriever.hf_api.repo_info.side_effect = _Forbidden()
        retriever.hf_api.whoami.return_value = {"name": "user1", "orgs": []}

        res = retriever._ensure_hf_repo_ready()
        assert res.is_failure()
        err_msg = str(res.unwrap_err())
        assert "403" in err_msg or "forbidden" in err_msg.lower()

    def test_whoami_raises_is_handled_gracefully(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()

        class _Unauthorized(Exception):
            def __init__(self):
                self.response = types.SimpleNamespace(status_code=401)

        retriever.hf_api.repo_info.side_effect = _Unauthorized()
        retriever.hf_api.whoami.side_effect = RuntimeError("whoami failed")

        res = retriever._ensure_hf_repo_ready()
        assert res.is_failure()
        assert "whoami_check_failed" in str(res.unwrap_err())


class TestSha12:
    def test_sha12_returns_12_char_hex(self):
        result = ModelRetriever._sha12("hello world")
        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)

    def test_sha12_is_deterministic(self):
        assert ModelRetriever._sha12("abc") == ModelRetriever._sha12("abc")

    def test_sha12_differs_for_different_inputs(self):
        assert ModelRetriever._sha12("abc") != ModelRetriever._sha12("xyz")


class TestExecuteCloseMasterException:
    def test_close_master_exception_does_not_propagate(self, mock_config_with_hf, mock_secrets, monkeypatch):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()

        monkeypatch.setattr(retriever, "_execute_retrieval", lambda ctx: Ok(ctx))

        class _BoomSSH:
            def close_master(self):
                raise RuntimeError("close failed")

        monkeypatch.setattr(
            "src.pipeline.stages.model_retriever.SSHClient", MagicMock(return_value=_BoomSSH())
        )

        ctx = {"GPU Deployer": {"ssh_host": "1.2.3.4"}}
        res = retriever.execute(ctx)
        assert res.is_success()


class TestExecuteRetrievalTypeErrorFallback:
    def test_upload_typeerror_fallback_calls_without_context(
        self, mock_config_with_hf, mock_secrets, monkeypatch
    ):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()

        calls: list[dict] = []

        def upload_mock(**kwargs):
            calls.append(dict(kwargs))
            if "context" in kwargs:
                raise TypeError("unexpected keyword argument 'context'")
            return Ok(None)

        monkeypatch.setattr(retriever, "_get_model_size", lambda: Ok(10.0))
        monkeypatch.setattr(retriever, "_upload_to_hf_from_remote", upload_mock)
        monkeypatch.setattr("src.pipeline.stages.model_retriever.SSHClient", MagicMock())

        ctx = {"GPU Deployer": {"ssh_host": "1.2.3.4", "ssh_port": 22, "workspace_path": "/w"}}
        res = retriever.execute(ctx)
        assert res.is_success()
        # First call had context kwarg, second did not
        assert len(calls) == 2
        assert "context" in calls[0]
        assert "context" not in calls[1]


class TestUploadToHfFromRemoteEdgeCases:
    def test_readme_creation_failure_logs_warning_but_continues(
        self, mock_config_with_hf, mock_secrets, monkeypatch
    ):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()
        retriever._workspace_path = "/w"

        class _FailReadmeSSH(_SSHStub):
            def exec_command(self, *, command, background=False, timeout=None, **kwargs):
                if "base64 -d >" in command:
                    return False, "", "base64 error"  # README creation fails
                return super().exec_command(command=command, background=background, timeout=timeout)

        retriever._ssh_client = _FailReadmeSSH()
        monkeypatch.setattr(retriever, "_generate_model_card", lambda ctx=None: "CARD")
        monkeypatch.setattr(retriever, "_ensure_hf_repo_ready", lambda: Ok(None))

        # Should still succeed – README failure is non-fatal
        res = retriever._upload_to_hf_from_remote()
        assert res.is_success()

    def test_unexpected_exception_returns_err(self, mock_config_with_hf, mock_secrets, monkeypatch):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        retriever.hf_api = MagicMock()
        retriever._workspace_path = "/w"
        retriever._ssh_client = _SSHStub()

        def _raise(*args, **kwargs):
            raise RuntimeError("boom from inside")

        monkeypatch.setattr(retriever, "_ensure_hf_repo_ready", _raise)

        res = retriever._upload_to_hf_from_remote()
        assert res.is_failure()
        assert "direct upload failed" in str(res.unwrap_err()).lower()


class TestGenerateModelCardPhaseMetrics:
    def test_phase_metrics_rendered_in_table(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        ctx = ModelCardContext(
            phase_metrics=[
                {
                    "phase_idx": 0,
                    "strategy_type": "sft",
                    "status": "completed",
                    "train_loss": 1.2345,
                    "eval_loss": None,
                    "global_step": 100,
                    "epoch": 3.0,
                    "train_runtime": 120.5,
                    "peak_memory_gb": 8.0,
                }
            ],
            datasets=[],
            training_started_at="2024-01-01T00:00:00",
            training_completed_at="2024-01-01T01:00:00",
        )
        card = retriever._generate_model_card(ctx)
        assert "1.2345" in card  # train_loss formatted to 4 digits
        assert "—" in card  # None eval_loss renders as em-dash
        assert "3.00" in card  # epoch to 2 digits
        assert "### Run timeline" in card
        assert "Started at" in card
        assert "Completed at" in card
        # Table header present
        assert "| Phase |" in card

    def test_no_phase_metrics_shows_placeholder(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        ctx = ModelCardContext(phase_metrics=[], datasets=[])
        card = retriever._generate_model_card(ctx)
        assert "No per-phase metrics were found" in card

    def test_fmt_bool_values_in_phase_metrics(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        ctx = ModelCardContext(
            phase_metrics=[
                {
                    "phase_idx": 0,
                    "strategy_type": "sft",
                    "status": True,  # bool → "true"
                }
            ],
            datasets=[],
        )
        card = retriever._generate_model_card(ctx)
        assert "true" in card

    def test_fmt_int_values_in_phase_metrics(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        ctx = ModelCardContext(
            phase_metrics=[
                {
                    "phase_idx": 0,
                    "strategy_type": "sft",
                    "global_step": 42,  # int → "42"
                }
            ],
            datasets=[],
        )
        card = retriever._generate_model_card(ctx)
        assert "42" in card

    def test_only_started_at_no_completed(self, mock_config_with_hf, mock_secrets):
        retriever = ModelRetriever(mock_config_with_hf, mock_secrets)
        ctx = ModelCardContext(
            phase_metrics=[],
            datasets=[],
            training_started_at="2024-01-01T00:00:00",
            training_completed_at=None,
        )
        card = retriever._generate_model_card(ctx)
        assert "Started at" in card
        assert "Completed at" not in card


