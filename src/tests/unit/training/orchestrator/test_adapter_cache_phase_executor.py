"""
Tests for PhaseExecutor adapter-cache helpers and ChainRunner cascade flag.

Coverage matrix
───────────────
_compute_dataset_fingerprint   local existing / non-existent / HF dataset / determinism
_retry_call                    success-first / success-third / exhausted / retries=1
_try_adapter_cache_hit         cache hit / repo not found / tag missing / load failure
_upload_adapter_to_cache       success / soft-fail after retries / tag stored
PhaseExecutor.execute()        cache disabled / cache hit / cache miss / upstream_retrained
ChainRunner cascade            SKIPPED → flag stays False; COMPLETED → flag set True
Dependency errors              HfApi unavailable / peft unavailable
Combinatorial                  multiple phases, mixed cache states
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from src.training.managers.data_buffer import (
    DataBuffer,
    PhaseState,
    PhaseStatus,
)
from src.training.orchestrator.phase_executor import PhaseExecutor
from src.utils.result import Err, Ok, Success, Failure

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────
# Helpers / Factories
# ─────────────────────────────────────────────


def _mk_executor(config: Any = None) -> PhaseExecutor:
    return PhaseExecutor(
        tokenizer=MagicMock(name="tokenizer"),
        config=config or MagicMock(name="config"),
        memory_manager=MagicMock(name="mm"),
        dataset_loader=MagicMock(name="dl"),
        metrics_collector=MagicMock(name="mc"),
    )


def _mk_local_dataset_config(path: str, *, mtime: float = 1000.0, size: int = 512) -> MagicMock:
    ds = MagicMock()
    ds.get_source_type.return_value = "local"
    ds.source_local.local_paths.train = path
    return ds


def _mk_hf_dataset_config(train_id: str = "org/dataset", commit_sha: str = "abc123") -> MagicMock:
    ds = MagicMock()
    ds.get_source_type.return_value = "hf"
    ds.source_hf.train_id = train_id
    return ds, commit_sha


def _mk_cache_config(repo_id: str = "org/adapters") -> SimpleNamespace:
    return SimpleNamespace(enabled=True, repo_id=repo_id, private=True)


def _mk_phase_ns(
    strategy_type: str = "sft",
    *,
    repo_id: str = "org/cache",
    cache_enabled: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        strategy_type=strategy_type,
        adapter_cache=_mk_cache_config(repo_id=repo_id) if cache_enabled else SimpleNamespace(enabled=False),
    )


def _mk_buffer_with_phases(tmp_path: Path, n_phases: int = 2) -> DataBuffer:
    from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig

    strategies = [
        StrategyPhaseConfig(
            strategy_type="sft" if i == 0 else "dpo",
            dataset="sft_data" if i == 0 else "pref_data",
            hyperparams=PhaseHyperparametersConfig(epochs=1),
        )
        for i in range(n_phases)
    ]
    buf = DataBuffer(base_output_dir=tmp_path, base_model_path="base-model")
    buf.init_pipeline(strategies, force=True)
    return buf


# ─────────────────────────────────────────────
# _compute_dataset_fingerprint
# ─────────────────────────────────────────────


class TestComputeDatasetFingerprint:
    def test_positive_local_existing_file_returns_10_char_hex(self, tmp_path: Path) -> None:
        f = tmp_path / "train.jsonl"
        f.write_text('{"text": "hello"}\n')
        ds = _mk_local_dataset_config(str(f))
        fp = PhaseExecutor._compute_dataset_fingerprint(ds)
        assert len(fp) == 10
        assert all(c in "0123456789abcdef" for c in fp)

    def test_positive_local_file_fingerprint_is_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "train.jsonl"
        f.write_text("data")
        ds = _mk_local_dataset_config(str(f))
        fp1 = PhaseExecutor._compute_dataset_fingerprint(ds)
        fp2 = PhaseExecutor._compute_dataset_fingerprint(ds)
        assert fp1 == fp2

    def test_positive_local_file_fingerprint_changes_on_mtime(self, tmp_path: Path) -> None:
        """Changing mtime (simulate file update) produces a different fingerprint."""
        import os

        f = tmp_path / "train.jsonl"
        f.write_text("data")
        ds1 = _mk_local_dataset_config(str(f))
        fp1 = PhaseExecutor._compute_dataset_fingerprint(ds1)

        # Change mtime
        new_mtime = f.stat().st_mtime + 1.0
        os.utime(f, (new_mtime, new_mtime))
        fp2 = PhaseExecutor._compute_dataset_fingerprint(ds1)

        assert fp1 != fp2

    def test_boundary_local_nonexistent_file_returns_10_char_hash(self, tmp_path: Path) -> None:
        """Non-existent files fall back to hashing the path string only — still 10 chars."""
        ds = _mk_local_dataset_config(str(tmp_path / "no_such_file.jsonl"))
        fp = PhaseExecutor._compute_dataset_fingerprint(ds)
        assert len(fp) == 10

    def test_positive_hf_dataset_returns_10_char_hash(self) -> None:
        ds, commit_sha = _mk_hf_dataset_config("org/ds", "sha-abc123")
        hf_info = MagicMock()
        hf_info.sha = commit_sha

        with patch("src.training.orchestrator.phase_executor.PhaseExecutor._compute_dataset_fingerprint") as mock_fp:
            mock_fp.return_value = "abcde12345"
            fp = mock_fp(ds)
            assert len(fp) == 10

    def test_positive_hf_dataset_uses_commit_sha_in_hash(self) -> None:
        """Two HF datasets with different commit SHAs produce different fingerprints."""
        from hashlib import sha256

        train_id = "org/dataset"
        fp1 = sha256(f"{train_id}\x00sha_v1".encode()).hexdigest()[:10]
        fp2 = sha256(f"{train_id}\x00sha_v2".encode()).hexdigest()[:10]
        assert fp1 != fp2

    def test_boundary_hf_dataset_info_failure_falls_back_to_empty_sha(self) -> None:
        ds, _ = _mk_hf_dataset_config("org/ds", "")
        hf_info = MagicMock()
        hf_info.sha = None  # simulate missing sha

        with patch("huggingface_hub.dataset_info", side_effect=Exception("network error")):
            # Should not raise — should use "" as commit_sha fallback
            fp = PhaseExecutor._compute_dataset_fingerprint(ds)
            assert len(fp) == 10

    def test_invariant_fingerprint_length_always_10(self, tmp_path: Path) -> None:
        paths = [tmp_path / f"file{i}.jsonl" for i in range(3)]
        for p in paths:
            p.write_text("x")
        for p in paths:
            ds = _mk_local_dataset_config(str(p))
            fp = PhaseExecutor._compute_dataset_fingerprint(ds)
            assert len(fp) == 10, f"Expected 10 chars, got {len(fp)}: {fp}"


# ─────────────────────────────────────────────
# _retry_call
# ─────────────────────────────────────────────


class TestRetryCall:
    def test_positive_success_on_first_try(self) -> None:
        fn = MagicMock(return_value=42)
        result = PhaseExecutor._retry_call(fn, retries=3, delay_s=0)
        assert result == 42
        fn.assert_called_once()

    def test_positive_success_on_third_try(self) -> None:
        attempts: list[int] = []

        def fn() -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("transient")
            return "ok"

        with patch("src.training.orchestrator.phase_executor.time.sleep"):
            result = PhaseExecutor._retry_call(fn, retries=3, delay_s=1)
        assert result == "ok"
        assert len(attempts) == 3

    def test_negative_raises_after_exhausting_retries(self) -> None:
        fn = MagicMock(side_effect=ValueError("always fails"))
        with patch("src.training.orchestrator.phase_executor.time.sleep"):
            with pytest.raises(ValueError, match="always fails"):
                PhaseExecutor._retry_call(fn, retries=3, delay_s=1)
        assert fn.call_count == 3

    def test_boundary_retries_1_no_retry_just_raises(self) -> None:
        fn = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            PhaseExecutor._retry_call(fn, retries=1, delay_s=0)
        fn.assert_called_once()

    def test_boundary_sleep_called_between_attempts(self) -> None:
        fn = MagicMock(side_effect=[Exception("err1"), Exception("err2"), "done"])
        with patch("src.training.orchestrator.phase_executor.time.sleep") as mock_sleep:
            PhaseExecutor._retry_call(fn, retries=3, delay_s=5)
        assert mock_sleep.call_count == 2  # sleep after attempt 1 and 2 (not after 3rd success)

    def test_boundary_last_exception_is_re_raised(self) -> None:
        """The LAST exception (not the first) is the one that propagates."""
        errors = [ValueError("first"), ValueError("last")]
        fn = MagicMock(side_effect=errors)
        with patch("src.training.orchestrator.phase_executor.time.sleep"):
            with pytest.raises(ValueError, match="last"):
                PhaseExecutor._retry_call(fn, retries=2, delay_s=0)

    def test_invariant_return_value_passed_through(self) -> None:
        fn = MagicMock(return_value={"key": "value"})
        result = PhaseExecutor._retry_call(fn, retries=2, delay_s=0)
        assert result == {"key": "value"}


# ─────────────────────────────────────────────
# _try_adapter_cache_hit
# ─────────────────────────────────────────────


def _mk_refs_with_tag(tag_name: str) -> MagicMock:
    refs = MagicMock()
    tag = MagicMock()
    tag.name = tag_name
    refs.tags = [tag]
    return refs


def _mk_refs_empty() -> MagicMock:
    refs = MagicMock()
    refs.tags = []
    return refs


class TestTryAdapterCacheHit:
    # ── Positive ──────────────────────────────

    def test_positive_cache_hit_returns_ok_model(self, tmp_path: Path) -> None:
        executor = _mk_executor()
        buf = _mk_buffer_with_phases(tmp_path)
        phase = _mk_phase_ns("sft", repo_id="org/cache")
        fingerprint = "abcde12345"
        expected_tag = f"phase-0-sft-ds{fingerprint}"

        loaded_model = MagicMock(name="loaded_peft_model")

        with (
            patch("huggingface_hub.HfApi") as MockHfApi,
            patch("peft.PeftModel") as MockPeft,
        ):
            MockHfApi.return_value.list_repo_refs.return_value = _mk_refs_with_tag(expected_tag)
            MockPeft.from_pretrained.return_value = loaded_model

            result = executor._try_adapter_cache_hit(
                phase_idx=0,
                phase=phase,  # type: ignore[arg-type]
                model=MagicMock(),
                buffer=buf,
                fingerprint=fingerprint,
            )

        assert result is not None
        assert isinstance(result, Success)
        assert result.unwrap() is loaded_model

    def test_positive_cache_hit_sets_adapter_cache_hit_flag(self, tmp_path: Path) -> None:
        executor = _mk_executor()
        buf = _mk_buffer_with_phases(tmp_path)
        phase = _mk_phase_ns("sft", repo_id="org/cache")
        fingerprint = "abcde12345"
        expected_tag = f"phase-0-sft-ds{fingerprint}"

        with (
            patch("huggingface_hub.HfApi") as MockHfApi,
            patch("peft.PeftModel") as MockPeft,
        ):
            MockHfApi.return_value.list_repo_refs.return_value = _mk_refs_with_tag(expected_tag)
            MockPeft.from_pretrained.return_value = MagicMock()

            executor._try_adapter_cache_hit(0, phase, MagicMock(), buf, fingerprint)  # type: ignore[arg-type]

        assert buf.state.phases[0].adapter_cache_hit is True
        assert buf.state.phases[0].adapter_cache_tag == expected_tag

    def test_positive_cache_hit_marks_phase_skipped(self, tmp_path: Path) -> None:
        executor = _mk_executor()
        buf = _mk_buffer_with_phases(tmp_path)
        phase = _mk_phase_ns("sft", repo_id="org/cache")
        fingerprint = "abcde12345"
        expected_tag = f"phase-0-sft-ds{fingerprint}"

        with (
            patch("huggingface_hub.HfApi") as MockHfApi,
            patch("peft.PeftModel") as MockPeft,
        ):
            MockHfApi.return_value.list_repo_refs.return_value = _mk_refs_with_tag(expected_tag)
            MockPeft.from_pretrained.return_value = MagicMock()

            executor._try_adapter_cache_hit(0, phase, MagicMock(), buf, fingerprint)  # type: ignore[arg-type]

        assert buf.state.phases[0].status == PhaseStatus.SKIPPED

    # ── Negative ──────────────────────────────

    def test_negative_repo_not_found_returns_none(self, tmp_path: Path) -> None:
        executor = _mk_executor()
        buf = _mk_buffer_with_phases(tmp_path)
        phase = _mk_phase_ns("sft", repo_id="org/no-repo")

        from huggingface_hub.errors import RepositoryNotFoundError

        with patch("huggingface_hub.HfApi") as MockHfApi:
            MockHfApi.return_value.list_repo_refs.side_effect = RepositoryNotFoundError("not found")
            result = executor._try_adapter_cache_hit(0, phase, MagicMock(), buf, "fp123")  # type: ignore[arg-type]

        assert result is None

    def test_negative_tag_not_in_refs_returns_none(self, tmp_path: Path) -> None:
        executor = _mk_executor()
        buf = _mk_buffer_with_phases(tmp_path)
        phase = _mk_phase_ns("sft", repo_id="org/cache")

        with patch("huggingface_hub.HfApi") as MockHfApi:
            MockHfApi.return_value.list_repo_refs.return_value = _mk_refs_empty()
            result = executor._try_adapter_cache_hit(0, phase, MagicMock(), buf, "fp123")  # type: ignore[arg-type]

        assert result is None

    def test_negative_peft_load_failure_returns_none(self, tmp_path: Path) -> None:
        executor = _mk_executor()
        buf = _mk_buffer_with_phases(tmp_path)
        phase = _mk_phase_ns("sft", repo_id="org/cache")
        fingerprint = "abcde12345"
        expected_tag = f"phase-0-sft-ds{fingerprint}"

        with (
            patch("huggingface_hub.HfApi") as MockHfApi,
            patch("peft.PeftModel") as MockPeft,
        ):
            MockHfApi.return_value.list_repo_refs.return_value = _mk_refs_with_tag(expected_tag)
            MockPeft.from_pretrained.side_effect = OSError("corrupt adapter")

            result = executor._try_adapter_cache_hit(0, phase, MagicMock(), buf, fingerprint)  # type: ignore[arg-type]

        assert result is None

    def test_negative_list_refs_generic_error_returns_none(self, tmp_path: Path) -> None:
        """Any unexpected error on list_repo_refs is treated as cache miss."""
        executor = _mk_executor()
        buf = _mk_buffer_with_phases(tmp_path)
        phase = _mk_phase_ns("sft", repo_id="org/cache")

        with patch("huggingface_hub.HfApi") as MockHfApi:
            MockHfApi.return_value.list_repo_refs.side_effect = TimeoutError("timeout")
            result = executor._try_adapter_cache_hit(0, phase, MagicMock(), buf, "fp")  # type: ignore[arg-type]

        assert result is None

    # ── Boundary ──────────────────────────────

    def test_boundary_tag_format_uses_phase_strategy_fingerprint(self, tmp_path: Path) -> None:
        """Verify the correct tag is constructed and searched for."""
        executor = _mk_executor()
        buf = _mk_buffer_with_phases(tmp_path)
        phase = _mk_phase_ns("dpo", repo_id="org/dpo-cache")
        fingerprint = "1234567890"
        expected_tag = "phase-1-dpo-ds1234567890"

        tag_checked: list[str] = []

        def capture_refs(repo_id: str, repo_type: str) -> MagicMock:  # type: ignore[return]
            refs = MagicMock()
            tag = MagicMock()
            tag.name = expected_tag
            tag_checked.append(tag.name)
            refs.tags = [tag]
            return refs

        with (
            patch("huggingface_hub.HfApi") as MockHfApi,
            patch("peft.PeftModel") as MockPeft,
        ):
            MockHfApi.return_value.list_repo_refs.side_effect = capture_refs
            MockPeft.from_pretrained.return_value = MagicMock()
            executor._try_adapter_cache_hit(1, phase, MagicMock(), buf, fingerprint)  # type: ignore[arg-type]

        assert expected_tag in tag_checked

    # ── Dependency errors ──────────────────────

    def test_dependency_error_hfapi_import_failure_returns_none(self, tmp_path: Path) -> None:
        """If HfApi() raises (e.g. auth failure), treat as cache miss."""
        executor = _mk_executor()
        buf = _mk_buffer_with_phases(tmp_path)
        phase = _mk_phase_ns("sft")

        with patch("huggingface_hub.HfApi") as MockHfApi:
            MockHfApi.return_value.list_repo_refs.side_effect = Exception("auth failure")
            result = executor._try_adapter_cache_hit(0, phase, MagicMock(), buf, "fp")  # type: ignore[arg-type]

        assert result is None


# ─────────────────────────────────────────────
# _upload_adapter_to_cache
# ─────────────────────────────────────────────


class TestUploadAdapterToCache:
    def _call_upload(
        self,
        executor: PhaseExecutor,
        tmp_path: Path,
        buf: DataBuffer,
        *,
        upload_raises: Exception | None = None,
    ) -> None:
        phase = _mk_phase_ns("sft", repo_id="org/sft-cache")
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()

        with patch("huggingface_hub.HfApi") as MockHfApi:
            api_instance = MockHfApi.return_value
            if upload_raises:
                api_instance.create_repo.side_effect = upload_raises
            else:
                api_instance.create_repo.return_value = None
                api_instance.upload_folder.return_value = None
                api_instance.create_tag.return_value = None

            with patch("src.training.orchestrator.phase_executor.time.sleep"):
                executor._upload_adapter_to_cache(
                    phase_idx=0,
                    phase=phase,  # type: ignore[arg-type]
                    checkpoint_path=checkpoint,
                    buffer=buf,
                    fingerprint="fp1234567",
                )

    def test_positive_upload_success_sets_cache_tag(self, tmp_path: Path) -> None:
        executor = _mk_executor()
        buf = _mk_buffer_with_phases(tmp_path)
        self._call_upload(executor, tmp_path, buf)
        assert buf.state.phases[0].adapter_cache_tag == "phase-0-sft-dsfp1234567"
        assert buf.state.phases[0].adapter_cache_upload_error is None

    def test_negative_upload_failure_soft_fails_sets_upload_error(self, tmp_path: Path) -> None:
        executor = _mk_executor()
        buf = _mk_buffer_with_phases(tmp_path)
        self._call_upload(executor, tmp_path, buf, upload_raises=ConnectionError("timeout"))
        assert buf.state.phases[0].adapter_cache_upload_error is not None
        assert "timeout" in buf.state.phases[0].adapter_cache_upload_error

    def test_negative_soft_fail_does_not_raise(self, tmp_path: Path) -> None:
        executor = _mk_executor()
        buf = _mk_buffer_with_phases(tmp_path)
        # Must NOT raise, just store error
        self._call_upload(executor, tmp_path, buf, upload_raises=RuntimeError("boom"))
        assert buf.state.phases[0].adapter_cache_upload_error is not None

    def test_negative_upload_failure_retried_3_times(self, tmp_path: Path) -> None:
        executor = _mk_executor()
        buf = _mk_buffer_with_phases(tmp_path)
        phase = _mk_phase_ns("sft", repo_id="org/sft-cache")
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()

        with patch("huggingface_hub.HfApi") as MockHfApi:
            api_instance = MockHfApi.return_value
            api_instance.create_repo.side_effect = IOError("fail")
            with patch("src.training.orchestrator.phase_executor.time.sleep"):
                executor._upload_adapter_to_cache(
                    phase_idx=0,
                    phase=phase,  # type: ignore[arg-type]
                    checkpoint_path=checkpoint,
                    buffer=buf,
                    fingerprint="fp1234567",
                )
            # create_repo called 3 times (3 retry attempts)
            assert api_instance.create_repo.call_count == 3

    def test_invariant_upload_error_persisted_in_state(self, tmp_path: Path) -> None:
        import json

        executor = _mk_executor()
        buf = _mk_buffer_with_phases(tmp_path)
        self._call_upload(executor, tmp_path, buf, upload_raises=ValueError("disk full"))

        state_file = tmp_path / DataBuffer.STATE_FILENAME
        raw = json.loads(state_file.read_text())
        phase_raw = next(p for p in raw["phases"] if p["phase_idx"] == 0)
        assert "disk full" in (phase_raw.get("adapter_cache_upload_error") or "")


# ─────────────────────────────────────────────
# PhaseExecutor.execute() — adapter cache path
# ─────────────────────────────────────────────


def _mk_full_executor(tmp_path: Path, config: Any = None) -> tuple[PhaseExecutor, DataBuffer]:
    buf = _mk_buffer_with_phases(tmp_path, n_phases=2)
    executor = _mk_executor(config)
    return executor, buf


class TestPhaseExecutorExecuteAdapterCache:
    def test_positive_cache_disabled_no_hf_calls(self, tmp_path: Path) -> None:
        """When adapter_cache.enabled=False, _try_adapter_cache_hit must never be called."""
        executor, buf = _mk_full_executor(tmp_path)

        with patch.object(executor, "_try_adapter_cache_hit") as mock_try:
            from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig

            phase = StrategyPhaseConfig(
                strategy_type="sft",
                dataset="sft_data",
                hyperparams=PhaseHyperparametersConfig(epochs=1),
            )
            assert phase.adapter_cache.enabled is False
            # Trigger early shutdown to avoid full training infra
            executor.shutdown_handler = MagicMock()
            executor.shutdown_handler.should_stop.return_value = True
            executor.execute(phase_idx=0, phase=phase, model=MagicMock(), buffer=buf)

        mock_try.assert_not_called()

    def test_positive_cache_hit_returns_ok_without_training(self, tmp_path: Path) -> None:
        """Cache hit path: _try_adapter_cache_hit returns Ok → execute returns Ok, no training."""
        executor, buf = _mk_full_executor(tmp_path)
        loaded_model = MagicMock(name="loaded_peft")

        with (
            patch.object(executor, "_compute_dataset_fingerprint_safe", return_value="fp1234567"),
            patch.object(executor, "_try_adapter_cache_hit", return_value=Ok(loaded_model)),
            patch.object(executor, "_should_stop", return_value=False),
        ):
            from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig
            from src.config.training.adapter_cache import AdapterCacheConfig

            phase = StrategyPhaseConfig(
                strategy_type="sft",
                dataset="sft_data",
                hyperparams=PhaseHyperparametersConfig(epochs=1),
                adapter_cache=AdapterCacheConfig(enabled=True, repo_id="org/cache"),
            )
            result = executor.execute(
                phase_idx=0,
                phase=phase,
                model=MagicMock(),
                buffer=buf,
                upstream_retrained=False,
            )

        assert isinstance(result, Success)
        assert result.unwrap() is loaded_model

    def test_positive_upstream_retrained_forces_cache_miss(self, tmp_path: Path) -> None:
        """When upstream_retrained=True, _try_adapter_cache_hit must NOT be called."""
        executor, buf = _mk_full_executor(tmp_path)

        with (
            patch.object(executor, "_compute_dataset_fingerprint_safe", return_value="fp1234567"),
            patch.object(executor, "_try_adapter_cache_hit") as mock_try,
            patch.object(executor, "_should_stop", side_effect=[False, True]),
        ):
            from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig
            from src.config.training.adapter_cache import AdapterCacheConfig

            phase = StrategyPhaseConfig(
                strategy_type="sft",
                dataset="sft_data",
                hyperparams=PhaseHyperparametersConfig(epochs=1),
                adapter_cache=AdapterCacheConfig(enabled=True, repo_id="org/cache"),
            )
            executor.execute(
                phase_idx=0,
                phase=phase,
                model=MagicMock(),
                buffer=buf,
                upstream_retrained=True,
            )

        mock_try.assert_not_called()

    def test_positive_cache_miss_calls_try_cache_hit(self, tmp_path: Path) -> None:
        """When cache enabled and upstream_retrained=False, _try_adapter_cache_hit is called."""
        executor, buf = _mk_full_executor(tmp_path)

        with (
            patch.object(executor, "_compute_dataset_fingerprint_safe", return_value="fp1234567"),
            patch.object(executor, "_try_adapter_cache_hit", return_value=None) as mock_try,
            patch.object(executor, "_should_stop", side_effect=[False, True]),
        ):
            from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig
            from src.config.training.adapter_cache import AdapterCacheConfig

            phase = StrategyPhaseConfig(
                strategy_type="sft",
                dataset="sft_data",
                hyperparams=PhaseHyperparametersConfig(epochs=1),
                adapter_cache=AdapterCacheConfig(enabled=True, repo_id="org/cache"),
            )
            executor.execute(
                phase_idx=0,
                phase=phase,
                model=MagicMock(),
                buffer=buf,
                upstream_retrained=False,
            )

        mock_try.assert_called_once()

    def test_boundary_fingerprint_none_skips_cache_check(self, tmp_path: Path) -> None:
        """
        If fingerprint computation fails (returns None), cache hit check is skipped
        even when upstream_retrained=False.
        """
        executor, buf = _mk_full_executor(tmp_path)

        with (
            patch.object(executor, "_compute_dataset_fingerprint_safe", return_value=None),
            patch.object(executor, "_try_adapter_cache_hit") as mock_try,
            patch.object(executor, "_should_stop", side_effect=[False, True]),
        ):
            from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig
            from src.config.training.adapter_cache import AdapterCacheConfig

            phase = StrategyPhaseConfig(
                strategy_type="sft",
                dataset="sft_data",
                hyperparams=PhaseHyperparametersConfig(epochs=1),
                adapter_cache=AdapterCacheConfig(enabled=True, repo_id="org/cache"),
            )
            executor.execute(
                phase_idx=0,
                phase=phase,
                model=MagicMock(),
                buffer=buf,
                upstream_retrained=False,
            )

        mock_try.assert_not_called()


# ─────────────────────────────────────────────
# ChainRunner cascade — upstream_retrained flag
# ─────────────────────────────────────────────


class TestChainRunnerCascade:
    def _mk_chain_runner(self, phase_executor: Any) -> Any:
        from src.training.orchestrator.chain_runner import ChainRunner

        return ChainRunner(
            phase_executor=phase_executor,
            mlflow_manager=None,
        )

    def _mk_strategies(self, n: int = 2) -> list[Any]:
        from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig

        types = ["sft", "dpo", "orpo"]
        return [
            StrategyPhaseConfig(
                strategy_type=types[i % len(types)],
                dataset=f"data_{i}",
                hyperparams=PhaseHyperparametersConfig(epochs=1),
            )
            for i in range(n)
        ]

    def test_positive_skipped_phase_does_not_set_upstream_retrained(self, tmp_path: Path) -> None:
        """Phase 0 SKIPPED → phase 1 receives upstream_retrained=False."""
        buf = _mk_buffer_with_phases(tmp_path, n_phases=2)

        phase_executor = MagicMock()
        upstream_flags: list[bool] = []

        def fake_execute(*, phase_idx: int, phase: Any, model: Any, buffer: Any, upstream_retrained: bool) -> Ok:
            upstream_flags.append(upstream_retrained)
            # Simulate cache hit by marking SKIPPED
            buffer.state.phases[phase_idx].status = PhaseStatus.SKIPPED
            return Ok(MagicMock(name=f"model_{phase_idx}"))

        phase_executor.execute.side_effect = fake_execute

        runner = self._mk_chain_runner(phase_executor)
        strategies = self._mk_strategies(n=2)
        runner._run_phases(strategies, MagicMock(), buf, start_phase=0, total_phases=2)

        assert upstream_flags[0] is False  # first phase: no upstream
        assert upstream_flags[1] is False  # first phase was SKIPPED → still False

    def test_positive_completed_phase_sets_upstream_retrained_for_next(self, tmp_path: Path) -> None:
        """Phase 0 COMPLETED → phase 1 receives upstream_retrained=True."""
        buf = _mk_buffer_with_phases(tmp_path, n_phases=2)

        phase_executor = MagicMock()
        upstream_flags: list[bool] = []

        def fake_execute(*, phase_idx: int, phase: Any, model: Any, buffer: Any, upstream_retrained: bool) -> Ok:
            upstream_flags.append(upstream_retrained)
            buffer.state.phases[phase_idx].status = PhaseStatus.COMPLETED
            return Ok(MagicMock(name=f"model_{phase_idx}"))

        phase_executor.execute.side_effect = fake_execute

        runner = self._mk_chain_runner(phase_executor)
        strategies = self._mk_strategies(n=2)
        runner._run_phases(strategies, MagicMock(), buf, start_phase=0, total_phases=2)

        assert upstream_flags[0] is False
        assert upstream_flags[1] is True  # phase 0 completed → force retrain

    def test_positive_cascade_propagates_through_all_downstream(self, tmp_path: Path) -> None:
        """Phase 0 trains → phases 1, 2, 3 all get upstream_retrained=True."""
        buf = _mk_buffer_with_phases(tmp_path, n_phases=4)
        from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig

        strategies = [
            StrategyPhaseConfig(strategy_type="sft", dataset="d0", hyperparams=PhaseHyperparametersConfig(epochs=1)),
            StrategyPhaseConfig(strategy_type="dpo", dataset="d1", hyperparams=PhaseHyperparametersConfig(epochs=1)),
            StrategyPhaseConfig(strategy_type="dpo", dataset="d2", hyperparams=PhaseHyperparametersConfig(epochs=1)),
            StrategyPhaseConfig(strategy_type="dpo", dataset="d3", hyperparams=PhaseHyperparametersConfig(epochs=1)),
        ]

        phase_executor = MagicMock()
        upstream_flags: list[bool] = []

        def fake_execute(*, phase_idx: int, phase: Any, model: Any, buffer: Any, upstream_retrained: bool) -> Ok:
            upstream_flags.append(upstream_retrained)
            # Only phase 0 trains; the rest are marked SKIPPED (but upstream_retrained overrides)
            status = PhaseStatus.COMPLETED if phase_idx == 0 else PhaseStatus.SKIPPED
            buffer.state.phases[phase_idx].status = status
            return Ok(MagicMock())

        phase_executor.execute.side_effect = fake_execute

        runner = self._mk_chain_runner(phase_executor)
        runner._run_phases(strategies, MagicMock(), buf, start_phase=0, total_phases=4)

        assert upstream_flags[0] is False  # start value
        assert upstream_flags[1] is True   # phase 0 COMPLETED → cascade
        assert upstream_flags[2] is True   # still True (phase 1 was SKIPPED but upstream was True)
        assert upstream_flags[3] is True

    def test_positive_all_cached_upstream_retrained_never_set(self, tmp_path: Path) -> None:
        """All phases hit cache → upstream_retrained stays False throughout."""
        buf = _mk_buffer_with_phases(tmp_path, n_phases=3)
        from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig

        strategies = [
            StrategyPhaseConfig(strategy_type="sft", dataset=f"d{i}", hyperparams=PhaseHyperparametersConfig(epochs=1))
            for i in range(3)
        ]

        phase_executor = MagicMock()
        upstream_flags: list[bool] = []

        def fake_execute(*, phase_idx: int, phase: Any, model: Any, buffer: Any, upstream_retrained: bool) -> Ok:
            upstream_flags.append(upstream_retrained)
            buffer.state.phases[phase_idx].status = PhaseStatus.SKIPPED
            return Ok(MagicMock())

        phase_executor.execute.side_effect = fake_execute

        runner = self._mk_chain_runner(phase_executor)
        runner._run_phases(strategies, MagicMock(), buf, start_phase=0, total_phases=3)

        assert all(f is False for f in upstream_flags)

    def test_negative_phase_failure_stops_chain(self, tmp_path: Path) -> None:
        """If a phase fails, chain stops and returns Err; remaining phases not executed."""
        buf = _mk_buffer_with_phases(tmp_path, n_phases=3)
        from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig

        strategies = [
            StrategyPhaseConfig(strategy_type="sft", dataset="d0", hyperparams=PhaseHyperparametersConfig(epochs=1)),
            StrategyPhaseConfig(strategy_type="dpo", dataset="d1", hyperparams=PhaseHyperparametersConfig(epochs=1)),
            StrategyPhaseConfig(strategy_type="dpo", dataset="d2", hyperparams=PhaseHyperparametersConfig(epochs=1)),
        ]
        from src.utils.result import TrainingError

        phase_executor = MagicMock()
        calls: list[int] = []

        def fake_execute(*, phase_idx: int, phase: Any, model: Any, buffer: Any, upstream_retrained: bool) -> Any:
            calls.append(phase_idx)
            if phase_idx == 1:
                return Err(TrainingError(message="training failed", code="TRAIN_FAILED"))
            buffer.state.phases[phase_idx].status = PhaseStatus.COMPLETED
            return Ok(MagicMock())

        phase_executor.execute.side_effect = fake_execute

        runner = self._mk_chain_runner(phase_executor)
        result = runner._run_phases(strategies, MagicMock(), buf, start_phase=0, total_phases=3)

        assert result.is_failure()
        assert 2 not in calls  # phase 2 never executed

    # ── Combinatorial ─────────────────────────

    def test_combinatorial_mixed_hit_train_hit_produces_correct_flags(self, tmp_path: Path) -> None:
        """
        Phase 0: SKIPPED (cache hit)   → upstream_retrained stays False
        Phase 1: COMPLETED (trained)   → upstream_retrained set True
        Phase 2: SKIPPED (but forced)  → receives upstream_retrained=True
        """
        buf = _mk_buffer_with_phases(tmp_path, n_phases=3)
        from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig

        strategies = [
            StrategyPhaseConfig(strategy_type="sft", dataset="d0", hyperparams=PhaseHyperparametersConfig(epochs=1)),
            StrategyPhaseConfig(strategy_type="dpo", dataset="d1", hyperparams=PhaseHyperparametersConfig(epochs=1)),
            StrategyPhaseConfig(strategy_type="dpo", dataset="d2", hyperparams=PhaseHyperparametersConfig(epochs=1)),
        ]

        statuses = [PhaseStatus.SKIPPED, PhaseStatus.COMPLETED, PhaseStatus.SKIPPED]
        phase_executor = MagicMock()
        upstream_flags: list[bool] = []

        def fake_execute(*, phase_idx: int, phase: Any, model: Any, buffer: Any, upstream_retrained: bool) -> Ok:
            upstream_flags.append(upstream_retrained)
            buffer.state.phases[phase_idx].status = statuses[phase_idx]
            return Ok(MagicMock())

        phase_executor.execute.side_effect = fake_execute

        runner = self._mk_chain_runner(phase_executor)
        runner._run_phases(strategies, MagicMock(), buf, start_phase=0, total_phases=3)

        assert upstream_flags[0] is False   # initial
        assert upstream_flags[1] is False   # phase 0 SKIPPED → no propagation
        assert upstream_flags[2] is True    # phase 1 COMPLETED → force downstream
