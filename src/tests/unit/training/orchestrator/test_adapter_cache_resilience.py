"""
Resilience tests for AdapterCacheManager upload machinery.

Coverage matrix
───────────────
_call_with_timeout   success / timeout / label in error / timeout_s in error /
                     stalled-hint / fn exception propagated / complex value / empty label
_retry_call defaults default retries/delay use HF_UPLOAD_RETRIES/HF_UPLOAD_RETRY_DELAY_S
upload_large_folder  called (not upload_folder) / allow_patterns / no commit_message /
                     create_repo before upload / create_tag after upload
timeout wrapping     _call_with_timeout wraps do_upload (timeout_s = HF_UPLOAD_TIMEOUT_S)
retry cascade        timeout → retry / success on 2nd attempt / all exhausted → error in state
combinatorial        1 timeout + 2nd success / 3 timeouts → soft-fail / mixed errors
regression           upload_folder must NOT appear as a call
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from src.constants import (
    HF_UPLOAD_RETRIES,
    HF_UPLOAD_RETRY_DELAY_S,
    HF_UPLOAD_TIMEOUT_S,
    LORA_CHECKPOINT_PATTERNS,
)
from src.training.orchestrator.phase_executor.adapter_cache import (
    _call_with_timeout,
    _retry_call,
)
from src.training.managers.data_buffer import DataBuffer

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _mk_buffer(tmp_path: Path, n: int = 1) -> DataBuffer:
    from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig

    phases = [
        StrategyPhaseConfig(
            strategy_type="sft",
            dataset="ds",
            hyperparams=PhaseHyperparametersConfig(epochs=1),
        )
        for _ in range(n)
    ]
    buf = DataBuffer(base_output_dir=tmp_path, base_model_path="base")
    buf.init_pipeline(phases, force=True)
    return buf


def _mk_phase(repo_id: str = "org/cache") -> SimpleNamespace:
    return SimpleNamespace(
        strategy_type="sft",
        adapter_cache=SimpleNamespace(enabled=True, repo_id=repo_id, private=True),
    )


def _mk_cache_manager(tmp_path: Path) -> Any:
    from src.training.orchestrator.phase_executor.adapter_cache import AdapterCacheManager

    config = MagicMock()
    return AdapterCacheManager(config=config)


# ─────────────────────────────────────────────────────────────
# _call_with_timeout
# ─────────────────────────────────────────────────────────────


class TestCallWithTimeout:
    # ── Positive ─────────────────────────────────────────────

    def test_positive_fast_fn_returns_result(self) -> None:
        result = _call_with_timeout(lambda: 42, timeout_s=5)
        assert result == 42

    def test_positive_complex_return_value_preserved(self) -> None:
        obj = {"nested": [1, 2, {"x": True}]}
        result = _call_with_timeout(lambda: obj, timeout_s=5)
        assert result is obj

    def test_positive_none_return_value_preserved(self) -> None:
        result = _call_with_timeout(lambda: None, timeout_s=5)
        assert result is None

    # ── Negative ─────────────────────────────────────────────

    def test_negative_timeout_exceeded_raises_timeout_error(self) -> None:
        with patch("concurrent.futures.Future.result", side_effect=concurrent.futures.TimeoutError()):
            with pytest.raises(TimeoutError):
                _call_with_timeout(lambda: None, timeout_s=1, label="upload")

    def test_negative_error_message_contains_label(self) -> None:
        with patch("concurrent.futures.Future.result", side_effect=concurrent.futures.TimeoutError()):
            with pytest.raises(TimeoutError, match="my_special_label"):
                _call_with_timeout(lambda: None, timeout_s=1, label="my_special_label")

    def test_negative_error_message_contains_timeout_duration(self) -> None:
        with patch("concurrent.futures.Future.result", side_effect=concurrent.futures.TimeoutError()):
            with pytest.raises(TimeoutError, match="1800s"):
                _call_with_timeout(lambda: None, timeout_s=1800)

    def test_negative_error_message_contains_stalled_hint(self) -> None:
        with patch("concurrent.futures.Future.result", side_effect=concurrent.futures.TimeoutError()):
            with pytest.raises(TimeoutError, match="stalled"):
                _call_with_timeout(lambda: None, timeout_s=1)

    def test_negative_fn_exception_propagates_directly(self) -> None:
        """Exceptions from fn must surface as-is, NOT wrapped in TimeoutError."""
        def fail() -> None:
            raise ValueError("disk full")

        with pytest.raises(ValueError, match="disk full"):
            _call_with_timeout(fail, timeout_s=5)

    def test_negative_fn_exception_is_not_timeout_error(self) -> None:
        """Fn exception should never be converted to TimeoutError."""
        def fail() -> None:
            raise RuntimeError("connection reset")

        with pytest.raises(RuntimeError):
            _call_with_timeout(fail, timeout_s=5)

    # ── Boundary ──────────────────────────────────────────────

    def test_boundary_empty_label_still_raises_timeout_error(self) -> None:
        with patch("concurrent.futures.Future.result", side_effect=concurrent.futures.TimeoutError()):
            with pytest.raises(TimeoutError):
                _call_with_timeout(lambda: None, timeout_s=1, label="")

    def test_boundary_timeout_zero_simulated(self) -> None:
        """Mocked timeout=0 behaves the same as larger values — raises TimeoutError."""
        with patch("concurrent.futures.Future.result", side_effect=concurrent.futures.TimeoutError()):
            with pytest.raises(TimeoutError):
                _call_with_timeout(lambda: None, timeout_s=0)

    # ── Invariants ────────────────────────────────────────────

    def test_invariant_raised_error_is_builtin_timeout_error_not_futures(self) -> None:
        """We raise builtins.TimeoutError. In Python 3.11+ concurrent.futures.TimeoutError
        is an alias for builtins.TimeoutError, so both isinstance checks pass — that is fine.
        What matters is that it is a TimeoutError (not some other exception)."""
        with patch("concurrent.futures.Future.result", side_effect=concurrent.futures.TimeoutError()):
            with pytest.raises(TimeoutError) as exc_info:
                _call_with_timeout(lambda: None, timeout_s=1)
        # Must be a TimeoutError (covers both builtins.TimeoutError and its cf alias)
        assert isinstance(exc_info.value, TimeoutError)


# ─────────────────────────────────────────────────────────────
# _retry_call — default constants
# ─────────────────────────────────────────────────────────────


class TestRetryCallDefaults:
    """Verify that _retry_call default args use the centralized constants."""

    def test_invariant_default_retries_match_hf_upload_retries(self) -> None:
        import inspect
        sig = inspect.signature(_retry_call)
        assert sig.parameters["retries"].default == HF_UPLOAD_RETRIES

    def test_invariant_default_delay_match_hf_upload_retry_delay(self) -> None:
        import inspect
        sig = inspect.signature(_retry_call)
        assert sig.parameters["delay_s"].default == HF_UPLOAD_RETRY_DELAY_S

    def test_positive_default_retries_is_3(self) -> None:
        """HF_UPLOAD_RETRIES must be 3 (documented community best-practice)."""
        assert HF_UPLOAD_RETRIES == 3

    def test_positive_default_timeout_is_1800(self) -> None:
        """HF_UPLOAD_TIMEOUT_S must be 1800s (matches MR_UPLOAD_TIMEOUT)."""
        assert HF_UPLOAD_TIMEOUT_S == 1800


# ─────────────────────────────────────────────────────────────
# upload_large_folder usage (not upload_folder)
# ─────────────────────────────────────────────────────────────


class TestUploadLargeFolderUsage:
    """
    Verify that AdapterCacheManager.upload uses upload_large_folder (not upload_folder)
    with the correct parameters.
    """

    def _run_upload(
        self,
        tmp_path: Path,
        *,
        upload_side_effect: Any = None,
    ) -> tuple[Any, MagicMock]:
        mgr = _mk_cache_manager(tmp_path)
        buf = _mk_buffer(tmp_path)
        phase = _mk_phase("org/test-cache")
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()

        with patch("huggingface_hub.HfApi") as MockApi:
            api = MockApi.return_value
            if upload_side_effect:
                api.upload_large_folder.side_effect = upload_side_effect
            with patch("src.training.orchestrator.phase_executor.adapter_cache.time.sleep"):
                mgr.upload(
                    phase_idx=0,
                    phase=phase,  # type: ignore[arg-type]
                    checkpoint_path=checkpoint,
                    buffer=buf,
                    fingerprint="abc1234567",
                )
            return buf, api

    # ── Positive ─────────────────────────────────────────────

    def test_positive_upload_large_folder_called(self, tmp_path: Path) -> None:
        _, api = self._run_upload(tmp_path)
        api.upload_large_folder.assert_called_once()

    def test_regression_upload_folder_never_called(self, tmp_path: Path) -> None:
        """Regression: old upload_folder must NOT be called."""
        _, api = self._run_upload(tmp_path)
        api.upload_folder.assert_not_called()

    def test_positive_allow_patterns_matches_lora_checkpoint_patterns(self, tmp_path: Path) -> None:
        _, api = self._run_upload(tmp_path)
        kwargs = api.upload_large_folder.call_args.kwargs
        assert kwargs["allow_patterns"] == list(LORA_CHECKPOINT_PATTERNS)

    def test_positive_allow_patterns_contains_adapter_safetensors(self, tmp_path: Path) -> None:
        _, api = self._run_upload(tmp_path)
        patterns = api.upload_large_folder.call_args.kwargs["allow_patterns"]
        assert "adapter_model.safetensors" in patterns

    def test_positive_allow_patterns_contains_tokenizer_files(self, tmp_path: Path) -> None:
        _, api = self._run_upload(tmp_path)
        patterns = api.upload_large_folder.call_args.kwargs["allow_patterns"]
        assert "tokenizer.json" in patterns
        assert "tokenizer_config.json" in patterns
        assert "special_tokens_map.json" in patterns

    def test_regression_no_commit_message_kwarg(self, tmp_path: Path) -> None:
        """upload_large_folder does not accept commit_message — must not be passed."""
        _, api = self._run_upload(tmp_path)
        kwargs = api.upload_large_folder.call_args.kwargs
        assert "commit_message" not in kwargs

    def test_positive_repo_id_passed_to_upload(self, tmp_path: Path) -> None:
        _, api = self._run_upload(tmp_path)
        kwargs = api.upload_large_folder.call_args.kwargs
        assert kwargs["repo_id"] == "org/test-cache"

    def test_positive_repo_type_is_model(self, tmp_path: Path) -> None:
        _, api = self._run_upload(tmp_path)
        kwargs = api.upload_large_folder.call_args.kwargs
        assert kwargs["repo_type"] == "model"

    def test_positive_create_repo_called_before_upload(self, tmp_path: Path) -> None:
        """create_repo must be called before upload_large_folder."""
        _, api = self._run_upload(tmp_path)
        # Verify call order via mock_calls
        call_names = [c[0] for c in api.mock_calls]
        create_idx = next(i for i, n in enumerate(call_names) if "create_repo" in n)
        upload_idx = next(i for i, n in enumerate(call_names) if "upload_large_folder" in n)
        assert create_idx < upload_idx

    def test_positive_create_tag_called_after_upload(self, tmp_path: Path) -> None:
        """create_tag must be called after upload_large_folder completes."""
        _, api = self._run_upload(tmp_path)
        call_names = [c[0] for c in api.mock_calls]
        upload_idx = next(i for i, n in enumerate(call_names) if "upload_large_folder" in n)
        tag_idx = next(i for i, n in enumerate(call_names) if "create_tag" in n)
        assert upload_idx < tag_idx

    def test_positive_tag_contains_fingerprint(self, tmp_path: Path) -> None:
        _, api = self._run_upload(tmp_path)
        tag_kwargs = api.create_tag.call_args.kwargs
        assert "abc1234567" in tag_kwargs["tag"]

    # ── Boundary ──────────────────────────────────────────────

    def test_boundary_folder_path_passed_as_string(self, tmp_path: Path) -> None:
        """folder_path must be a str (some HF versions don't accept Path objects)."""
        _, api = self._run_upload(tmp_path)
        kwargs = api.upload_large_folder.call_args.kwargs
        assert isinstance(kwargs["folder_path"], str)


# ─────────────────────────────────────────────────────────────
# Timeout wrapping in upload()
# ─────────────────────────────────────────────────────────────


class TestUploadTimeoutWrapping:
    """Verify that do_upload is wrapped in _call_with_timeout with HF_UPLOAD_TIMEOUT_S."""

    def test_positive_timeout_used_is_hf_upload_timeout_s(self, tmp_path: Path) -> None:
        mgr = _mk_cache_manager(tmp_path)
        buf = _mk_buffer(tmp_path)
        phase = _mk_phase()
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()

        timeout_values: list[int] = []

        def fake_call_with_timeout(fn: Any, timeout_s: int, label: str = "") -> None:
            timeout_values.append(timeout_s)
            fn()

        with (
            patch("huggingface_hub.HfApi"),
            patch(
                "src.training.orchestrator.phase_executor.adapter_cache._call_with_timeout",
                side_effect=fake_call_with_timeout,
            ),
            patch("src.training.orchestrator.phase_executor.adapter_cache.time.sleep"),
        ):
            mgr.upload(
                phase_idx=0,
                phase=phase,  # type: ignore[arg-type]
                checkpoint_path=checkpoint,
                buffer=buf,
                fingerprint="fp123",
            )

        assert len(timeout_values) == 1
        assert timeout_values[0] == HF_UPLOAD_TIMEOUT_S

    def test_positive_label_contains_phase_idx_and_repo(self, tmp_path: Path) -> None:
        mgr = _mk_cache_manager(tmp_path)
        buf = _mk_buffer(tmp_path, n=3)  # 3 phases so phase_idx=2 is valid
        phase = _mk_phase("myorg/my-cache")
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()

        labels: list[str] = []

        def fake_call_with_timeout(fn: Any, timeout_s: int, label: str = "") -> None:
            labels.append(label)
            fn()

        with (
            patch("huggingface_hub.HfApi"),
            patch(
                "src.training.orchestrator.phase_executor.adapter_cache._call_with_timeout",
                side_effect=fake_call_with_timeout,
            ),
            patch("src.training.orchestrator.phase_executor.adapter_cache.time.sleep"),
        ):
            mgr.upload(
                phase_idx=2,
                phase=phase,  # type: ignore[arg-type]
                checkpoint_path=checkpoint,
                buffer=buf,
                fingerprint="fp",
            )

        assert labels
        assert "2" in labels[0]
        assert "myorg/my-cache" in labels[0]


# ─────────────────────────────────────────────────────────────
# Timeout → retry cascade
# ─────────────────────────────────────────────────────────────


class TestTimeoutRetryCascade:
    """
    Integration: when _call_with_timeout raises TimeoutError,
    _retry_call must retry and eventually soft-fail or succeed.
    """

    def _upload_with_fake_timeout(
        self,
        tmp_path: Path,
        *,
        n_timeouts: int,
        succeed_after: bool = False,
    ) -> tuple[Any, list[int]]:
        mgr = _mk_cache_manager(tmp_path)
        buf = _mk_buffer(tmp_path)
        phase = _mk_phase()
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()

        call_counts: list[int] = []
        attempt = [0]

        def fake_call_with_timeout(fn: Any, timeout_s: int, label: str = "") -> None:
            attempt[0] += 1
            call_counts.append(attempt[0])
            if attempt[0] <= n_timeouts:
                raise TimeoutError(f"simulated timeout #{attempt[0]}")
            fn()

        with (
            patch("huggingface_hub.HfApi"),
            patch(
                "src.training.orchestrator.phase_executor.adapter_cache._call_with_timeout",
                side_effect=fake_call_with_timeout,
            ),
            patch("src.training.orchestrator.phase_executor.adapter_cache.time.sleep"),
        ):
            mgr.upload(
                phase_idx=0,
                phase=phase,  # type: ignore[arg-type]
                checkpoint_path=checkpoint,
                buffer=buf,
                fingerprint="fp",
            )

        return buf, call_counts

    # ── Positive ─────────────────────────────────────────────

    def test_positive_timeout_triggers_retry(self, tmp_path: Path) -> None:
        """One timeout → retry → second attempt succeeds."""
        buf, counts = self._upload_with_fake_timeout(tmp_path, n_timeouts=1)
        assert len(counts) == 2
        assert buf.state.phases[0].adapter_cache_upload_error is None

    def test_positive_two_timeouts_then_success(self, tmp_path: Path) -> None:
        buf, counts = self._upload_with_fake_timeout(tmp_path, n_timeouts=2)
        assert len(counts) == 3
        assert buf.state.phases[0].adapter_cache_upload_error is None

    # ── Negative ─────────────────────────────────────────────

    def test_negative_all_retries_exhausted_soft_fails(self, tmp_path: Path) -> None:
        """3 timeouts (= HF_UPLOAD_RETRIES) → soft-fail, error stored in state."""
        buf, counts = self._upload_with_fake_timeout(tmp_path, n_timeouts=HF_UPLOAD_RETRIES)
        assert len(counts) == HF_UPLOAD_RETRIES
        assert buf.state.phases[0].adapter_cache_upload_error is not None

    def test_negative_error_message_stored_in_state(self, tmp_path: Path) -> None:
        buf, _ = self._upload_with_fake_timeout(tmp_path, n_timeouts=HF_UPLOAD_RETRIES)
        assert "timeout" in (buf.state.phases[0].adapter_cache_upload_error or "").lower()

    def test_negative_soft_fail_does_not_raise(self, tmp_path: Path) -> None:
        """upload() must never raise even when all retries are exhausted."""
        mgr = _mk_cache_manager(tmp_path)
        buf = _mk_buffer(tmp_path)
        phase = _mk_phase()
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()

        with (
            patch("huggingface_hub.HfApi"),
            patch(
                "src.training.orchestrator.phase_executor.adapter_cache._call_with_timeout",
                side_effect=TimeoutError("always times out"),
            ),
            patch("src.training.orchestrator.phase_executor.adapter_cache.time.sleep"),
        ):
            mgr.upload(
                phase_idx=0,
                phase=phase,  # type: ignore[arg-type]
                checkpoint_path=checkpoint,
                buffer=buf,
                fingerprint="fp",
            )
        # No exception raised — test passes if we reach here

    # ── Invariants ────────────────────────────────────────────

    def test_invariant_retry_count_bounded_by_hf_upload_retries(self, tmp_path: Path) -> None:
        """Total attempts must never exceed HF_UPLOAD_RETRIES even on all timeouts."""
        buf, counts = self._upload_with_fake_timeout(tmp_path, n_timeouts=100)
        assert len(counts) == HF_UPLOAD_RETRIES

    def test_invariant_state_persisted_on_soft_fail(self, tmp_path: Path) -> None:
        """Error must be saved to state file, not just in-memory."""
        import json

        buf, _ = self._upload_with_fake_timeout(tmp_path, n_timeouts=HF_UPLOAD_RETRIES)
        state_file = tmp_path / DataBuffer.STATE_FILENAME
        raw = json.loads(state_file.read_text())
        phase_raw = next(p for p in raw["phases"] if p["phase_idx"] == 0)
        assert phase_raw.get("adapter_cache_upload_error") is not None

    # ── Combinatorial ─────────────────────────────────────────

    def test_combinatorial_mixed_errors_last_error_stored(self, tmp_path: Path) -> None:
        """Mix of connection errors and timeouts — last error stored in state."""
        mgr = _mk_cache_manager(tmp_path)
        buf = _mk_buffer(tmp_path)
        phase = _mk_phase()
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()

        errors = [
            ConnectionError("connection reset"),
            TimeoutError("stalled"),
            IOError("network down"),
        ]
        attempt = [0]

        def fake_timeout(fn: Any, timeout_s: int, label: str = "") -> None:
            e = errors[attempt[0]]
            attempt[0] += 1
            raise e

        with (
            patch("huggingface_hub.HfApi"),
            patch(
                "src.training.orchestrator.phase_executor.adapter_cache._call_with_timeout",
                side_effect=fake_timeout,
            ),
            patch("src.training.orchestrator.phase_executor.adapter_cache.time.sleep"),
        ):
            mgr.upload(
                phase_idx=0,
                phase=phase,  # type: ignore[arg-type]
                checkpoint_path=checkpoint,
                buffer=buf,
                fingerprint="fp",
            )

        err = buf.state.phases[0].adapter_cache_upload_error
        assert err is not None
        assert "network down" in err  # last error is persisted

    def test_combinatorial_no_timeout_no_error_tag_stored(self, tmp_path: Path) -> None:
        """Clean upload: no timeouts, no errors → tag stored and no upload_error."""
        buf, counts = self._upload_with_fake_timeout(tmp_path, n_timeouts=0)
        assert len(counts) == 1
        assert buf.state.phases[0].adapter_cache_upload_error is None
        assert buf.state.phases[0].adapter_cache_tag is not None
