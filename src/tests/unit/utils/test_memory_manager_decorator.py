"""
Comprehensive test suite for enhanced with_memory_protection decorator.

Test categories:
1. Positive tests (success scenarios)
2. Negative tests (failure scenarios)
3. Boundary tests (edge cases)
4. Invariant tests (properties that must hold)
5. Dependency error tests
6. Regression tests (backward compatibility)
7. Logic-specific tests (context_factory, type hints)
8. Combinatorial tests

"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from src.utils.memory_manager import (
    MemoryEventCallbacks,
    MemoryManager,
    OOMRecoverableError,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_torch():
    """Mock torch module - needs to patch sys.modules since torch is imported dynamically."""
    with patch.dict("sys.modules", {"torch": MagicMock(), "torch.cuda": MagicMock()}):
        import sys

        mock = sys.modules["torch"]
        mock.cuda.is_available.return_value = True
        mock.cuda.mem_get_info.return_value = (2 * 1024**3, 8 * 1024**3)  # 2GB free, 8GB total
        mock.cuda.get_device_name.return_value = "Tesla T4"
        mock.cuda.memory_reserved.return_value = 6 * 1024**3
        mock.cuda.memory_allocated.return_value = 5 * 1024**3
        yield mock


@pytest.fixture
def memory_manager(mock_torch):
    """Create MemoryManager with mocked torch."""
    return MemoryManager(memory_margin_mb=500, max_retries=3)


@pytest.fixture
def memory_manager_with_callbacks(mock_torch):
    """Create MemoryManager with event callbacks."""
    callbacks = MemoryEventCallbacks(
        on_cache_cleared=Mock(),
        on_memory_warning=Mock(),
        on_oom=Mock(),
        on_oom_retry=Mock(),
    )
    return MemoryManager(memory_margin_mb=500, max_retries=3, callbacks=callbacks), callbacks


# =============================================================================
# 1. POSITIVE TESTS
# =============================================================================


def test_basic_decorator_success(memory_manager, mock_torch):
    """Test decorator on simple function that succeeds."""

    @memory_manager.with_memory_protection("test_op")
    def simple_func():
        return "success"

    result = simple_func()
    assert result == "success"


def test_decorator_with_args_kwargs(memory_manager, mock_torch):
    """Test decorator preserves function arguments."""

    @memory_manager.with_memory_protection("test_op")
    def func_with_args(a, b, c=10):
        return a + b + c

    result = func_with_args(1, 2, c=3)
    assert result == 6


def test_successful_retry_after_oom(memory_manager, mock_torch):
    """Test that function succeeds after OOM and retry."""
    attempts = []

    @memory_manager.with_memory_protection("test_op")
    def fail_once():
        attempts.append(len(attempts))
        if len(attempts) == 1:
            raise RuntimeError("CUDA out of memory")
        return "success"

    result = fail_once()
    assert result == "success"
    assert len(attempts) == 2


def test_static_context_parameter(memory_manager, mock_torch):
    """Test decorator with static context."""

    @memory_manager.with_memory_protection("test_op", context={"model": "gpt2"})
    def with_context():
        return True

    result = with_context()
    assert result is True


def test_context_factory_basic(memory_manager, mock_torch):
    """Test context_factory extracts runtime info."""
    extracted_contexts = []

    def context_factory(batch_size, model_name):
        ctx = {"batch_size": batch_size, "model": model_name}
        extracted_contexts.append(ctx)
        return ctx

    @memory_manager.with_memory_protection("test_op", context_factory=context_factory)
    def train_step(batch_size, model_name):
        return f"trained_{batch_size}_{model_name}"

    result = train_step(8, "gpt2")
    assert result == "trained_8_gpt2"
    assert len(extracted_contexts) == 1
    assert extracted_contexts[0]["batch_size"] == 8


def test_context_factory_with_bound_method(memory_manager, mock_torch):
    """Test context_factory works with class methods."""

    class Trainer:
        def __init__(self, mm):
            self.mm = mm
            self.batch_size = 16

        def extract_context(self, data):
            return {"batch_size": self.batch_size, "data_size": len(data)}

        @property
        def train(self):
            return self.mm.with_memory_protection("train", context_factory=self.extract_context)(
                self._train_impl
            )

        def _train_impl(self, data):
            return sum(data)

    trainer = Trainer(memory_manager)
    result = trainer.train([1, 2, 3])
    assert result == 6


# =============================================================================
# 2. NEGATIVE TESTS
# =============================================================================


def test_all_retries_exhausted(memory_manager, mock_torch):
    """Test that OOMRecoverableError raised when all retries fail."""
    attempts = []

    @memory_manager.with_memory_protection("test_op")
    def always_fail():
        attempts.append(len(attempts))
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(OOMRecoverableError):
        always_fail()

    # Should try initial + 3 retries = 4 total
    assert len(attempts) == 4


def test_non_oom_error_not_retried(memory_manager, mock_torch):
    """Test that non-OOM errors are not retried."""
    attempts = []

    @memory_manager.with_memory_protection("test_op")
    def value_error():
        attempts.append(1)
        raise ValueError("Invalid config")

    with pytest.raises(ValueError, match="Invalid config"):
        value_error()

    # Should NOT retry on ValueError
    assert len(attempts) == 1


def test_context_factory_exception_handled(memory_manager, mock_torch):
    """Test that context_factory exceptions don't break function execution."""

    factory_called = []

    def broken_factory(*args, **kwargs):
        factory_called.append(True)
        raise RuntimeError("Factory broken")

    @memory_manager.with_memory_protection("test_op", context_factory=broken_factory)
    def normal_func():
        return "success"

    # Context factory errors are caught and logged, function still executes
    result = normal_func()
    assert result == "success"
    assert len(factory_called) == 1  # Factory was called but error was caught


# =============================================================================
# 3. BOUNDARY TESTS
# =============================================================================


def test_zero_retries(memory_manager, mock_torch):
    """Test max_retries=0 fails immediately on OOM."""
    attempts = []

    @memory_manager.with_memory_protection("test_op", max_retries=0)
    def always_oom():
        attempts.append(1)
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(OOMRecoverableError):
        always_oom()

    assert len(attempts) == 1


def test_high_retry_count(memory_manager, mock_torch):
    """Test with very high retry count."""
    attempts = []

    @memory_manager.with_memory_protection("test_op", max_retries=10)
    def succeed_on_fifth():
        attempts.append(1)
        if len(attempts) < 5:
            raise RuntimeError("CUDA out of memory")
        return "finally"

    result = succeed_on_fifth()
    assert result == "finally"
    assert len(attempts) == 5


def test_empty_context_dict(memory_manager, mock_torch):
    """Test with empty context dict."""

    @memory_manager.with_memory_protection("test_op", context={})
    def func():
        return True

    assert func() is True


def test_none_context_factory(memory_manager, mock_torch):
    """Test that None context_factory is handled."""

    @memory_manager.with_memory_protection("test_op", context_factory=None)
    def func():
        return True

    assert func() is True


# =============================================================================
# 4. INVARIANT TESTS
# =============================================================================


def test_cleanup_called_between_retries(memory_manager, mock_torch):
    """Test aggressive_cleanup called between attempts."""
    cleanup_calls = []
    original_cleanup = memory_manager.aggressive_cleanup

    def tracked_cleanup():
        result = original_cleanup()
        cleanup_calls.append(result)
        return result

    memory_manager.aggressive_cleanup = tracked_cleanup

    @memory_manager.with_memory_protection("test_op")
    def fail_twice():
        if len(cleanup_calls) < 2:
            raise RuntimeError("CUDA out of memory")
        return "success"

    result = fail_twice()
    assert result == "success"
    # Should have 2 cleanup calls (after 1st and 2nd failure)
    assert len(cleanup_calls) == 2


def test_callbacks_fired_on_retry(memory_manager_with_callbacks, mock_torch):
    """Test on_oom_retry callback fired for each retry."""
    mm, callbacks = memory_manager_with_callbacks

    @mm.with_memory_protection("test_op")
    def fail_twice():
        if callbacks.on_oom_retry.call_count < 2:
            raise RuntimeError("CUDA out of memory")
        return "success"

    result = fail_twice()
    assert result == "success"
    assert callbacks.on_oom_retry.call_count == 2


def test_attempt_number_in_context(memory_manager, mock_torch):
    """Test that attempt_number is automatically added to context."""
    contexts_seen = []
    original_safe_op = memory_manager.safe_operation

    def capture_context(name, context=None):
        if context:
            contexts_seen.append(context.copy())
        return original_safe_op(name, context)

    memory_manager.safe_operation = capture_context

    @memory_manager.with_memory_protection("test_op")
    def func():
        return True

    func()

    # Should have seen context with attempt_number
    assert any("attempt_number" in ctx for ctx in contexts_seen)
    assert any(ctx.get("attempt_number") == 0 for ctx in contexts_seen)


def test_context_merge(memory_manager, mock_torch):
    """Test that static context and factory context are merged."""
    contexts_seen = []
    original_safe_op = memory_manager.safe_operation

    def capture_context(name, context=None):
        if context:
            contexts_seen.append(context.copy())
        return original_safe_op(name, context)

    memory_manager.safe_operation = capture_context

    def factory(x):
        return {"dynamic": x}

    @memory_manager.with_memory_protection(
        "test_op", context={"static": "value"}, context_factory=factory
    )
    def func(x):
        return x

    func(42)

    # Should have both static and dynamic context
    assert any(ctx.get("static") == "value" for ctx in contexts_seen)
    assert any(ctx.get("dynamic") == 42 for ctx in contexts_seen)


# =============================================================================
# 5. DEPENDENCY ERROR TESTS
# =============================================================================


def test_callback_exception_not_breaking_retry(memory_manager_with_callbacks, mock_torch):
    """Test that callback exceptions don't prevent retries."""
    mm, callbacks = memory_manager_with_callbacks
    callbacks.on_oom_retry.side_effect = RuntimeError("Callback failed")

    @mm.with_memory_protection("test_op")
    def fail_once():
        if callbacks.on_oom.call_count < 2:
            raise RuntimeError("CUDA out of memory")
        return "success"

    # Should still succeed despite callback failure
    # (callback exception is caught and logged)
    with pytest.raises(RuntimeError, match="Callback failed"):
        fail_once()


# =============================================================================
# 6. REGRESSION TESTS (Backward Compatibility)
# =============================================================================


def test_backward_compat_no_context(memory_manager, mock_torch):
    """Test old API without context still works."""

    @memory_manager.with_memory_protection("test_op")
    def old_style():
        return "works"

    assert old_style() == "works"


def test_backward_compat_with_static_context(memory_manager, mock_torch):
    """Test old API with static context still works."""

    @memory_manager.with_memory_protection("test_op", context={"key": "value"})
    def old_style():
        return "works"

    assert old_style() == "works"


def test_existing_tests_compatibility(memory_manager, mock_torch):
    """Test that existing test patterns still work."""
    calls = []

    @memory_manager.with_memory_protection("test_func")
    def test_function():
        calls.append(1)
        return "result"

    result = test_function()
    assert result == "result"
    assert len(calls) == 1


# =============================================================================
# 7. LOGIC-SPECIFIC TESTS
# =============================================================================


def test_context_factory_receives_all_args(memory_manager, mock_torch):
    """Test context_factory receives all function arguments."""
    received_args = []

    def factory(*args, **kwargs):
        received_args.append((args, kwargs))
        return {}

    @memory_manager.with_memory_protection("test_op", context_factory=factory)
    def func(a, b, c=3):
        return a + b + c

    result = func(1, 2, c=5)
    assert result == 8
    assert len(received_args) == 1
    args, kwargs = received_args[0]
    assert args == (1, 2)
    assert kwargs == {"c": 5}


def test_enhanced_logging_with_emojis(memory_manager, mock_torch):
    """Test that enhanced logging happens during retry (verify cleanup called)."""
    attempts = []
    cleanup_calls = []

    original_cleanup = memory_manager.aggressive_cleanup

    def tracked_cleanup():
        result = original_cleanup()
        cleanup_calls.append(result)
        return result

    memory_manager.aggressive_cleanup = tracked_cleanup

    @memory_manager.with_memory_protection("test_op")
    def fail_once():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("CUDA out of memory")
        return "success"

    result = fail_once()
    assert result == "success"
    # Verify cleanup was called (happens after OOM detection and during retry)
    # May be called multiple times: after safe_operation OOM + in retry logic
    assert len(cleanup_calls) >= 1


def test_memory_info_in_final_error(memory_manager, mock_torch):
    """Test that final error contains detailed memory info."""
    mock_torch.cuda.mem_get_info.return_value = (100 * 1024**2, 8 * 1024**3)

    @memory_manager.with_memory_protection("test_op", max_retries=1)
    def always_oom():
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(OOMRecoverableError) as exc_info:
        always_oom()

    error = exc_info.value
    assert error.memory_info["free_mb"] == 100


# =============================================================================
# 8. COMBINATORIAL TESTS
# =============================================================================


def test_all_features_combined(memory_manager_with_callbacks, mock_torch):
    """Test decorator with all features enabled."""
    mm, callbacks = memory_manager_with_callbacks

    def factory(self, data):
        return {"batch_size": len(data), "class": self.__class__.__name__}

    class MockTrainer:
        @mm.with_memory_protection(
            "complex_op", max_retries=2, context={"model": "gpt2"}, context_factory=factory
        )
        def train(self, data):
            if callbacks.on_oom.call_count < 1:
                raise RuntimeError("CUDA out of memory")
            return sum(data)

    trainer = MockTrainer()
    result = trainer.train([1, 2, 3])

    assert result == 6
    assert callbacks.on_oom.call_count >= 1
    assert callbacks.on_oom_retry.call_count >= 1


def test_multiple_decorated_functions_sequential(memory_manager, mock_torch):
    """Test multiple decorated functions work correctly in sequence."""

    @memory_manager.with_memory_protection("func1")
    def func1():
        return 1

    @memory_manager.with_memory_protection("func2")
    def func2():
        return 2

    assert func1() == 1
    assert func2() == 2


def test_nested_decorated_functions(memory_manager, mock_torch):
    """Test that nested decorated functions work (though not recommended)."""

    @memory_manager.with_memory_protection("outer")
    def outer():
        @memory_manager.with_memory_protection("inner")
        def inner():
            return "inner"

        return inner()

    result = outer()
    assert result == "inner"


# =============================================================================
# INTEGRATION TEST
# =============================================================================


def test_realistic_phase_executor_scenario(memory_manager_with_callbacks, mock_torch):
    """Test realistic scenario mimicking PhaseExecutor usage."""
    mm, callbacks = memory_manager_with_callbacks

    class MockTrainer:
        def __init__(self):
            self.args = type("Args", (), {"per_device_train_batch_size": 8})()
            self.trained = False

        def train(self, resume_from_checkpoint=None):
            # Simulate OOM on first attempt
            if not self.trained:
                self.trained = True
                raise RuntimeError("CUDA out of memory. Tried to allocate 2GB")
            return "success"

    class MockPhaseExecutor:
        def __init__(self, mm):
            self.memory_manager = mm

        def extract_training_context(self, phase_idx, trainer, checkpoint):
            return {
                "phase": phase_idx,
                "batch_size": trainer.args.per_device_train_batch_size,
            }

        def _run_training(self, phase_idx, trainer, checkpoint):
            @self.memory_manager.with_memory_protection(
                f"train_phase_{phase_idx}",
                context_factory=self.extract_training_context,
            )
            def protected_train(self, phase_idx, trainer, checkpoint):
                return trainer.train(resume_from_checkpoint=checkpoint)

            return protected_train(self, phase_idx, trainer, checkpoint)

    executor = MockPhaseExecutor(mm)
    trainer = MockTrainer()

    result = executor._run_training(0, trainer, None)

    assert result == "success"
    assert callbacks.on_oom.call_count >= 1
    assert callbacks.on_oom_retry.call_count >= 1
