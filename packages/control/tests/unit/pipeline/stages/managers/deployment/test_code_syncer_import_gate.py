"""Unit tests for CodeSyncer's post-sync importability gate (PR-A).

Covers the seven test categories from the plan
(``docs/plans/2026-05-02-fail-fast-prevention-and-log-visibility.md`` §7):

* Positive — gate passes on a clean OK response.
* Negative — gate blocks deployment when ``src.providers`` is missing
  (regression for the 2026-05-02 15-crash incident) and on syntax
  errors.
* Boundary — gate parses GATE_RC trailer correctly.
* Invariants — gate uses ``/opt/helix/runtime_check.py --check-source``
  and the same workspace as PYTHONPATH.
* Dependency-error — runtime_check.py missing → actionable rebuild
  message; SSH timeout → not silently masked as success.
* Regression — rsync rc=0 + missing src/providers/ end-to-end.
* Combinatorial covered by the existing decision tests; here we exercise
  the parser exhaustively.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.stages.managers.deployment.code_syncer import (
    RUNTIME_CHECK_SCRIPT,
    CodeSyncer,
)
from src.config import (
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceLocal,
    GlobalHyperparametersConfig,
    InferenceConfig,
    InferenceEnginesConfig,
    InferenceVLLMEngineConfig,
    ModelConfig,
    PipelineConfig,
    QLoRAConfig,
    TrainingOnlyConfig,
)
from src.utils.result import Ok

pytestmark = pytest.mark.unit


DATASET_CHAT_FIXTURE = "src/tests/fixtures/datasets/test_chat.jsonl"

SINGLE_NODE_PROVIDER_CFG: dict = {
    "connect": {"ssh": {"alias": "pc"}},
    "training": {"workspace_path": "/tmp/workspace"},
}


@dataclass(frozen=True)
class DummySecrets:
    hf_token: str = "hf_test_token"


@pytest.fixture
def secrets() -> DummySecrets:
    return DummySecrets()


@pytest.fixture
def base_config() -> PipelineConfig:
    return PipelineConfig(
        model=ModelConfig(name="gpt2", torch_dtype="bfloat16", trust_remote_code=False),
        providers={"single_node": SINGLE_NODE_PROVIDER_CFG},
        training=TrainingOnlyConfig(
            provider="single_node",
            type="qlora",
            qlora=QLoRAConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules="all-linear",
                use_dora=False,
                use_rslora=False,
                init_lora_weights="gaussian",
            ),
            hyperparams=GlobalHyperparametersConfig(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=2e-4,
                warmup_ratio=0.0,
                epochs=1,
            ),
        ),
        datasets={
            "default": DatasetConfig(
                source_type="local",
                source_local=DatasetSourceLocal(
                    local_paths=DatasetLocalPaths(train=DATASET_CHAT_FIXTURE, eval=None)
                ),
            )
        },
        inference=InferenceConfig(
            enabled=False,
            provider="single_node",
            engine="vllm",
            engines=InferenceEnginesConfig(vllm=InferenceVLLMEngineConfig()),
        ),
    )


@pytest.fixture
def syncer(base_config: PipelineConfig, secrets: DummySecrets) -> CodeSyncer:
    s = CodeSyncer(config=base_config, secrets=secrets)
    s.set_workspace("/workspace")
    return s


def _make_ssh(stdout: str = "OK", *, success: bool = True, stderr: str = "") -> MagicMock:
    """Build a mock SSH client that returns the same triple for every command."""
    ssh = MagicMock()
    ssh.ssh_base_opts = None
    ssh._is_alias_mode = True
    ssh.ssh_target = "pc"
    ssh.key_path = ""
    ssh.port = 22
    ssh.exec_command.return_value = (success, stdout, stderr)
    return ssh


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


def test_gate_passes_on_clean_ok(syncer: CodeSyncer):
    """When the pod returns a clean OK manifest, the gate returns Ok and
    sync() reports a success message including '+ importable'."""
    ok_manifest = "OK\nsrc.providers=importable\nsrc.training.run_training=importable"
    ssh = _make_ssh(stdout=ok_manifest)

    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = ""
    completed.stderr = ""

    with patch("src.pipeline.stages.managers.deployment.code_syncer.subprocess.run", return_value=completed):
        result = syncer.sync(ssh)

    assert result.is_ok()


def test_verify_importability_direct_ok(syncer: CodeSyncer):
    """Direct call to _verify_importability with a clean OK response → Ok."""
    ssh = _make_ssh(stdout="OK\nsrc.providers=importable")
    result = syncer._verify_importability(ssh)
    assert result.is_ok()


# ---------------------------------------------------------------------------
# Negative — regression for the 15-crash incident
# ---------------------------------------------------------------------------


def test_gate_blocks_when_providers_missing_regression(syncer: CodeSyncer):
    """Regression for run_20260502_101003_6uebn (15 consecutive crashes):
    rsync rc=0, but ``src.providers`` not importable on pod → gate
    blocks Stage 1 with named module + actionable error."""
    failed_manifest = (
        "FAILED\n"
        "src.workspace.integrations.loader=importable\n"
        "src.config=importable\n"
        "src.providers=NOT_IMPORTABLE (ModuleNotFoundError: No module named 'src.providers')\n"
        "GATE_RC=2"
    )
    ssh = _make_ssh(stdout=failed_manifest)

    result = syncer._verify_importability(ssh)

    assert result.is_err()
    err = result.unwrap_err()
    assert err.code == "IMPORT_GATE_FAILED"
    assert "src.providers" in err.message
    # operator-friendly action present
    assert "ensure" in err.message.lower()
    # details preserved for forensics
    assert err.details is not None
    assert err.details["failed_modules"] == ["src.providers"]


def test_gate_blocks_on_syntax_error(syncer: CodeSyncer):
    """A SyntaxError surfaces as NOT_IMPORTABLE just like a missing module."""
    failed_manifest = (
        "FAILED\n"
        "src.providers=NOT_IMPORTABLE (SyntaxError: invalid syntax (foo.py, line 42))\n"
        "GATE_RC=2"
    )
    ssh = _make_ssh(stdout=failed_manifest)
    result = syncer._verify_importability(ssh)
    assert result.is_err()
    assert result.unwrap_err().code == "IMPORT_GATE_FAILED"


def test_gate_blocks_when_multiple_modules_fail(syncer: CodeSyncer):
    """Multiple failing modules end up in the failed_modules list."""
    failed_manifest = (
        "FAILED\n"
        "src.config=NOT_IMPORTABLE (ImportError: ...)\n"
        "src.providers=NOT_IMPORTABLE (ModuleNotFoundError: ...)\n"
        "GATE_RC=2"
    )
    ssh = _make_ssh(stdout=failed_manifest)
    result = syncer._verify_importability(ssh)
    assert result.is_err()
    assert sorted(result.unwrap_err().details["failed_modules"]) == ["src.config", "src.providers"]


# ---------------------------------------------------------------------------
# Boundary — GATE_RC parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("OK", None),
        ("OK\nsrc.providers=importable", None),
        ("FAILED\nsrc.providers=NOT_IMPORTABLE\nGATE_RC=2", 2),
        ("FAILED\nGATE_RC=127", 127),
        ("garbage\nGATE_RC=2\ntrailing", 2),  # trailer mid-output still parsed
        ("GATE_RC=not_an_int", None),  # malformed value tolerated
        ("", None),
        ("no trailer at all", None),
    ],
)
def test_extract_gate_rc_parses_trailer(output: str, expected: int | None):
    assert CodeSyncer._extract_gate_rc(output) == expected


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_gate_invokes_runtime_check_with_check_source(syncer: CodeSyncer):
    """Invariant: the gate runs ``/opt/helix/runtime_check.py --check-source``
    inside the workspace with PYTHONPATH set to the workspace. This is what
    keeps the gate validating the same import path the trainer uses."""
    ssh = _make_ssh(stdout="OK")
    syncer._verify_importability(ssh)

    cmd = ssh.exec_command.call_args.kwargs["command"]
    assert RUNTIME_CHECK_SCRIPT in cmd
    assert "--check-source" in cmd
    assert "/workspace" in cmd
    assert "PYTHONPATH=" in cmd


def test_gate_uses_python_verify_timeout(syncer: CodeSyncer):
    """Gate command runs with the same timeout as the runtime contract
    check (DEPLOYMENT_PYTHON_VERIFY_TIMEOUT), not the short SSH command
    timeout — heavy imports (torch, transformers) can take 30+ seconds."""
    from src.pipeline.stages.managers.deployment_constants import (
        DEPLOYMENT_PYTHON_VERIFY_TIMEOUT,
    )

    ssh = _make_ssh(stdout="OK")
    syncer._verify_importability(ssh)
    assert ssh.exec_command.call_args.kwargs["timeout"] == DEPLOYMENT_PYTHON_VERIFY_TIMEOUT


# ---------------------------------------------------------------------------
# Dependency-error
# ---------------------------------------------------------------------------


def test_gate_returns_unavailable_when_runtime_check_missing(syncer: CodeSyncer):
    """rc=127 → actionable 'rebuild image' error, not silent skip."""
    output = (
        "/bin/sh: /opt/helix/runtime_check.py: No such file or directory\n"
        "GATE_RC=127"
    )
    ssh = _make_ssh(stdout=output)
    result = syncer._verify_importability(ssh)
    assert result.is_err()
    err = result.unwrap_err()
    assert err.code == "IMPORT_GATE_UNAVAILABLE"
    assert "rebuild" in err.message.lower()


def test_gate_handles_ssh_failure_without_masking_as_passed(syncer: CodeSyncer):
    """SSH-level failure (success=False) → Err, not silently masked as Ok."""
    ssh = _make_ssh(stdout="", success=False, stderr="ssh: connection timed out")
    result = syncer._verify_importability(ssh)
    assert result.is_err()
    err = result.unwrap_err()
    assert err.code == "IMPORT_GATE_ERROR"
    assert "ssh" in err.message.lower() or "unexpected" in err.message.lower()


def test_gate_handles_empty_output_as_error(syncer: CodeSyncer):
    """Empty stdout with success=True (highly unlikely but possible) is
    treated as an unexpected result, NOT as an implicit pass."""
    ssh = _make_ssh(stdout="", success=True)
    result = syncer._verify_importability(ssh)
    assert result.is_err()
    assert result.unwrap_err().code == "IMPORT_GATE_ERROR"


# ---------------------------------------------------------------------------
# Regression — full sync() path with gate failure
# ---------------------------------------------------------------------------


def test_sync_returns_err_when_gate_fails_after_successful_rsync(syncer: CodeSyncer):
    """End-to-end regression: rsync rc=0 (success) → gate fails → sync()
    propagates the gate failure so the deployment pipeline halts at
    Stage 1 instead of proceeding to Training Monitor and crashing."""
    failed_manifest = (
        "FAILED\n"
        "src.providers=NOT_IMPORTABLE (ModuleNotFoundError)\n"
        "GATE_RC=2"
    )
    ssh = _make_ssh(stdout=failed_manifest)

    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = ""
    completed.stderr = ""

    with patch("src.pipeline.stages.managers.deployment.code_syncer.subprocess.run", return_value=completed):
        result = syncer.sync(ssh)

    assert result.is_err()
    err = result.unwrap_err()
    assert err.code == "IMPORT_GATE_FAILED"


def test_sync_returns_err_when_gate_fails_in_tar_fallback_path(syncer: CodeSyncer):
    """Same gate enforcement on the per-module tar fallback path —
    rsync failed, all tars succeeded, but importability still wrong."""
    failed_manifest = "FAILED\nsrc.providers=NOT_IMPORTABLE (ModuleNotFoundError)\nGATE_RC=2"
    ssh = _make_ssh(stdout=failed_manifest)

    rsync_failing = MagicMock()
    rsync_failing.returncode = 1
    rsync_failing.stdout = ""
    rsync_failing.stderr = "rsync failed"

    with (
        patch("src.pipeline.stages.managers.deployment.code_syncer.subprocess.run", return_value=rsync_failing),
        patch.object(syncer, "_sync_module_tar", return_value=Ok(None)),
    ):
        result = syncer.sync(ssh)

    assert result.is_err()
    assert result.unwrap_err().code == "IMPORT_GATE_FAILED"
