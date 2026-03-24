"""
RunPod Pods inference constants.

We intentionally keep base URLs and remote filesystem paths out of user configs
to reduce misconfiguration surface area.
"""

# Canonical RunPod REST API base URL (v1).
RUNPOD_REST_API_BASE_URL = "https://rest.runpod.io/v1"

# ---------------------------------------------------------------------------
# Remote filesystem layout inside RunPod inference pods
# ---------------------------------------------------------------------------
# All inference pods share the same container image and filesystem layout.
# Centralising these paths here ensures pod_session.py, provider.py, and
# the generated chat_inference.py script stay in sync without duplication.

POD_WORKSPACE = "/workspace"
POD_HF_CACHE_DIR = f"{POD_WORKSPACE}/hf_cache"
POD_RUNS_DIR = f"{POD_WORKSPACE}/runs"
POD_LOCK_DIR = f"{POD_WORKSPACE}/.helix_inference_lock"

# Absolute path to the LoRA merge helper script installed in the container image.
POD_MERGE_SCRIPT = "/opt/helix/merge_lora.py"  # noqa: WPS226


def pod_run_dir(run_key: str) -> str:
    """Return workspace subdirectory for a given run key (pod_id or run_name)."""
    return f"{POD_RUNS_DIR}/{run_key}"


def pod_merged_dir(run_key: str) -> str:
    """Return the directory where the merged (LoRA-applied) model is stored."""
    return f"{pod_run_dir(run_key)}/model"


def pod_pid_file(run_key: str) -> str:
    """Return the path to the vLLM PID file for a given run key."""
    return f"{pod_run_dir(run_key)}/vllm.pid"


def pod_log_file(run_key: str) -> str:
    """Return the path to the vLLM log file for a given run key."""
    return f"{pod_run_dir(run_key)}/vllm.log"


def pod_hash_file(run_key: str) -> str:
    """Return the path to the config hash file used for idempotent merge checks."""
    return f"{pod_run_dir(run_key)}/config_hash.txt"


# ---------------------------------------------------------------------------
# Session parameter dict keys (shared between provider.py and pod_session.py)
# ---------------------------------------------------------------------------
SESSION_KEY_BASE_MODEL_ID = "base_model_id"
SESSION_KEY_ADAPTER_REF = "adapter_ref"


__all__ = [
    "POD_HF_CACHE_DIR",
    "POD_LOCK_DIR",
    "POD_MERGE_SCRIPT",
    "POD_RUNS_DIR",
    "POD_WORKSPACE",
    "RUNPOD_REST_API_BASE_URL",
    "SESSION_KEY_ADAPTER_REF",
    "SESSION_KEY_BASE_MODEL_ID",
    "pod_hash_file",
    "pod_log_file",
    "pod_merged_dir",
    "pod_pid_file",
    "pod_run_dir",
]
