from __future__ import annotations

from src.pipeline.stages.constants import StageNames


def test_stage_names_is_str_compatible_for_context_keys() -> None:
    """
    StageNames is a StrEnum (str-compatible).

    This guarantees backward compatibility:
    - old code may use string keys
    - new code may use StageNames keys
    Both must be able to read the same context dict.
    """
    ctx_enum_key = {StageNames.GPU_DEPLOYER: {"ssh_host": "1.2.3.4"}}
    assert ctx_enum_key["GPU Deployer"]["ssh_host"] == "1.2.3.4"

    ctx_str_key = {"GPU Deployer": {"ssh_host": "1.2.3.4"}}
    assert ctx_str_key[StageNames.GPU_DEPLOYER]["ssh_host"] == "1.2.3.4"


