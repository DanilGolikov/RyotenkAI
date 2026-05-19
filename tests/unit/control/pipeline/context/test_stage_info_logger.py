"""Unit tests for the :class:`StageInfoLogger` no-op shim.

After the wide ``IMLflowManager`` retirement,
:class:`ryotenkai_control.pipeline.context.stage_info_logger.StageInfoLogger`
became a no-op shim retained only for bootstrap-wiring compatibility (see
its module docstring). These tests pin the no-op contract so a future
revival of the wide-manager surface fails loudly here first.
"""

from __future__ import annotations

import pytest

from ryotenkai_control.pipeline.context.stage_info_logger import StageInfoLogger
from ryotenkai_control.pipeline.stages import StageNames


@pytest.fixture
def logger_under_test() -> StageInfoLogger:
    return StageInfoLogger()


def test_log_does_not_raise_with_empty_context(
    logger_under_test: StageInfoLogger,
) -> None:
    """Calling log with an empty context must not raise."""
    logger_under_test.log(context={}, stage_name=StageNames.GPU_DEPLOYER)


def test_log_does_not_raise_with_populated_context(
    logger_under_test: StageInfoLogger,
) -> None:
    """Calling log with a populated stage context must not raise."""
    logger_under_test.log(
        context={
            StageNames.GPU_DEPLOYER: {
                "provider_name": "runpod",
                "provider_type": "cloud",
            }
        },
        stage_name=StageNames.GPU_DEPLOYER,
    )


def test_log_does_not_raise_for_unknown_stage(
    logger_under_test: StageInfoLogger,
) -> None:
    """Calling log with an unknown stage name must not raise."""
    logger_under_test.log(context={"NoSuchStage": {}}, stage_name="NoSuchStage")
