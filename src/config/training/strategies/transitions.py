"""
Strategy chain constraints.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from src.constants import (
    STRATEGY_COT,
    STRATEGY_CPT,
    STRATEGY_DPO,
    STRATEGY_GRPO,
    STRATEGY_ORPO,
    STRATEGY_SAPO,
    STRATEGY_SFT,
)

if TYPE_CHECKING:
    from .phase import StrategyPhaseConfig
    from src.utils.result import Result, StrategyError

# Valid strategy transitions (from → to)
# WPS407: use MappingProxyType for immutable module-level constant
VALID_STRATEGY_TRANSITIONS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        STRATEGY_CPT: (STRATEGY_SFT, STRATEGY_COT),
        STRATEGY_SFT: (STRATEGY_COT, STRATEGY_DPO, STRATEGY_ORPO, STRATEGY_GRPO, STRATEGY_SAPO),
        STRATEGY_COT: (STRATEGY_DPO, STRATEGY_ORPO, STRATEGY_GRPO, STRATEGY_SAPO),
        STRATEGY_DPO: (),
        STRATEGY_ORPO: (),
        STRATEGY_GRPO: (),
        STRATEGY_SAPO: (),
    }
)

# Starting strategies (can be first in chain)
# DPO is included: a pre-trained SFT checkpoint can be provided via model.name,
# making a standalone DPO run valid (same pattern as GRPO/SAPO).
VALID_START_STRATEGIES: tuple[str, ...] = (STRATEGY_CPT, STRATEGY_SFT, STRATEGY_ORPO, STRATEGY_GRPO, STRATEGY_SAPO, STRATEGY_DPO)


def _strategy_chain_error(message: str, code: str) -> Result[None, StrategyError]:
    from src.utils.result import Err, StrategyError

    return Err(StrategyError(message=message, code=code))


def validate_strategy_chain(strategies: list[StrategyPhaseConfig]) -> Result[None, StrategyError]:
    """
    Validate a chain of training strategies.

    Rules:
    - Invalid start strategies and invalid transitions emit warnings only
    - Chain must not be empty
    - Chain cannot contain None values
    - Datasets must still be unique across non-cached phases
    """
    # Local import to avoid heavy side-effects at module import time.
    from src.utils.logger import logger
    from src.utils.result import Ok

    if not strategies:
        logger.debug("[CFG:CHAIN_INVALID] reason=empty_chain")
        return _strategy_chain_error("Strategy chain cannot be empty", "STRATEGY_CHAIN_EMPTY")

    # FIX BUG-010: Check for None elements before accessing attributes
    if any(s is None for s in strategies):
        logger.debug("[CFG:CHAIN_INVALID] reason=contains_none")
        return _strategy_chain_error("Strategy chain cannot contain None values", "STRATEGY_CHAIN_CONTAINS_NONE")

    chain_str = " → ".join(s.strategy_type for s in strategies)
    logger.debug(f"[CFG:CHAIN_VALIDATING] chain={chain_str}")

    # Check first strategy
    first = strategies[0].strategy_type
    if first not in VALID_START_STRATEGIES:
        logger.warning(
            f"[CFG:CHAIN_WARNING] reason=invalid_start, got={first}, allowed={VALID_START_STRATEGIES}, "
            f"chain={chain_str}"
        )

    # Check transitions
    for i in range(len(strategies) - 1):
        current = strategies[i].strategy_type
        next_strategy = strategies[i + 1].strategy_type
        valid_next = VALID_STRATEGY_TRANSITIONS.get(current, ())

        if next_strategy not in valid_next:
            logger.warning(
                f"[CFG:CHAIN_WARNING] reason=invalid_transition, from={current}, to={next_strategy}, "
                f"valid={valid_next}, chain={chain_str}"
            )

    # Check dataset uniqueness across strategies
    # Phases with adapter_cache.enabled are excluded: they use their dataset only for
    # fingerprinting (cache validation), not for actual training data upload — no conflict.
    if len(strategies) > 1:
        seen_datasets: dict[str, str] = {}
        for phase in strategies:
            if hasattr(phase, "adapter_cache") and phase.adapter_cache.enabled:
                continue
            resolved = phase.dataset or "default"
            if resolved in seen_datasets:
                prev_type = seen_datasets[resolved]
                cur_type = phase.strategy_type
                logger.debug(
                    f"[CFG:CHAIN_INVALID] reason=duplicate_dataset, dataset={resolved}, "
                    f"strategies={prev_type}+{cur_type}"
                )
                return _strategy_chain_error(
                    (
                        f"Duplicate dataset '{resolved}': strategies '{prev_type}' and '{cur_type}' "
                        f"reference the same dataset. Each strategy must use its own dataset entry "
                        f"(different data formats are uploaded to separate remote paths)."
                    ),
                    "STRATEGY_CHAIN_DUPLICATE_DATASET",
                )
            seen_datasets[resolved] = phase.strategy_type

    logger.debug(f"[CFG:CHAIN_CHECKED] chain={chain_str}, phases={len(strategies)}")
    return Ok(None)


__all__ = [
    "VALID_START_STRATEGIES",
    "VALID_STRATEGY_TRANSITIONS",
    "validate_strategy_chain",
]
