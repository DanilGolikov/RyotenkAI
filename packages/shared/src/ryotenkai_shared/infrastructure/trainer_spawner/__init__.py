"""Provider-agnostic ``ITrainerSpawner`` Protocol (definition-only in Phase 4)."""

from __future__ import annotations

from ryotenkai_shared.infrastructure.trainer_spawner.protocol import (
    ITrainerSpawner,
    TrainerEvent,
    TrainerHandle,
    TrainerSpawnError,
    TrainerSpec,
    TrainerStatus,
)

__all__ = [
    "ITrainerSpawner",
    "TrainerEvent",
    "TrainerHandle",
    "TrainerSpawnError",
    "TrainerSpec",
    "TrainerStatus",
]
