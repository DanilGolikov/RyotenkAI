"""Closed discriminated union assembly and cached :class:`TypeAdapter`.

The ``Event`` alias is the union of every concrete event class in
:mod:`ryotenkai_shared.events.types` plus the forward-compat
:class:`~ryotenkai_shared.events.types.unknown.UnknownEvent` variant. The
discriminator is the envelope's ``type`` field, which every subclass
pins via ``Literal[...]``.

A single :class:`pydantic.TypeAdapter` is cached at import time —
constructing the adapter is the expensive part of validation, so we pay
it once. :data:`EVENT_ADAPTER.validate_python` dispatches by discriminator
in O(1) for known types and raises :class:`pydantic.ValidationError` for
unknowns (the codec wraps to :class:`UnknownEvent` in non-strict mode).
"""

from __future__ import annotations

from typing import Annotated, Union

from pydantic import Discriminator, TypeAdapter

from ryotenkai_shared.events.types import (
    DatasetValidationCompletedEvent,
    DatasetValidationFailedEvent,
    DatasetValidationPluginCompletedEvent,
    DatasetValidationPluginFailedEvent,
    DatasetValidationPluginStartedEvent,
    DatasetValidationStartedEvent,
    EvaluationCompletedEvent,
    EvaluationPluginCompletedEvent,
    EvaluationPluginFailedEvent,
    EvaluationPluginStartedEvent,
    EvaluationStartedEvent,
    GPUCodeSyncedEvent,
    GPUDeploymentCompletedEvent,
    GPUDeploymentFailedEvent,
    GPUDeploymentStartedEvent,
    GPUPreemptedEvent,
    GPUSSHProvisionedEvent,
    GpuCleanupCompletedEvent,
    GpuCleanupFailedEvent,
    GpuCleanupStartedEvent,
    HealthIdleDetectedEvent,
    HealthMaxLifetimeExceededEvent,
    HealthMaxLifetimeReachedEvent,
    HealthSnapshotEvent,
    InferenceDeactivatedEvent,
    InferenceDeployedEvent,
    InferenceDeploymentFailedEvent,
    InferenceDeploymentStartedEvent,
    InferenceHealthCheckCompletedEvent,
    InferenceHealthCheckStartedEvent,
    JobSubmittedEvent,
    JournalDiskPressureEvent,
    JournalRotatedEvent,
    MemoryCacheClearedEvent,
    MemoryOOMDetectedEvent,
    MemoryPressureWarningEvent,
    MemoryThresholdReachedEvent,
    MetricsBufferOversizedEvent,
    ModelMetricsBufferRetrievedEvent,
    ModelRetrievalCompletedEvent,
    ModelRetrievalStartedEvent,
    PluginsUnpackedEvent,
    RunCancelledEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunnerShutdownEvent,
    RunnerStartedEvent,
    RunStartedEvent,
    StageCompletedEvent,
    StageFailedEvent,
    StageInterruptedEvent,
    StageSkippedEvent,
    StageStartedEvent,
    StopRequestedEvent,
    TrainerExitedEvent,
    TrainerSpawnedEvent,
    TrainerSpawnFailedEvent,
    TrainerStderrEvent,
    TrainerStdoutEvent,
    TrainingCheckpointSavedEvent,
    TrainingCompletedEvent,
    TrainingEpochCompletedEvent,
    TrainingEpochStartedEvent,
    TrainingEvalMetricsEvent,
    TrainingFailedEvent,
    TrainingLogEvent,
    TrainingMonitorStartedEvent,
    TrainingMonitorTimeoutEvent,
    TrainingStartedEvent,
    TrainingStepEvent,
    UnknownEvent,
)

# Closed discriminated union of every known event class. Grouped by
# domain (pod lifecycle / training / memory / health / io, control run /
# stage / dataset / gpu / training-monitor / model / evaluation /
# inference, plus UnknownEvent for forward-compat). One class per line so
# the diff for a new event type is a single insertion, and so the
# discriminator key set is grep-able. We use ``Union[...]`` rather than
# the ``|`` operator because the type checker output for 50+ ``|``
# operands is noisy and Pydantic's discriminator logic is identical for
# both forms.
Event = Annotated[
    Union[  # noqa: UP007 — Union[] gives a readable per-class-per-line layout for 50+ members; `|` chains scale poorly.
        # Pod lifecycle
        RunnerStartedEvent,
        RunnerShutdownEvent,
        JobSubmittedEvent,
        TrainerSpawnedEvent,
        TrainerSpawnFailedEvent,
        TrainerExitedEvent,
        StopRequestedEvent,
        PluginsUnpackedEvent,
        # Pod journal
        JournalRotatedEvent,
        JournalDiskPressureEvent,
        # Pod training
        TrainingStartedEvent,
        TrainingEpochStartedEvent,
        TrainingEpochCompletedEvent,
        TrainingStepEvent,
        TrainingLogEvent,
        TrainingEvalMetricsEvent,
        TrainingCheckpointSavedEvent,
        TrainingCompletedEvent,
        TrainingFailedEvent,
        # Pod memory
        MemoryCacheClearedEvent,
        MemoryOOMDetectedEvent,
        MemoryPressureWarningEvent,
        MemoryThresholdReachedEvent,
        # Pod health
        HealthSnapshotEvent,
        HealthIdleDetectedEvent,
        HealthMaxLifetimeReachedEvent,
        HealthMaxLifetimeExceededEvent,
        # Pod IO
        TrainerStdoutEvent,
        TrainerStderrEvent,
        # Control run
        RunStartedEvent,
        RunCompletedEvent,
        RunFailedEvent,
        RunCancelledEvent,
        # Control stage
        StageStartedEvent,
        StageCompletedEvent,
        StageFailedEvent,
        StageSkippedEvent,
        StageInterruptedEvent,
        # Control dataset
        DatasetValidationStartedEvent,
        DatasetValidationCompletedEvent,
        DatasetValidationFailedEvent,
        DatasetValidationPluginStartedEvent,
        DatasetValidationPluginCompletedEvent,
        DatasetValidationPluginFailedEvent,
        # Control GPU
        GPUDeploymentStartedEvent,
        GPUDeploymentCompletedEvent,
        GPUDeploymentFailedEvent,
        GPUPreemptedEvent,
        GPUSSHProvisionedEvent,
        GPUCodeSyncedEvent,
        GpuCleanupStartedEvent,
        GpuCleanupCompletedEvent,
        GpuCleanupFailedEvent,
        # Control training (monitor)
        TrainingMonitorStartedEvent,
        TrainingMonitorTimeoutEvent,
        # Control model
        ModelRetrievalStartedEvent,
        ModelRetrievalCompletedEvent,
        ModelMetricsBufferRetrievedEvent,
        MetricsBufferOversizedEvent,
        # Control evaluation
        EvaluationStartedEvent,
        EvaluationPluginStartedEvent,
        EvaluationPluginCompletedEvent,
        EvaluationPluginFailedEvent,
        EvaluationCompletedEvent,
        # Control inference
        InferenceDeploymentStartedEvent,
        InferenceHealthCheckStartedEvent,
        InferenceHealthCheckCompletedEvent,
        InferenceDeployedEvent,
        InferenceDeactivatedEvent,
        InferenceDeploymentFailedEvent,
        # Forward-compat catch-all
        UnknownEvent,
    ],
    Discriminator("kind"),
]


# Cache the adapter once at import time. Constructing it is ~100x more
# expensive than calling `.validate_python()`.
EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)


__all__ = ["EVENT_ADAPTER", "Event"]
