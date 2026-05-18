"""Canonical Fake* implementations per Protocol.

Filled in Phase 1. Naming: ``Fake<ProtocolMinusI>``. One reference impl per
production Protocol; never a Mock; works in-process and behind sidecar HTTP.

MLflow narrow-Protocol fakes (target architecture, replaces wide
``FakeMLflowManager``):

* :class:`FakeTrackingClient` -- :class:`ITrackingClient`
* :class:`FakeMetricSink` -- :class:`IMetricSink`
* :class:`FakeArtifactSink` -- :class:`IArtifactSink`
* :class:`FakeRunQuery` -- :class:`IRunQuery`
* :class:`FakeModelRegistry` -- :class:`IModelRegistry`
* :class:`FakeJournalUploader` -- :class:`IJournalUploader`
* :class:`FakePromptRegistry` -- :class:`IPromptRegistry`
"""

from tests._fakes.mlflow_artifact_sink import (
    ChecksumMismatchError,
    FakeArtifactSink,
    TransientArtifactError,
    UploadCall,
)
from tests._fakes.mlflow_journal_uploader import (
    FakeJournalUploader,
    JournalConflictError,
    JournalUploadCall,
    TransientJournalError,
)
from tests._fakes.mlflow_metric_sink import (
    FakeMetricSink,
    LogCall,
    MetricSample,
    TransientMetricError,
)
from tests._fakes.mlflow_model_registry import (
    FakeModelRegistry,
    FakeModelVersion,
    RegisterCall,
    SetAliasCall,
    TransientRegistryError,
    UnknownAliasError,
    UnknownVersionError,
)
from tests._fakes.mlflow_prompt_registry import (
    FakePromptArtifact,
    FakePromptRegistry,
    PromptTimeoutError,
    TransientPromptError,
)
from tests._fakes.mlflow_run_query import (
    FakeRunQuery,
    SearchCall,
    SearchPredicate,
    TransientQueryError,
)
from tests._fakes.mlflow_tracking_client import (
    FakeTrackingClient,
    SetTagsCall,
    StartNestedRunCall,
    StartRunCall,
    TrackingUnavailableError,
    TransientTrackingError,
)

__all__ = [
    "ChecksumMismatchError",
    "FakeArtifactSink",
    "FakeJournalUploader",
    "FakeMetricSink",
    "FakeModelRegistry",
    "FakeModelVersion",
    "FakePromptArtifact",
    "FakePromptRegistry",
    "FakeRunQuery",
    "FakeTrackingClient",
    "JournalConflictError",
    "JournalUploadCall",
    "LogCall",
    "MetricSample",
    "PromptTimeoutError",
    "RegisterCall",
    "SearchCall",
    "SearchPredicate",
    "SetAliasCall",
    "SetTagsCall",
    "StartNestedRunCall",
    "StartRunCall",
    "TrackingUnavailableError",
    "TransientArtifactError",
    "TransientJournalError",
    "TransientMetricError",
    "TransientPromptError",
    "TransientQueryError",
    "TransientRegistryError",
    "TransientTrackingError",
    "UnknownAliasError",
    "UnknownVersionError",
    "UploadCall",
]
