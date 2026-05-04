"""DTO round-trip contract tests (Phase 3 PR-3.4).

For every wire DTO that crosses Mac↔pod, assert
``DTO → JSON → DTO`` produces an identical object. Catches:

* Field accidentally typed as ``Any`` so JSON-deserialised value
  loses its Pydantic invariants.
* ``model_dump(exclude_none=True)`` stripping a field that the
  destination requires.
* ``mode="json"`` vs ``mode="python"`` mismatch — enums returning
  Python instances on one side and string values on the other.

This is the canonical test the OpenAPI client also has to pass
(plan §11 Phase 3 PR-3.4) — adding a new DTO without round-trip
coverage is a regression we want CI to surface immediately.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from ryotenkai_shared.contracts.problem_details import (
    ErrorCode,
    FieldError,
    ProblemDetails,
)
from ryotenkai_shared.contracts.runner_api import (
    ControlHeartbeatRequest,
    ControlHeartbeatResponse,
    DiagnosticsBlockError,
    DiagnosticsResponse,
    DmesgReport,
    EventResponse,
    FileUploadResponse,
    FileUploadTarget,
    GpuReport,
    GpuRow,
    ImportCheckReport,
    ImportCheckRequest,
    ImportResult,
    InternalEventRequest,
    JobSnapshotResponse,
    JobSpec,
    JobStopAcceptedResponse,
    JobSubmittedResponse,
    KernelSignalsReport,
    LogChunkResponse,
    LogName,
    LogSizeResponse,
    ResourceSnapshot,
)


def _round_trip(model: BaseModel) -> BaseModel:
    """Serialise via ``mode="json"`` (the wire shape — string enums,
    no Python repr leakage), drop nulls per RFC 9457 §3.1, then
    re-parse from JSON. Mirrors the actual httpx ``response.json()``
    path the Mac-side parser uses."""
    payload = model.model_dump(mode="json", exclude_none=True)
    wire_bytes = json.dumps(payload).encode()
    return type(model).model_validate(json.loads(wire_bytes))


# ---------------------------------------------------------------------------
# Fixtures: one populated instance per DTO class
# ---------------------------------------------------------------------------


_FIXTURES: dict[type[BaseModel], BaseModel] = {
    # ---- problem_details
    ProblemDetails: ProblemDetails(
        title="Job not found", status=404,
        detail="job xyz", instance="/api/v1/jobs/xyz",
        code=ErrorCode.JOB_NOT_FOUND, trace_id="abc12345",
        request_id="req-1",
    ),
    FieldError: FieldError(
        loc=["body", "command", 0], type="missing", msg="Field required",
    ),
    # ---- jobs
    JobSpec: JobSpec(
        job_id="r-1", command=["python", "-c", "pass"],
        env={"K": "V"}, workdir="/workspace",
    ),
    JobSubmittedResponse: JobSubmittedResponse(
        job_id="r-1", sequence=0, offset=0,
    ),
    JobSnapshotResponse: JobSnapshotResponse(
        job_id="r-1", state="running", sequence=2,
        started_at="2026-05-04T00:00:00Z",
        updated_at="2026-05-04T00:00:01Z",
        message="ok", last_event_offset=5,
    ),
    JobStopAcceptedResponse: JobStopAcceptedResponse(
        job_id="r-1", state="stopping", sequence=3,
    ),
    # ---- events / internal / control
    EventResponse: EventResponse(
        offset=0, timestamp="2026-05-04T00:00:00Z",
        kind="step", payload={"loss": 0.5},
    ),
    InternalEventRequest: InternalEventRequest(
        kind="step", payload={"loss": 0.5},
    ),
    ControlHeartbeatRequest: ControlHeartbeatRequest(ttl_seconds=120.0),
    ControlHeartbeatResponse: ControlHeartbeatResponse(
        ok=True, ttl_seconds_applied=120.0,
    ),
    # ---- diagnostics
    DmesgReport: DmesgReport(
        lines=["a", "b"], truncated=True,
        error=DiagnosticsBlockError.PERMISSION_DENIED,
    ),
    KernelSignalsReport: KernelSignalsReport(
        matches=["nvrm xid 31"], truncated=False,
    ),
    GpuRow: GpuRow(
        name="RTX 4090", utilization_gpu_percent=78,
        memory_used_mib=12345, memory_total_mib=24576,
    ),
    GpuReport: GpuReport(rows=[]),
    DiagnosticsResponse: DiagnosticsResponse(
        dmesg=DmesgReport(lines=["x"]),
        gpu=GpuReport(rows=[]),
    ),
    # ---- resources
    ResourceSnapshot: ResourceSnapshot(
        gpu_util_percent=78.0, gpu_memory_percent=50.0,
        cpu_percent=12.5, ram_used_gb=8.0, ram_total_gb=32.0,
    ),
    # ---- logs
    LogChunkResponse: LogChunkResponse(
        content="some log content\n", total_size=1024,
        next_offset=18, truncated=True,
    ),
    LogSizeResponse: LogSizeResponse(size_bytes=1024),
    # ---- files
    FileUploadResponse: FileUploadResponse(
        target=FileUploadTarget.CONFIG,
        bytes_written=128,
        sha256="0" * 64,
        path="/workspace/config/pipeline_config.yaml",
    ),
    # ---- runtime
    ImportResult: ImportResult(
        module="ryotenkai_shared", importable=True,
    ),
    ImportCheckRequest: ImportCheckRequest(
        modules=["ryotenkai_shared", "ryotenkai_pod.runner.main"],
    ),
    ImportCheckReport: ImportCheckReport(
        results=[
            ImportResult(module="json", importable=True),
            ImportResult(module="missing_xyz", importable=False, error="ModuleNotFoundError: ..."),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,instance", _FIXTURES.items(), ids=lambda v: getattr(v, "__name__", "instance"),
)
def test_round_trip_preserves_object(
    cls: type[BaseModel], instance: BaseModel,
) -> None:
    """Each DTO must survive ``→ JSON → DTO`` losslessly."""
    restored = _round_trip(instance)
    assert restored == instance, (
        f"{cls.__name__} round-trip lost data:\n"
        f"  before: {instance.model_dump(mode='json', exclude_none=True)}\n"
        f"  after:  {restored.model_dump(mode='json', exclude_none=True)}"
    )


# ---------------------------------------------------------------------------
# Enum round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", list(ErrorCode))
def test_error_code_serialises_as_string(code: ErrorCode) -> None:
    """Each ErrorCode value, when shipped via ProblemDetails through
    ``mode="json"``, comes through as its string value (not Python
    repr)."""
    p = ProblemDetails(title="t", status=500, code=code)
    payload = p.model_dump(mode="json", exclude_none=True)
    assert payload["code"] == code.value
    assert isinstance(payload["code"], str)


# ---------------------------------------------------------------------------
# Coverage check — every public DTO listed in __all__ has a fixture
# ---------------------------------------------------------------------------


def test_every_runner_dto_has_round_trip_coverage() -> None:
    """If a new DTO is added to ``shared.contracts.runner_api`` but
    no fixture is added to ``_FIXTURES``, this test fails — a CI gate
    against silent contract gaps."""
    import ryotenkai_shared.contracts.runner_api as runner_api

    # Every name in __all__ that resolves to a Pydantic model class.
    public_models: set[type[BaseModel]] = set()
    for name in runner_api.__all__:
        attr = getattr(runner_api, name)
        if isinstance(attr, type) and issubclass(attr, BaseModel):
            public_models.add(attr)

    covered = set(_FIXTURES.keys())
    missing = public_models - covered
    assert not missing, (
        "DTO without round-trip coverage:\n  "
        + "\n  ".join(c.__name__ for c in missing)
        + "\nAdd a fixture to _FIXTURES."
    )
