"""Cross-package wire contracts (DTOs, error shapes).

Lives in the leaf ``ryotenkai_shared`` package so both pod-side
(in-pod runner) and Mac-side (control orchestrator + JobClient) can
import the same Pydantic models without crossing the
``control → pod`` import boundary.

Layout (Phase 0 — transport unification v2):
    contracts/
    ├── runner_api/      runner HTTP+WS surface DTOs
    │   ├── _strict.py   ``_StrictModel`` base (extra="forbid")
    │   ├── jobs.py      JobSpec, JobSnapshotResponse, ...
    │   ├── events.py    EventResponse, WS close codes
    │   ├── internal.py  InternalEventRequest (loopback only)
    │   └── control.py   ControlHeartbeatRequest/Response
    └── (Phase 1) problem_details.py  RFC 9457
"""
