"""REST endpoints for job lifecycle — Phase 0 placeholder.

Routes (final shape, Phase 1 implements):
- ``POST /jobs`` — multipart submit (job_spec + plugins_payload). 202 on accept.
- ``GET /jobs/{id}`` — current snapshot (state, started_at, last event offset).
- ``POST /jobs/{id}/stop`` — request graceful stop. 202 on accept.
- ``GET /jobs/{id}/log?tail=N&follow=true`` — chunked transfer of ``training.log``.

Phase 0 ships only health / readiness so the entrypoint can boot.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/_skeleton", include_in_schema=False)
def _skeleton() -> dict[str, str]:
    """Phase 0 placeholder — confirms the router is mounted.

    Hidden from OpenAPI; will be removed when Phase 1 lands real endpoints.
    """
    return {"status": "skeleton"}
