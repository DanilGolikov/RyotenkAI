"""User-facing pod-lifecycle thresholds.

Optional section of :class:`PipelineConfig` that lets operators
override the two hard-coded watchdog thresholds baked into the pod's
:class:`IdleDetector`:

* ``max_lifetime_hours``      hard kill switch (default 48h, max 30d)
* ``idle_threshold_minutes``  GPU-idle window before stop (default 20m)

When the YAML is absent both values fall through to the pod-side
defaults (``DEFAULT_MAX_LIFETIME`` / ``DEFAULT_IDLE_THRESHOLD``).
The training launcher (``_build_job_env``) translates the optional
block into env vars consumed by the runner.
"""

from __future__ import annotations

from pydantic import Field

from .base import StrictBaseModel


class PodLifecycleConfig(StrictBaseModel):
    """Pod auto-shutdown thresholds (E-СРЕД config wire-through).

    Both fields are optional with defaults that mirror the pod-side
    in-process constants. ``gt=0`` / upper bounds prevent operators
    from disabling the watchdog or asking for absurd values
    (a 30-day max-lifetime is already absurd but matches the existing
    pod_lifecycle research note).
    """

    max_lifetime_hours: float = Field(
        default=48.0,
        gt=0,
        le=720.0,
        description=(
            "Hard kill switch — the runner stops the trainer after this "
            "many hours regardless of GPU activity. Capped at 30 days."
        ),
    )

    idle_threshold_minutes: float = Field(
        default=20.0,
        gt=0,
        le=60 * 24,
        description=(
            "Sustained GPU-idle window (in minutes) before the watchdog "
            "asks the supervisor to stop. Capped at 24h."
        ),
    )


__all__ = ["PodLifecycleConfig"]
