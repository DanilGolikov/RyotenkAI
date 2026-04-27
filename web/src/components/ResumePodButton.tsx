/**
 * Phase 11.C-2 — Resume pod button + status feedback.
 *
 * Displayed on RunDetail / RunsList when a run's latest attempt has
 * `pod_status === 'stopped'` (Phase 11.B's natural-completion sleep
 * outcome). One click invokes POST /runs/{id}/resume-pod which:
 *
 *   1. Probes RunPod for current pod state (Phase 11.C-1 probe).
 *   2. If sleeping, calls resume_pod_with_retry (5min capacity-aware
 *      backoff).
 *   3. Returns ok=true once the pod is RUNNING again, or
 *      ok=false with an actionable message (capacity exhausted /
 *      pod terminated / RunPod outage).
 *
 * After a successful resume the operator can hit the standard Launch
 * affordance to continue the pipeline. We deliberately don't chain
 * launch into the same click — separation of concerns makes the
 * failure mode clearer (operator sees "pod woke up" before deciding
 * whether to also re-spawn the pipeline).
 */

import { useState } from 'react'
import { useResumePod, type ResumePodResponse } from '../api/hooks/useLaunch'

export interface ResumePodButtonProps {
  runId: string
  /**
   * Render-mode hint for the parent. ``"compact"`` is suitable for
   * row-level badges in RunsList; ``"full"`` is for RunDetail's
   * action area.
   */
  variant?: 'compact' | 'full'
  /** Called after a successful resume so parent can re-fetch state. */
  onResumed?: () => void
}

export function ResumePodButton({
  runId,
  variant = 'full',
  onResumed,
}: ResumePodButtonProps): JSX.Element {
  const [feedback, setFeedback] = useState<ResumePodResponse | null>(null)
  const mutation = useResumePod(runId)

  const handleClick = async () => {
    setFeedback(null)
    try {
      const response = await mutation.mutateAsync()
      setFeedback(response)
      if (response.ok) {
        onResumed?.()
      }
    } catch (err) {
      setFeedback({
        availability: 'probe_failed',
        ok: false,
        message: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const label = mutation.isPending
    ? 'Resuming pod (≤5min)...'
    : variant === 'compact'
      ? 'Resume'
      : 'Resume pod'

  return (
    <div className={`resume-pod resume-pod--${variant}`}>
      <button
        type="button"
        onClick={handleClick}
        disabled={mutation.isPending}
        aria-label={`Resume sleeping pod for run ${runId}`}
      >
        {label}
      </button>
      {feedback !== null && (
        <span
          className={`resume-pod__feedback resume-pod__feedback--${
            feedback.ok ? 'ok' : 'err'
          }`}
          role="status"
        >
          {feedback.message}
        </span>
      )}
    </div>
  )
}

/**
 * Phase 11.C-2 — sleeping-pod badge shown next to status pill.
 *
 * Pure presentational; reads ``pod_status`` from the run summary and
 * renders ``"(stopped)"`` when applicable. Stays silent for
 * RUNNING / TERMINATED / legacy attempts (no pod_metadata) so the
 * UI doesn't get noisy.
 */
export interface PodStatusBadgeProps {
  podStatus: string | null | undefined
}

export function PodStatusBadge({
  podStatus,
}: PodStatusBadgeProps): JSX.Element | null {
  if (podStatus !== 'stopped') {
    return null
  }
  return (
    <span
      className="pod-status-badge pod-status-badge--stopped"
      title="Pod is sleeping (Phase 11.B podStop). Use Resume to wake it."
    >
      stopped
    </span>
  )
}
