/**
 * Hooks for the Phase 7.2 Live Training MVP.
 *
 * The browser cannot open SSH tunnels itself, so these talk to the
 * server-side proxy in ``src/api/routers/jobs.py``: the FastAPI control
 * plane opens a short-lived tunnel + JobClient, calls the in-pod runner,
 * returns plain JSON. We poll instead of streaming via WebSocket — keeps
 * the UI MVP-simple and the server free of long-lived connections.
 *
 * Polling cadence:
 *  - status:  every 2 s while running, 10 s once terminal — same heuristic
 *             as ``useAttempt`` so the surrounding shell feels consistent.
 *  - events:  every 2 s; cursor (``since``) is held by the caller so we
 *             never re-fetch the full history.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '../client'

// ---------------------------------------------------------------------
// Types — narrow shape inferred from src/api/routers/jobs.py.
// ---------------------------------------------------------------------

/** State strings come from JobLifecycleFSM (preparing|running|stopping|completed|failed|cancelled). */
export type JobState =
  | 'preparing'
  | 'running'
  | 'stopping'
  | 'completed'
  | 'failed'
  | 'cancelled'

export interface JobSubmissionView {
  schema_version: number
  job_id: string
  provider_name: string
  pod_id: string | null
  ssh_host: string
  ssh_port: number
  ssh_username: string
  ssh_key_path: string | null
  created_at_iso: string
}

export interface JobSnapshot {
  job_id: string
  state: JobState
  sequence: number
  reason?: string | null
  detail?: Record<string, unknown> | null
  // Runner may include richer metrics; we keep this open-shape so the
  // UI can render unknown fields read-only.
  [key: string]: unknown
}

export interface JobStatusResponse {
  submission: JobSubmissionView
  snapshot: JobSnapshot
}

export interface JobEvent {
  offset: number
  kind: string
  payload?: Record<string, unknown>
  ts?: string | number
  [key: string]: unknown
}

export interface JobEventsResponse {
  events: JobEvent[]
  next_since: number
  error?: { code: string; message: string }
}

// ---------------------------------------------------------------------
// Query keys
// ---------------------------------------------------------------------

const jobKeys = {
  status: (runId: string, attempt: number | undefined) =>
    ['job', runId, 'status', attempt ?? 'latest'] as const,
  events: (runId: string, attempt: number | undefined, since: number) =>
    ['job', runId, 'events', attempt ?? 'latest', since] as const,
}

// ---------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------

export function useJobStatus(
  runId: string | undefined,
  attempt: number | undefined,
  enabled = true,
) {
  return useQuery<JobStatusResponse>({
    queryKey: runId ? jobKeys.status(runId, attempt) : ['disabled'],
    queryFn: () =>
      api.get<JobStatusResponse>(
        `/runs/${encodeURIComponent(runId!)}/job/status`,
        attempt !== undefined ? { attempt } : undefined,
      ),
    enabled: enabled && !!runId,
    refetchInterval: (query) => {
      const data = query.state.data as JobStatusResponse | undefined
      const state = data?.snapshot?.state
      if (!state) return 2_000
      return state === 'running' || state === 'preparing' || state === 'stopping'
        ? 2_000
        : 10_000
    },
  })
}

export function useJobEvents(
  runId: string | undefined,
  attempt: number | undefined,
  since: number,
  enabled = true,
) {
  return useQuery<JobEventsResponse>({
    queryKey: runId ? jobKeys.events(runId, attempt, since) : ['disabled'],
    queryFn: () =>
      api.get<JobEventsResponse>(
        `/runs/${encodeURIComponent(runId!)}/job/events`,
        {
          since,
          limit: 200,
          ...(attempt !== undefined ? { attempt } : {}),
        },
      ),
    enabled: enabled && !!runId,
    // Keep cursor-based pagination simple: each since-bucket is a fresh
    // key, and we drive the polling from the page itself.
    refetchInterval: 2_000,
  })
}

export function useStopJob(runId: string | undefined, attempt: number | undefined) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (grace?: number) =>
      api.post<JobSnapshot>(
        `/runs/${encodeURIComponent(runId!)}/job/stop` +
          buildQuery({ attempt, grace }),
      ),
    onSuccess: () => {
      if (runId) {
        qc.invalidateQueries({ queryKey: ['job', runId] })
      }
    },
  })
}

function buildQuery(params: Record<string, number | undefined>): string {
  const parts: string[] = []
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined) continue
    parts.push(`${key}=${encodeURIComponent(String(value))}`)
  }
  return parts.length === 0 ? '' : `?${parts.join('&')}`
}
