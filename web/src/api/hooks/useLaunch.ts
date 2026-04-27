import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import { qk } from '../queryKeys'
import type { InterruptResponse, LaunchRequestBody, LaunchResponse } from '../types'

export function useLaunch(runId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: LaunchRequestBody) =>
      api.post<LaunchResponse>(`/runs/${encodeURIComponent(runId)}/launch`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.run(runId) })
      qc.invalidateQueries({ queryKey: qk.runs() })
    },
  })
}

export function useInterrupt(runId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.post<InterruptResponse>(`/runs/${encodeURIComponent(runId)}/interrupt`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.run(runId) })
    },
  })
}

/**
 * Phase 11.C-2 — wake a sleeping pod for the given run.
 *
 * Calls the new POST /runs/{id}/resume-pod endpoint. The endpoint
 * always returns 200 with an `ok` field; a non-ok response means
 * capacity exhausted or pod gone — the UI should surface the
 * `message` to the user rather than treat it as a 5xx error.
 */
export interface ResumePodResponse {
  availability: string  // PodAvailability enum value
  ok: boolean
  message: string
}

export function useResumePod(runId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () =>
      api.post<ResumePodResponse>(
        `/runs/${encodeURIComponent(runId)}/resume-pod`,
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.run(runId) })
      qc.invalidateQueries({ queryKey: qk.runs() })
    },
  })
}
