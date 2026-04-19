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
