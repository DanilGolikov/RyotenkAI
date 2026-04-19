import { useQuery } from '@tanstack/react-query'
import { api } from '../client'
import { qk } from '../queryKeys'
import type { RestartPointsResponse } from '../types'

export function useRestartPoints(runId: string | undefined, enabled: boolean) {
  return useQuery({
    queryKey: runId ? qk.restartPoints(runId) : ['disabled'],
    queryFn: () => api.get<RestartPointsResponse>(`/runs/${encodeURIComponent(runId!)}/restart-points`),
    enabled: !!runId && enabled,
  })
}
