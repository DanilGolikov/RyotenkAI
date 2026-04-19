import { useQuery } from '@tanstack/react-query'
import { api } from '../client'
import { qk } from '../queryKeys'
import type { AttemptDetail, StagesResponse } from '../types'

export function useAttempt(runId: string | undefined, attemptNo: number | undefined) {
  return useQuery({
    queryKey: runId && attemptNo ? qk.attempt(runId, attemptNo) : ['disabled'],
    queryFn: () => api.get<AttemptDetail>(`/runs/${encodeURIComponent(runId!)}/attempts/${attemptNo}`),
    enabled: !!runId && !!attemptNo,
    refetchInterval: (query) => {
      const data = query.state.data as AttemptDetail | undefined
      return data?.status === 'running' ? 2_000 : 10_000
    },
  })
}

export function useStages(runId: string | undefined, attemptNo: number | undefined) {
  return useQuery({
    queryKey: runId && attemptNo ? qk.stages(runId, attemptNo) : ['disabled'],
    queryFn: () => api.get<StagesResponse>(`/runs/${encodeURIComponent(runId!)}/attempts/${attemptNo}/stages`),
    enabled: !!runId && !!attemptNo,
    refetchInterval: 3_000,
  })
}
