import { useQuery } from '@tanstack/react-query'
import { api } from '../client'
import { qk } from '../queryKeys'
import type { RunDetail } from '../types'

export function useRun(runId: string | undefined) {
  return useQuery({
    queryKey: runId ? qk.run(runId) : ['runs', 'disabled'],
    queryFn: () => api.get<RunDetail>(`/runs/${encodeURIComponent(runId!)}`),
    enabled: !!runId,
    refetchInterval: (query) => {
      const data = query.state.data as RunDetail | undefined
      return data?.status === 'running' ? 2_000 : 10_000
    },
  })
}
