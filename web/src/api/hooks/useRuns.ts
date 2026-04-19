import { useQuery } from '@tanstack/react-query'
import { api } from '../client'
import { qk } from '../queryKeys'
import type { RunsListResponse } from '../types'

export function useRuns(enabled = true) {
  return useQuery({
    queryKey: qk.runs(),
    queryFn: () => api.get<RunsListResponse>('/runs'),
    refetchInterval: 5_000,
    enabled,
  })
}
