import { useQuery } from '@tanstack/react-query'
import { api } from '../client'
import { qk } from '../queryKeys'
import type { PluginKind, PluginListResponse } from '../types'

export function usePlugins(kind: PluginKind) {
  return useQuery({
    queryKey: qk.plugins(kind),
    queryFn: () => api.get<PluginListResponse>(`/plugins/${kind}`),
    staleTime: 5 * 60 * 1000,
  })
}
