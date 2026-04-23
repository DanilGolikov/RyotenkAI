import { useQuery } from '@tanstack/react-query'
import { api } from '../client'
import { qk } from '../queryKeys'
import type { PluginKind, PluginListResponse } from '../types'

/** Plugins are loaded from ``community/<kind>/`` and change whenever the
 *  user drops a new plugin folder in. The backend catalogue re-scans on
 *  mtime change, so we give the client a short freshness window and
 *  refetch on window focus. */
export function usePlugins(kind: PluginKind) {
  return useQuery({
    queryKey: qk.plugins(kind),
    queryFn: () => api.get<PluginListResponse>(`/plugins/${kind}`),
    staleTime: 10 * 1000,
    refetchOnWindowFocus: true,
  })
}
