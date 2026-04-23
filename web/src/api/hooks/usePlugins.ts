import { useQueries, useQuery } from '@tanstack/react-query'
import { api } from '../client'
import { qk } from '../queryKeys'
import type { PluginKind, PluginListResponse } from '../types'

const ALL_KINDS: readonly PluginKind[] = [
  'validation',
  'evaluation',
  'reward',
  'reports',
] as const

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

/** Fetch every plugin kind at once for the Settings/Catalog page.
 *  Four parallel queries, one cache entry per kind so invalidation stays
 *  granular (e.g. hot-reloading a single kind after ``sync`` doesn't
 *  refetch the others). */
export function useAllPlugins() {
  const results = useQueries({
    queries: ALL_KINDS.map((kind) => ({
      queryKey: qk.plugins(kind),
      queryFn: () => api.get<PluginListResponse>(`/plugins/${kind}`),
      staleTime: 10 * 1000,
      refetchOnWindowFocus: true,
    })),
  })

  const isLoading = results.some((r) => r.isLoading)
  const error = results.find((r) => r.error)?.error as Error | undefined
  const byKind: Record<PluginKind, PluginListResponse['plugins']> = {
    validation: [],
    evaluation: [],
    reward: [],
    reports: [],
  }
  ALL_KINDS.forEach((kind, idx) => {
    const data = results[idx].data
    if (data) byKind[kind] = data.plugins
  })
  return { byKind, isLoading, error }
}
