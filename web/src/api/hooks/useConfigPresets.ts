import { useQuery } from '@tanstack/react-query'
import { api } from '../client'
import { qk } from '../queryKeys'
import type { ConfigPresetsResponse } from '../types'

/** Presets come from ``community/presets/`` on the backend and can change
 *  whenever the user adds/edits a preset folder. The backend re-scans on
 *  mtime change, so we keep the client cache short (10 s) + refetch on
 *  window focus so "add a preset → alt-tab → it's there" works. */
export function useConfigPresets() {
  return useQuery({
    queryKey: qk.configPresets(),
    queryFn: () => api.get<ConfigPresetsResponse>('/config/presets'),
    staleTime: 10 * 1000,
    refetchOnWindowFocus: true,
  })
}
