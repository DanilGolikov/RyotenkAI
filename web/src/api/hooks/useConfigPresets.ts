import { useQuery } from '@tanstack/react-query'
import { api } from '../client'
import { qk } from '../queryKeys'
import type { ConfigPresetsResponse } from '../types'

export function useConfigPresets() {
  return useQuery({
    queryKey: qk.configPresets(),
    queryFn: () => api.get<ConfigPresetsResponse>('/config/presets'),
    staleTime: 30 * 60 * 1000,
  })
}
