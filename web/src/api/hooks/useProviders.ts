import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import { qk } from '../queryKeys'
import type {
  ConfigValidationResult,
  ConfigVersionDetail,
  ConfigVersionsResponse,
  ConfigResponse,
  CreateProviderRequest,
  ProviderDetail,
  ProviderSummary,
  ProviderTypesResponse,
  SaveConfigResponse,
} from '../types'

export function useProviderTypes() {
  return useQuery({
    queryKey: qk.providerTypes(),
    queryFn: () => api.get<ProviderTypesResponse>('/providers/types'),
    staleTime: 60 * 60 * 1000,
  })
}

export function useProviders() {
  return useQuery({
    queryKey: qk.providers(),
    queryFn: () => api.get<ProviderSummary[]>('/providers'),
  })
}

export function useProvider(providerId: string | undefined) {
  return useQuery({
    queryKey: providerId ? qk.provider(providerId) : ['providers', 'disabled'],
    queryFn: () => api.get<ProviderDetail>(`/providers/${encodeURIComponent(providerId!)}`),
    enabled: !!providerId,
  })
}

export function useProviderConfig(providerId: string | undefined) {
  return useQuery({
    queryKey: providerId ? qk.providerConfig(providerId) : ['providers', 'config', 'disabled'],
    queryFn: () => api.get<ConfigResponse>(`/providers/${encodeURIComponent(providerId!)}/config`),
    enabled: !!providerId,
  })
}

export function useProviderConfigVersions(providerId: string | undefined) {
  return useQuery({
    queryKey: providerId
      ? qk.providerConfigVersions(providerId)
      : ['providers', 'versions', 'disabled'],
    queryFn: () =>
      api.get<ConfigVersionsResponse>(
        `/providers/${encodeURIComponent(providerId!)}/config/versions`,
      ),
    enabled: !!providerId,
  })
}

export function useCreateProvider() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateProviderRequest) => api.post<ProviderSummary>('/providers', body),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.providers() }),
  })
}

export function useSaveProviderConfig(providerId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (yaml: string) =>
      api.put<SaveConfigResponse>(`/providers/${encodeURIComponent(providerId)}/config`, { yaml }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.provider(providerId) })
      qc.invalidateQueries({ queryKey: qk.providerConfig(providerId) })
      qc.invalidateQueries({ queryKey: qk.providerConfigVersions(providerId) })
    },
  })
}

export function useValidateProviderConfig(providerId: string) {
  return useMutation({
    mutationFn: (yaml: string) =>
      api.post<ConfigValidationResult>(
        `/providers/${encodeURIComponent(providerId)}/config/validate`,
        { yaml },
      ),
  })
}

export function useReadProviderVersion(providerId: string, filename: string | null) {
  return useQuery({
    queryKey:
      providerId && filename
        ? qk.providerConfigVersion(providerId, filename)
        : ['providers', 'version', 'disabled'],
    queryFn: () =>
      api.get<ConfigVersionDetail>(
        `/providers/${encodeURIComponent(providerId)}/config/versions/${encodeURIComponent(filename!)}`,
      ),
    enabled: !!(providerId && filename),
  })
}

export function useRestoreProviderVersion(providerId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (filename: string) =>
      api.post<SaveConfigResponse>(
        `/providers/${encodeURIComponent(providerId)}/config/versions/${encodeURIComponent(filename)}/restore`,
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.provider(providerId) })
      qc.invalidateQueries({ queryKey: qk.providerConfig(providerId) })
      qc.invalidateQueries({ queryKey: qk.providerConfigVersions(providerId) })
    },
  })
}

export function useDeleteProvider() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (providerId: string) => api.del<void>(`/providers/${encodeURIComponent(providerId)}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.providers() }),
  })
}
