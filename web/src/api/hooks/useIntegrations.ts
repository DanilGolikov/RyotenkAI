import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import { qk } from '../queryKeys'
import type {
  ConfigResponse,
  ConfigValidationResult,
  ConfigVersionDetail,
  ConfigVersionsResponse,
  ConnectionTestResult,
  CreateIntegrationRequest,
  IntegrationDetail,
  IntegrationSummary,
  IntegrationTypesResponse,
  SaveConfigResponse,
} from '../types'

export function useIntegrationTypes() {
  return useQuery({
    queryKey: qk.integrationTypes(),
    queryFn: () => api.get<IntegrationTypesResponse>('/integrations/types'),
    staleTime: 60 * 60 * 1000,
  })
}

export function useIntegrations(typeFilter?: string) {
  return useQuery({
    queryKey: typeFilter ? [...qk.integrations(), { type: typeFilter }] : qk.integrations(),
    queryFn: async () => {
      const all = await api.get<IntegrationSummary[]>('/integrations')
      return typeFilter ? all.filter((i) => i.type === typeFilter) : all
    },
  })
}

export function useIntegration(id: string | undefined) {
  return useQuery({
    queryKey: id ? qk.integration(id) : ['integrations', 'disabled'],
    queryFn: () => api.get<IntegrationDetail>(`/integrations/${encodeURIComponent(id!)}`),
    enabled: !!id,
  })
}

export function useIntegrationConfig(id: string | undefined) {
  return useQuery({
    queryKey: id ? qk.integrationConfig(id) : ['integrations', 'config', 'disabled'],
    queryFn: () =>
      api.get<ConfigResponse>(`/integrations/${encodeURIComponent(id!)}/config`),
    enabled: !!id,
  })
}

export function useIntegrationConfigVersions(id: string | undefined) {
  // Reuse ``ConfigVersionsResponse`` so ``ConfigVersionsPanel`` can be
  // shared between providers and integrations. The optional
  // ``is_favorite`` field on ``ConfigVersion`` is unused here — no
  // favorites API for integrations in v1.
  return useQuery({
    queryKey: id
      ? qk.integrationConfigVersions(id)
      : ['integrations', 'versions', 'disabled'],
    queryFn: () =>
      api.get<ConfigVersionsResponse>(
        `/integrations/${encodeURIComponent(id!)}/config/versions`,
      ),
    enabled: !!id,
  })
}

export function useCreateIntegration() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateIntegrationRequest) =>
      api.post<IntegrationSummary>('/integrations', body),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.integrations() }),
  })
}

export function useSaveIntegrationConfig(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (yaml: string) =>
      api.put<SaveConfigResponse>(
        `/integrations/${encodeURIComponent(id)}/config`,
        { yaml },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.integration(id) })
      qc.invalidateQueries({ queryKey: qk.integrationConfig(id) })
      qc.invalidateQueries({ queryKey: qk.integrationConfigVersions(id) })
    },
  })
}

export function useValidateIntegrationConfig(id: string) {
  return useMutation({
    mutationFn: (yaml: string) =>
      api.post<ConfigValidationResult>(
        `/integrations/${encodeURIComponent(id)}/config/validate`,
        { yaml },
      ),
  })
}

export function useReadIntegrationVersion(
  id: string,
  filename: string | null,
) {
  return useQuery({
    queryKey:
      id && filename
        ? qk.integrationConfigVersion(id, filename)
        : ['integrations', 'version', 'disabled'],
    queryFn: () =>
      api.get<ConfigVersionDetail>(
        `/integrations/${encodeURIComponent(id)}/config/versions/${encodeURIComponent(filename!)}`,
      ),
    enabled: !!(id && filename),
  })
}

export function useRestoreIntegrationVersion(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (filename: string) =>
      api.post<SaveConfigResponse>(
        `/integrations/${encodeURIComponent(id)}/config/versions/${encodeURIComponent(filename)}/restore`,
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.integration(id) })
      qc.invalidateQueries({ queryKey: qk.integrationConfig(id) })
      qc.invalidateQueries({ queryKey: qk.integrationConfigVersions(id) })
    },
  })
}

export function useDeleteIntegration() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      api.del<void>(`/integrations/${encodeURIComponent(id)}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.integrations() }),
  })
}

export function useSetIntegrationToken(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (token: string) =>
      api.put<void>(`/integrations/${encodeURIComponent(id)}/token`, { token }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.integration(id) })
      qc.invalidateQueries({ queryKey: qk.integrations() })
    },
  })
}

export function useDeleteIntegrationToken(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.del<void>(`/integrations/${encodeURIComponent(id)}/token`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.integration(id) })
      qc.invalidateQueries({ queryKey: qk.integrations() })
    },
  })
}

export function useTestIntegrationConnection(id: string) {
  return useMutation({
    mutationFn: () =>
      api.post<ConnectionTestResult>(
        `/integrations/${encodeURIComponent(id)}/test-connection`,
      ),
  })
}
