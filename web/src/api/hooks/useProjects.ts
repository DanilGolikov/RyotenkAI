import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../client'
import { qk } from '../queryKeys'
import type {
  ConfigResponse,
  ConfigVersionDetail,
  ConfigVersionsResponse,
  ConfigValidationResult,
  CreateProjectRequest,
  ProjectDetail,
  ProjectRunsResponse,
  ProjectSummary,
  SaveConfigResponse,
} from '../types'

export function useProjects() {
  return useQuery({
    queryKey: qk.projects(),
    queryFn: () => api.get<ProjectSummary[]>('/projects'),
  })
}

interface ProjectEnvResponse {
  env: Record<string, string>
}

export function useProjectEnv(projectId: string | undefined) {
  return useQuery({
    queryKey: projectId ? qk.projectEnv(projectId) : ['projects', 'env', 'disabled'],
    queryFn: () => api.get<ProjectEnvResponse>(`/projects/${encodeURIComponent(projectId!)}/env`),
    enabled: !!projectId,
  })
}

export function useSaveProjectEnv(projectId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (env: Record<string, string>) =>
      api.put<ProjectEnvResponse>(`/projects/${encodeURIComponent(projectId)}/env`, { env }),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.projectEnv(projectId) }),
  })
}

export function useProject(projectId: string | undefined) {
  return useQuery({
    queryKey: projectId ? qk.project(projectId) : ['projects', 'disabled'],
    queryFn: () => api.get<ProjectDetail>(`/projects/${encodeURIComponent(projectId!)}`),
    enabled: !!projectId,
  })
}

export function useProjectConfig(projectId: string | undefined) {
  return useQuery({
    queryKey: projectId ? qk.projectConfig(projectId) : ['projects', 'config', 'disabled'],
    queryFn: () =>
      api.get<ConfigResponse>(`/projects/${encodeURIComponent(projectId!)}/config`),
    enabled: !!projectId,
  })
}

export function useProjectConfigVersions(projectId: string | undefined) {
  return useQuery({
    queryKey: projectId
      ? qk.projectConfigVersions(projectId)
      : ['projects', 'versions', 'disabled'],
    queryFn: () =>
      api.get<ConfigVersionsResponse>(
        `/projects/${encodeURIComponent(projectId!)}/config/versions`,
      ),
    enabled: !!projectId,
  })
}

export function useCreateProject() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateProjectRequest) => api.post<ProjectSummary>('/projects', body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.projects() })
    },
  })
}

export function useSaveProjectConfig(projectId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (yaml: string) =>
      api.put<SaveConfigResponse>(
        `/projects/${encodeURIComponent(projectId)}/config`,
        { yaml },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.project(projectId) })
      qc.invalidateQueries({ queryKey: qk.projectConfig(projectId) })
      qc.invalidateQueries({ queryKey: qk.projectConfigVersions(projectId) })
    },
  })
}

export function useValidateProjectConfig(projectId: string) {
  return useMutation({
    mutationFn: (yaml: string) =>
      api.post<ConfigValidationResult>(
        `/projects/${encodeURIComponent(projectId)}/config/validate`,
        { yaml },
      ),
  })
}

export function useReadConfigVersion(projectId: string, filename: string | null) {
  return useQuery({
    queryKey:
      projectId && filename
        ? qk.projectConfigVersion(projectId, filename)
        : ['projects', 'version', 'disabled'],
    queryFn: () =>
      api.get<ConfigVersionDetail>(
        `/projects/${encodeURIComponent(projectId)}/config/versions/${encodeURIComponent(filename!)}`,
      ),
    enabled: !!(projectId && filename),
  })
}

export function useRestoreConfigVersion(projectId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (filename: string) =>
      api.post<SaveConfigResponse>(
        `/projects/${encodeURIComponent(projectId)}/config/versions/${encodeURIComponent(filename)}/restore`,
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.project(projectId) })
      qc.invalidateQueries({ queryKey: qk.projectConfig(projectId) })
      qc.invalidateQueries({ queryKey: qk.projectConfigVersions(projectId) })
    },
  })
}

export function useUpdateProjectDescription(projectId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (description: string) =>
      api.put<ProjectDetail>(
        `/projects/${encodeURIComponent(projectId)}/description`,
        { description },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.project(projectId) })
      qc.invalidateQueries({ queryKey: qk.projects() })
    },
  })
}


export function useDeleteProject() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({
      projectId,
      deleteFiles = true,
    }: {
      projectId: string
      deleteFiles?: boolean
    }) =>
      api.del<void>(
        `/projects/${encodeURIComponent(projectId)}?delete_files=${deleteFiles}`,
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.projects() }),
  })
}

/**
 * Fetch the project's launched-run ledger from
 * ``GET /projects/{id}/runs`` (Step 6 of Variant 1). The endpoint
 * surfaces ``<project>/runs/index.json`` newest-first.
 *
 * Pass ``status`` to filter (e.g. ``"running"`` / ``"completed"``) and
 * ``limit`` to cap the number of returned rows. The hook polls every
 * 5 s by default so the Runs tab reflects new launches without manual
 * refresh — adjust via the React Query devtools when debugging.
 */
export function useProjectRuns(
  projectId: string | undefined,
  options: { status?: string; limit?: number } = {},
) {
  const params = new URLSearchParams()
  if (options.status) params.set('status', options.status)
  if (options.limit !== undefined) params.set('limit', String(options.limit))
  const suffix = params.toString() ? `?${params.toString()}` : ''
  return useQuery({
    queryKey: projectId
      ? qk.projectRuns(projectId, options)
      : ['projects', 'runs', 'disabled'],
    queryFn: () =>
      api.get<ProjectRunsResponse>(
        `/projects/${encodeURIComponent(projectId!)}/runs${suffix}`,
      ),
    enabled: !!projectId,
    refetchInterval: 5_000,
  })
}


export function useToggleFavoriteVersion(projectId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ filename, favorite }: { filename: string; favorite: boolean }) =>
      api.put<{ favorite_versions: string[] }>(
        `/projects/${encodeURIComponent(projectId)}/config/versions/${encodeURIComponent(filename)}/favorite`,
        { favorite },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.projectConfigVersions(projectId) })
      qc.invalidateQueries({ queryKey: qk.project(projectId) })
    },
  })
}
