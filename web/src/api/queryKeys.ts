export const qk = {
  health: () => ['health'] as const,
  runs: () => ['runs'] as const,
  run: (runId: string) => ['runs', runId] as const,
  attempt: (runId: string, attemptNo: number) => ['runs', runId, 'attempts', attemptNo] as const,
  stages: (runId: string, attemptNo: number) => ['runs', runId, 'attempts', attemptNo, 'stages'] as const,
  restartPoints: (runId: string) => ['runs', runId, 'restart-points'] as const,
  defaultLaunchMode: (runId: string) => ['runs', runId, 'default-launch-mode'] as const,
  logs: (runId: string, attemptNo: number, file: string, offset: number) =>
    ['runs', runId, 'attempts', attemptNo, 'logs', file, offset] as const,
  report: (runId: string) => ['runs', runId, 'report'] as const,
  configTemplates: () => ['config', 'default'] as const,
  configSchema: () => ['config', 'schema'] as const,
  projects: () => ['projects'] as const,
  project: (id: string) => ['projects', id] as const,
  projectConfig: (id: string) => ['projects', id, 'config'] as const,
  projectConfigVersions: (id: string) => ['projects', id, 'config', 'versions'] as const,
  projectConfigVersion: (id: string, filename: string) =>
    ['projects', id, 'config', 'versions', filename] as const,
  projectEnv: (id: string) => ['projects', id, 'env'] as const,
  plugins: (kind: string) => ['plugins', kind] as const,
  providers: () => ['providers'] as const,
  provider: (id: string) => ['providers', id] as const,
  providerTypes: () => ['providers', 'types'] as const,
  providerConfig: (id: string) => ['providers', id, 'config'] as const,
  providerConfigVersions: (id: string) => ['providers', id, 'config', 'versions'] as const,
  providerConfigVersion: (id: string, filename: string) =>
    ['providers', id, 'config', 'versions', filename] as const,
  integrations: () => ['integrations'] as const,
  integration: (id: string) => ['integrations', id] as const,
  integrationTypes: () => ['integrations', 'types'] as const,
  integrationConfig: (id: string) => ['integrations', id, 'config'] as const,
  integrationConfigVersions: (id: string) => ['integrations', id, 'config', 'versions'] as const,
  integrationConfigVersion: (id: string, filename: string) =>
    ['integrations', id, 'config', 'versions', filename] as const,
  configPresets: () => ['config', 'presets'] as const,
  configPresetPreview: (id: string, currentHash: string) =>
    ['config', 'presets', id, 'preview', currentHash] as const,
}
