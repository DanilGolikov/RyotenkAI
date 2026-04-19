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
}
