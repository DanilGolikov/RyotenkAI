import { useEffect, useState } from 'react'
import { useRestartPoints } from '../api/hooks/useRestartPoints'
import { useLaunch } from '../api/hooks/useLaunch'
import type { LaunchMode } from '../api/types'

export function LaunchModal({
  runId,
  open,
  onClose,
  defaultMode = 'resume',
  defaultConfigPath,
}: {
  runId: string
  open: boolean
  onClose: () => void
  defaultMode?: LaunchMode
  defaultConfigPath?: string | null
}) {
  const [mode, setMode] = useState<LaunchMode>(defaultMode)
  const [configPath, setConfigPath] = useState(defaultConfigPath ?? '')
  const [restartStage, setRestartStage] = useState('')
  const [logLevel, setLogLevel] = useState<'INFO' | 'DEBUG'>('INFO')
  const launchMut = useLaunch(runId)
  const restartQuery = useRestartPoints(runId, open && (mode === 'restart' || mode === 'resume'))

  useEffect(() => {
    if (open) {
      setMode(defaultMode)
      setConfigPath(defaultConfigPath ?? '')
      setRestartStage('')
    }
  }, [open, defaultMode, defaultConfigPath])

  if (!open) return null

  const availablePoints = restartQuery.data?.points.filter((p) => p.available) ?? []

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault()
    const body = {
      mode,
      config_path: mode === 'resume' ? (configPath || null) : (configPath || null),
      restart_from_stage: mode === 'restart' ? restartStage : null,
      log_level: logLevel,
    }
    try {
      await launchMut.mutateAsync(body)
      onClose()
    } catch {
      /* error is rendered below */
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <form onSubmit={onSubmit} className="bg-surface-raised border border-surface-muted rounded shadow-lg w-full max-w-lg p-6 space-y-4">
        <h2 className="text-lg text-accent">Launch run</h2>
        <div className="grid grid-cols-4 gap-2">
          {(['new_run', 'fresh', 'resume', 'restart'] as LaunchMode[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`px-3 py-2 rounded text-sm ${mode === m ? 'bg-accent-muted text-white' : 'bg-surface border border-surface-muted text-gray-400 hover:text-gray-100'}`}
            >
              {m}
            </button>
          ))}
        </div>

        <label className="block text-sm">
          Config path
          <input
            value={configPath}
            onChange={(event) => setConfigPath(event.target.value)}
            placeholder="config/pipeline.yaml"
            className="w-full mt-1 bg-surface border border-surface-muted rounded px-3 py-2 text-sm"
          />
        </label>

        {mode === 'restart' && (
          <label className="block text-sm">
            Restart from stage
            <select
              value={restartStage}
              onChange={(event) => setRestartStage(event.target.value)}
              className="w-full mt-1 bg-surface border border-surface-muted rounded px-3 py-2 text-sm"
            >
              <option value="">— select stage —</option>
              {availablePoints.map((p) => (
                <option key={p.stage} value={p.stage}>
                  {p.stage} ({p.mode})
                </option>
              ))}
            </select>
          </label>
        )}

        <label className="block text-sm">
          Log level
          <select
            value={logLevel}
            onChange={(event) => setLogLevel(event.target.value as 'INFO' | 'DEBUG')}
            className="w-full mt-1 bg-surface border border-surface-muted rounded px-3 py-2 text-sm"
          >
            <option value="INFO">INFO</option>
            <option value="DEBUG">DEBUG</option>
          </select>
        </label>

        {launchMut.error && (
          <div className="text-sm text-rose-400">{(launchMut.error as Error).message}</div>
        )}

        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose} className="px-3 py-1.5 text-sm text-gray-400">Cancel</button>
          <button type="submit" disabled={launchMut.isPending} className="px-3 py-1.5 text-sm bg-accent text-surface rounded">
            {launchMut.isPending ? 'Launching…' : 'Launch'}
          </button>
        </div>
      </form>
    </div>
  )
}
