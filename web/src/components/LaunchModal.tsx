import { useEffect, useState } from 'react'
import { useRestartPoints } from '../api/hooks/useRestartPoints'
import { useLaunch } from '../api/hooks/useLaunch'
import type { LaunchMode } from '../api/types'

const MODES: { id: LaunchMode; label: string; hint: string }[] = [
  { id: 'new_run', label: 'New run',     hint: 'Fresh directory' },
  { id: 'fresh',   label: 'Fresh',       hint: 'Restart existing' },
  { id: 'resume',  label: 'Resume',      hint: 'Continue where stopped' },
  { id: 'restart', label: 'Restart from', hint: 'Jump to a stage' },
]

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
    if (!open) return
    setMode(defaultMode)
    setConfigPath(defaultConfigPath ?? '')
    setRestartStage('')
  }, [open, defaultMode, defaultConfigPath])

  if (!open) return null

  const availablePoints = restartQuery.data?.points.filter((p) => p.available) ?? []
  const unavailablePoints = restartQuery.data?.points.filter((p) => !p.available) ?? []

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault()
    try {
      await launchMut.mutateAsync({
        mode,
        config_path: configPath || null,
        restart_from_stage: mode === 'restart' ? restartStage : null,
        log_level: logLevel,
      })
      onClose()
    } catch {
      /* error shown below */
    }
  }

  return (
    <div
      className="fixed inset-0 z-40 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose}
    >
      <form
        onSubmit={onSubmit}
        onClick={(event) => event.stopPropagation()}
        className="w-full max-w-2xl rounded-xl border border-line-2 bg-surface-1 shadow-card overflow-hidden"
      >
        <div className="px-5 py-4 bg-gradient-brand-soft border-b border-line-1 flex items-center gap-3">
          <div className="w-8 h-8 rounded-md bg-gradient-brand shadow-glow-burgundy" />
          <div>
            <div className="text-sm font-semibold">Launch pipeline</div>
            <div className="text-2xs text-ink-mute">{runId}</div>
          </div>
        </div>

        <div className="p-5 space-y-5">
          <div className="grid grid-cols-4 gap-2">
            {MODES.map((m) => {
              const active = mode === m.id
              return (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setMode(m.id)}
                  className={[
                    'rounded-md px-3 py-2 text-left transition border',
                    active
                      ? 'bg-gradient-brand text-white border-transparent shadow-glow-burgundy'
                      : 'bg-surface-2 border-line-2 text-ink-dim hover:text-ink hover:border-violet-400',
                  ].join(' ')}
                >
                  <div className="text-xs font-medium">{m.label}</div>
                  <div className={`text-2xs mt-0.5 ${active ? 'text-white/80' : 'text-ink-mute'}`}>{m.hint}</div>
                </button>
              )
            })}
          </div>

          <label className="block">
            <span className="text-2xs uppercase tracking-wider text-ink-mute">Config path</span>
            <input
              value={configPath}
              onChange={(event) => setConfigPath(event.target.value)}
              placeholder="config/pipeline.yaml"
              className="w-full mt-1 bg-surface-2 border border-line-2 rounded-md px-3 py-2 text-sm font-mono text-ink focus:border-burgundy-400 focus:outline-none"
            />
          </label>

          {mode === 'restart' && (
            <label className="block">
              <span className="text-2xs uppercase tracking-wider text-ink-mute">Restart from stage</span>
              <select
                value={restartStage}
                onChange={(event) => setRestartStage(event.target.value)}
                className="w-full mt-1 bg-surface-2 border border-line-2 rounded-md px-3 py-2 text-sm text-ink focus:border-burgundy-400 focus:outline-none"
              >
                <option value="">— select stage —</option>
                {availablePoints.map((p) => (
                  <option key={p.stage} value={p.stage}>{p.stage} ({p.mode})</option>
                ))}
                {unavailablePoints.length > 0 && (
                  <optgroup label="Unavailable">
                    {unavailablePoints.map((p) => (
                      <option key={`u:${p.stage}`} value={p.stage} disabled>
                        {p.stage} — {p.reason}
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>
            </label>
          )}

          <label className="block">
            <span className="text-2xs uppercase tracking-wider text-ink-mute">Log level</span>
            <div className="mt-1 grid grid-cols-2 gap-2 max-w-[220px]">
              {(['INFO', 'DEBUG'] as const).map((lvl) => (
                <button
                  key={lvl}
                  type="button"
                  onClick={() => setLogLevel(lvl)}
                  className={[
                    'px-3 py-1.5 rounded-md text-xs font-mono border transition',
                    logLevel === lvl
                      ? 'bg-surface-3 border-burgundy-400 text-ink'
                      : 'bg-surface-2 border-line-2 text-ink-mute hover:text-ink',
                  ].join(' ')}
                >
                  {lvl}
                </button>
              ))}
            </div>
          </label>

          {launchMut.error && (
            <div className="text-xs text-status-err bg-status-err/10 border border-status-err/30 px-3 py-2 rounded">
              {(launchMut.error as Error).message}
            </div>
          )}
        </div>

        <div className="px-5 py-3 border-t border-line-1 flex items-center justify-end gap-2 bg-surface-0/60">
          <button type="button" onClick={onClose} className="btn-ghost">Cancel</button>
          <button type="submit" disabled={launchMut.isPending} className="btn-primary">
            {launchMut.isPending ? 'Launching…' : 'Launch'}
          </button>
        </div>
      </form>
    </div>
  )
}
