import { usePlugins } from '../api/hooks/usePlugins'
import type { PluginKind, PluginManifest } from '../api/types'
import { Spinner } from './ui'

const STABILITY_CLASS: Record<string, string> = {
  stable: 'text-ok',
  beta: 'text-warn',
  experimental: 'text-brand-alt',
}

function renderJson(value: Record<string, unknown>): string {
  if (!value || Object.keys(value).length === 0) return ''
  return JSON.stringify(value)
}

function PluginCard({ plugin }: { plugin: PluginManifest }) {
  const stabilityCls = STABILITY_CLASS[plugin.stability] ?? 'text-ink-3'
  const params = renderJson(plugin.suggested_params)
  const thresholds = renderJson(plugin.suggested_thresholds)
  return (
    <div className="card p-4 space-y-2">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="text-sm font-medium text-ink-1 truncate">{plugin.name}</div>
          <div className="text-2xs text-ink-3 font-mono">v{plugin.version}</div>
        </div>
        <div className="text-[0.65rem] flex flex-col items-end gap-0.5">
          {plugin.category && (
            <span className="text-ink-3 bg-surface-2 px-1.5 py-0.5 rounded border border-line-1">
              {plugin.category}
            </span>
          )}
          <span className={stabilityCls}>{plugin.stability}</span>
        </div>
      </div>
      {plugin.description && (
        <div className="text-xs text-ink-2 leading-snug">{plugin.description}</div>
      )}
      {params && (
        <div className="text-[0.65rem] font-mono text-ink-3">
          <span className="text-ink-4">params:</span> {params}
        </div>
      )}
      {thresholds && (
        <div className="text-[0.65rem] font-mono text-ink-3">
          <span className="text-ink-4">thresholds:</span> {thresholds}
        </div>
      )}
    </div>
  )
}

export function PluginBrowser({ kind }: { kind: PluginKind }) {
  const { data, isLoading, error } = usePlugins(kind)

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-ink-3">
        <Spinner /> loading plugins
      </div>
    )
  }
  if (error) return <div className="text-sm text-err">{(error as Error).message}</div>
  const plugins = data?.plugins ?? []
  if (plugins.length === 0) {
    return <div className="text-xs text-ink-3">No plugins registered for {kind}.</div>
  }
  return (
    <div className="grid gap-3 grid-cols-1 lg:grid-cols-2">
      {plugins.map((p) => (
        <PluginCard key={p.id} plugin={p} />
      ))}
    </div>
  )
}
