import type { PluginManifest } from '../api/types'

interface Props {
  plugin: PluginManifest
  onInfoClick: () => void
}

const STABILITY_CLS: Record<string, string> = {
  stable: 'text-ok border-ok/40 bg-ok/10',
  beta: 'text-warn border-warn/40 bg-warn/10',
  experimental: 'text-err border-err/40 bg-err/10',
}

const KIND_CLS: Record<string, string> = {
  validation: 'text-brand-alt border-brand-alt/40 bg-brand-alt/10',
  evaluation: 'text-brand border-brand/40 bg-brand/10',
  reward: 'text-warn border-warn/40 bg-warn/10',
  reports: 'text-ink-2 border-line-2 bg-surface-2',
}

/**
 * Compact catalog card for the Settings/Catalog page. Shows the
 * plugin's name, kind, description, and a small ``i`` button that
 * opens :class:`PluginInfoModal` with the full schema. Kept visually
 * aligned with :class:`ProviderCard` so the Settings surface feels
 * cohesive.
 */
export function PluginCatalogCard({ plugin, onInfoClick }: Props) {
  const paramsCount = Object.keys(
    (plugin.params_schema as { properties?: object })?.properties ?? {},
  ).length
  const thresholdsCount = Object.keys(
    (plugin.thresholds_schema as { properties?: object })?.properties ?? {},
  ).length

  return (
    <div className="card p-3 flex flex-col gap-2 min-w-0">
      <div className="flex items-start gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-2 flex-wrap">
            <div className="text-xs font-semibold text-ink-1 truncate">
              {plugin.name || plugin.id}
            </div>
            <span
              className={`inline-flex items-center rounded border px-1.5 py-0 text-[0.6rem] uppercase tracking-wide ${KIND_CLS[plugin.kind] ?? ''}`}
              title={`Plugin kind: ${plugin.kind}`}
            >
              {plugin.kind}
            </span>
          </div>
          <div className="text-2xs text-ink-3 font-mono truncate mt-0.5">{plugin.id}</div>
        </div>
        <button
          type="button"
          onClick={onInfoClick}
          className="w-5 h-5 rounded-full border border-line-2 text-ink-3 hover:text-ink-1 hover:border-brand-alt text-[0.7rem] font-semibold flex items-center justify-center shrink-0 transition"
          aria-label={`Show details for ${plugin.id}`}
          title="Show details"
        >
          i
        </button>
      </div>

      {plugin.description && (
        <div className="text-2xs text-ink-3 leading-snug line-clamp-2">
          {plugin.description}
        </div>
      )}

      <div className="flex items-center gap-1.5 flex-wrap text-[0.6rem] text-ink-3">
        <span>v{plugin.version}</span>
        {plugin.category && (
          <>
            <span className="text-ink-4">·</span>
            <span>{plugin.category}</span>
          </>
        )}
        {plugin.stability && (
          <span
            className={`inline-flex items-center rounded border px-1 py-0 uppercase tracking-wide ${STABILITY_CLS[plugin.stability] ?? 'text-ink-3 border-line-2'}`}
            title={`Stability: ${plugin.stability}`}
          >
            {plugin.stability}
          </span>
        )}
        {paramsCount > 0 && (
          <span className="text-ink-4" title="Configurable parameters">
            {paramsCount} param{paramsCount === 1 ? '' : 's'}
          </span>
        )}
        {thresholdsCount > 0 && (
          <span className="text-ink-4" title="Threshold knobs">
            {thresholdsCount} threshold{thresholdsCount === 1 ? '' : 's'}
          </span>
        )}
        {plugin.kind === 'reward' && plugin.supported_strategies
          && plugin.supported_strategies.length > 0 && (
          <span
            className="text-brand-alt"
            title="Strategies this reward plugin supports"
          >
            {plugin.supported_strategies.join(', ')}
          </span>
        )}
      </div>
    </div>
  )
}
