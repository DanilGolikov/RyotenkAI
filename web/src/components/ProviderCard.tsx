import { Link } from 'react-router-dom'
import { timeAgo } from '../lib/format'
import type { ProviderSummary } from '../api/types'

const TYPE_LABEL: Record<string, string> = {
  runpod: 'RunPod',
  single_node: 'Single node',
}

const TYPE_ACCENT: Record<string, string> = {
  runpod: 'from-brand-alt/70 to-brand/70',
  single_node: 'from-info/70 to-brand-alt/60',
}

export function ProviderCard({ provider }: { provider: ProviderSummary }) {
  const typeLabel = TYPE_LABEL[provider.type] ?? provider.type
  const accent = TYPE_ACCENT[provider.type] ?? 'from-brand-alt/60 to-brand/60'
  return (
    <Link
      to={`/settings/providers/${encodeURIComponent(provider.id)}`}
      className="group relative block rounded-lg border border-line-1 bg-surface-1 overflow-hidden hover:border-brand-alt/70 hover:bg-surface-2 transition shadow-card"
    >
      <div className={`h-0.5 bg-gradient-to-r ${accent} opacity-70 group-hover:opacity-100 transition`} />
      <div className="p-4 space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="text-sm font-medium text-ink-1 truncate">{provider.name}</div>
            <div className="text-2xs text-ink-3 mt-0.5 font-mono truncate">{provider.id}</div>
          </div>
          <div className="text-[0.65rem] flex flex-col items-end gap-0.5">
            <span className="text-brand-alt px-1.5 py-0.5 rounded border border-brand-alt/30 bg-brand-alt/5">
              {typeLabel}
            </span>
            <span className="text-ink-3">
              {provider.created_at ? timeAgo(provider.created_at) : ''}
            </span>
          </div>
        </div>
        {provider.description && (
          <div className="text-xs text-ink-2 line-clamp-2">{provider.description}</div>
        )}
        <div className="pt-2 border-t border-line-1/60 text-[0.65rem] font-mono text-ink-4 truncate">
          {provider.path}
        </div>
      </div>
    </Link>
  )
}
