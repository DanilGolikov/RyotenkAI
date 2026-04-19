import { Link } from 'react-router-dom'
import { timeAgo } from '../lib/format'
import type { ProviderSummary } from '../api/types'

const TYPE_LABEL: Record<string, string> = {
  runpod: 'RunPod',
  single_node: 'Single node',
}

export function ProviderCard({ provider }: { provider: ProviderSummary }) {
  const typeLabel = TYPE_LABEL[provider.type] ?? provider.type
  return (
    <Link
      to={`/settings/providers/${encodeURIComponent(provider.id)}`}
      className="block card p-4 hover:border-line-2 transition"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-medium text-ink-1 truncate">{provider.name}</div>
          <div className="text-2xs text-ink-3 mt-0.5 font-mono truncate">{provider.id}</div>
        </div>
        <div className="text-[0.65rem] flex flex-col items-end gap-0.5">
          <span className="text-brand-alt">{typeLabel}</span>
          <span className="text-ink-3">
            {provider.created_at ? timeAgo(provider.created_at) : ''}
          </span>
        </div>
      </div>
      {provider.description && (
        <div className="mt-2 text-xs text-ink-2 line-clamp-2">{provider.description}</div>
      )}
      <div className="mt-3 text-[0.65rem] font-mono text-ink-4 truncate">{provider.path}</div>
    </Link>
  )
}
