import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useProviders } from '../api/hooks/useProviders'
import { NewProviderModal } from '../components/NewProviderModal'
import { Card, EmptyState, SectionHeader, Spinner } from '../components/ui'
import { timeAgo } from '../lib/format'
import type { ProviderSummary } from '../api/types'

const TYPE_LABEL: Record<string, string> = {
  runpod: 'RunPod',
  single_node: 'Single node',
}

export function ProvidersPage() {
  const { data, isLoading, error } = useProviders()
  const [modalOpen, setModalOpen] = useState(false)

  // Deep-link: navigating with #new (from ProviderPickerField "Create in
  // Settings →") auto-opens the modal, then clears the hash.
  useEffect(() => {
    if (window.location.hash === '#new') {
      setModalOpen(true)
      history.replaceState(null, '', window.location.pathname + window.location.search)
    }
  }, [])

  const newProviderBtn = (
    <button
      type="button"
      onClick={() => setModalOpen(true)}
      className="btn-primary px-3 py-1.5 text-xs"
    >
      + New provider
    </button>
  )

  return (
    <div className="space-y-4">
      <Card padding="p-0">
        <div className="px-4 pt-4">
          <SectionHeader
            title="Providers"
            subtitle="Reusable compute configurations. Pick one when composing a project."
            action={newProviderBtn}
          />
        </div>
        <div className="p-4 pt-0">
          {error ? (
            <div className="px-3 py-4 text-sm text-err">{(error as Error).message}</div>
          ) : isLoading ? (
            <div className="px-3 py-4 text-sm text-ink-3 flex items-center gap-2">
              <Spinner /> loading
            </div>
          ) : !data || data.length === 0 ? (
            <EmptyState
              title="No providers yet"
              hint="Create one to use across many projects. Versioned on every save."
              action={newProviderBtn}
            />
          ) : (
            <ProvidersTable rows={data} />
          )}
        </div>
      </Card>
      <NewProviderModal open={modalOpen} onClose={() => setModalOpen(false)} />
    </div>
  )
}

function ProvidersTable({ rows }: { rows: ProviderSummary[] }) {
  const navigate = useNavigate()
  return (
    <div className="rounded-md border border-line-1 overflow-hidden">
      <table className="w-full text-xs">
        <thead className="bg-surface-3 text-2xs uppercase tracking-wide text-ink-2 border-b border-line-2">
          <tr>
            <th className="text-left font-medium px-3 py-2">Name</th>
            <th className="text-left font-medium px-3 py-2">Id</th>
            <th className="text-left font-medium px-3 py-2">Type</th>
            <th className="text-left font-medium px-3 py-2">Use</th>
            <th className="text-left font-medium px-3 py-2">Created</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const href = `/settings/providers/${encodeURIComponent(row.id)}`
            const go = () => navigate(href)
            return (
              <tr
                key={row.id}
                role="link"
                tabIndex={0}
                onClick={go}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    go()
                  }
                }}
                className="group border-t border-line-1 cursor-pointer hover:bg-surface-3 focus-visible:bg-surface-3 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-brand transition-colors"
              >
                <td className="px-3 py-2 min-w-0 bg-surface-2/70 group-hover:bg-surface-3 transition-colors">
                  <div className="text-ink-1 font-medium truncate">{row.name}</div>
                  {row.description && (
                    <div className="text-[0.65rem] text-ink-3 truncate mt-0.5">
                      {row.description}
                    </div>
                  )}
                </td>
                <td className="px-3 py-2 font-mono text-ink-3 whitespace-nowrap">
                  {row.id}
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  <span className="text-[0.65rem] text-brand-alt px-1.5 py-0.5 rounded border border-brand-alt/30 bg-brand-alt/5">
                    {TYPE_LABEL[row.type] ?? row.type}
                  </span>
                </td>
                <td className="px-3 py-2 whitespace-nowrap">
                  <CapabilityChips row={row} />
                </td>
                <td className="px-3 py-2 text-ink-3 whitespace-nowrap">
                  {row.created_at ? timeAgo(row.created_at) : ''}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function CapabilityChips({ row }: { row: ProviderSummary }) {
  const chips: { label: string; active: boolean }[] = [
    { label: 'training', active: row.has_training === true },
    { label: 'inference', active: row.has_inference === true },
  ]
  return (
    <div className="flex items-center gap-1.5">
      {chips.map((c) =>
        c.active ? (
          <span
            key={c.label}
            className="text-[0.6rem] text-ok border border-ok/40 bg-ok/10 rounded px-1.5 py-0.5"
          >
            {c.label}
          </span>
        ) : (
          <span
            key={c.label}
            className="text-[0.6rem] text-ink-4 border border-line-1 rounded px-1.5 py-0.5"
          >
            {c.label}
          </span>
        ),
      )}
    </div>
  )
}
