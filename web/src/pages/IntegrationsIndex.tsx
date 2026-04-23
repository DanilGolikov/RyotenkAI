import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useIntegrations, useIntegrationTypes } from '../api/hooks/useIntegrations'
import { NewIntegrationModal } from '../components/NewIntegrationModal'
import { Card, EmptyState, SectionHeader, Spinner } from '../components/ui'
import { timeAgo } from '../lib/format'
import type { IntegrationSummary } from '../api/types'

type TabId = 'mlflow' | 'huggingface'

const TAB_LABEL: Record<TabId, string> = {
  mlflow: 'MLflow',
  huggingface: 'HuggingFace',
}

export function IntegrationsIndexPage() {
  const typesQuery = useIntegrationTypes()
  const { data, isLoading, error } = useIntegrations()
  const [active, setActive] = useState<TabId>('mlflow')
  const [modalOpen, setModalOpen] = useState(false)

  // Deep-link: /settings/integrations#new opens the modal (used by
  // ConfigBuilder dropdowns in PR3).
  useEffect(() => {
    if (window.location.hash.startsWith('#new')) {
      const hashType = window.location.hash.slice('#new'.length).replace(/^[=-]/, '')
      if (hashType === 'mlflow' || hashType === 'huggingface') setActive(hashType)
      setModalOpen(true)
      history.replaceState(null, '', window.location.pathname + window.location.search)
    }
  }, [])

  const tabs: TabId[] = useMemo(() => {
    const ids = new Set(typesQuery.data?.types.map((t) => t.id) ?? [])
    return (['mlflow', 'huggingface'] as TabId[]).filter((t) => ids.has(t))
  }, [typesQuery.data])

  const filtered = (data ?? []).filter((i) => i.type === active)

  const newBtn = (
    <button
      type="button"
      onClick={() => setModalOpen(true)}
      className="btn-primary px-3 py-1.5 text-xs whitespace-nowrap"
    >
      + New integration
    </button>
  )

  return (
    <div className="space-y-4">
      <Card padding="p-0">
        <div className="px-4 pt-4">
          <SectionHeader
            title="Integrations"
            subtitle="Reusable accounts for MLflow tracking and HuggingFace Hub. Tokens are encrypted at rest; configs are versioned on every save."
            action={newBtn}
          />
        </div>

        <div className="px-4 border-b border-line-1 flex gap-1">
          {tabs.map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setActive(t)}
              className={[
                'px-3 py-2 text-xs rounded-t-md transition',
                active === t
                  ? 'text-ink-1 border-b-2 border-brand -mb-px'
                  : 'text-ink-3 hover:text-ink-1',
              ].join(' ')}
            >
              {TAB_LABEL[t]}
            </button>
          ))}
        </div>

        <div className="p-4">
          {error ? (
            <div className="px-3 py-4 text-sm text-err">{(error as Error).message}</div>
          ) : isLoading ? (
            <div className="px-3 py-4 text-sm text-ink-3 flex items-center gap-2">
              <Spinner /> loading
            </div>
          ) : filtered.length === 0 ? (
            <EmptyState
              title={`No ${TAB_LABEL[active]} integrations yet`}
              hint="Create one to share tokens across multiple projects."
              action={newBtn}
            />
          ) : (
            <IntegrationsTable rows={filtered} />
          )}
        </div>
      </Card>
      <NewIntegrationModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        initialType={active}
      />
    </div>
  )
}

const TYPE_LABEL: Record<string, string> = {
  mlflow: 'MLflow',
  huggingface: 'HuggingFace',
}

function IntegrationsTable({ rows }: { rows: IntegrationSummary[] }) {
  const navigate = useNavigate()
  return (
    <div className="rounded-md border border-line-1 overflow-hidden">
      <table className="w-full text-xs">
        <thead className="bg-surface-3 text-2xs uppercase tracking-wide text-ink-2 border-b border-line-2">
          <tr>
            <th className="text-left font-medium px-3 py-2">Name</th>
            <th className="text-left font-medium px-3 py-2">Id</th>
            <th className="text-left font-medium px-3 py-2">Type</th>
            <th className="text-left font-medium px-3 py-2">Created</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const href = `/settings/integrations/${encodeURIComponent(row.id)}`
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
                  <div className="text-ink-1 font-medium truncate">
                    {row.name}
                  </div>
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
