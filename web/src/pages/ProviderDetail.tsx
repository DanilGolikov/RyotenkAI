import { NavLink, Navigate, Route, Routes, useNavigate, useParams } from 'react-router-dom'
import {
  useDeleteProvider,
  useProvider,
  useProviderTypes,
} from '../api/hooks/useProviders'
import { ProviderConfigTab } from '../components/ProviderTabs/ProviderConfigTab'
import { ProviderVersionsTab } from '../components/ProviderTabs/ProviderVersionsTab'
import { Card, Spinner } from '../components/ui'

const TABS: { to: string; label: string }[] = [
  { to: 'config', label: 'Config' },
  { to: 'versions', label: 'Versions' },
]

export function ProviderDetailPage() {
  const { id } = useParams<{ id: string }>()
  const { data: provider, isLoading, error } = useProvider(id)
  const typesQuery = useProviderTypes()
  const deleteMut = useDeleteProvider()
  const navigate = useNavigate()

  if (!id) return <Navigate to="/settings/providers" replace />
  if (isLoading || typesQuery.isLoading) {
    return (
      <div className="p-6 text-sm text-ink-3 flex items-center gap-2">
        <Spinner /> loading provider
      </div>
    )
  }
  if (error) return <div className="p-6 text-sm text-err">{(error as Error).message}</div>
  if (!provider) return <div className="p-6 text-sm text-ink-3">Provider not found.</div>

  const typeInfo = typesQuery.data?.types.find((t) => t.id === provider.type)

  return (
    <Card padding="p-0">
      <div className="px-5 py-4 border-b border-line-1 bg-surface-2">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <div className="text-lg font-semibold text-ink-1">{provider.name}</div>
              <span className="text-[0.65rem] text-brand-alt px-1.5 py-0.5 rounded border border-brand-alt/40">
                {typeInfo?.label ?? provider.type}
              </span>
            </div>
            <div className="text-2xs text-ink-3 font-mono mt-0.5">{provider.id}</div>
            {provider.description && (
              <div className="text-xs text-ink-2 mt-2">{provider.description}</div>
            )}
            <div className="text-[0.65rem] text-ink-4 font-mono mt-2 truncate">
              {provider.path}
            </div>
          </div>
          <div className="shrink-0 flex items-center gap-2">
            <button
              type="button"
              disabled={deleteMut.isPending}
              onClick={async () => {
                const ok = window.confirm(
                  `Unregister provider "${provider.name}"?\n\n` +
                    `Files on disk at ${provider.path} are preserved — you can re-register ` +
                    `the same id later. Projects that already inlined this provider are not affected.`,
                )
                if (!ok) return
                try {
                  await deleteMut.mutateAsync(provider.id)
                  navigate('/settings/providers')
                } catch {
                  /* error already surfaced via hook state */
                }
              }}
              className="rounded-md border border-err/50 px-3 py-1.5 text-2xs text-err hover:bg-err/10 hover:border-err transition disabled:opacity-50"
            >
              {deleteMut.isPending ? 'Deleting…' : 'Delete'}
            </button>
          </div>
        </div>
        {deleteMut.error ? (
          <div className="mt-2 text-err text-2xs">
            {(deleteMut.error as Error).message}
          </div>
        ) : null}
      </div>

      <div className="px-3 pt-2 border-b border-line-1 flex gap-1">
        {TABS.map((t) => (
          <NavLink
            key={t.to}
            to={t.to}
            replace
            className={({ isActive }) =>
              [
                'px-3 py-2 text-xs rounded-t-md transition',
                isActive
                  ? 'text-ink-1 border-b-2 border-brand -mb-px'
                  : 'text-ink-3 hover:text-ink-1',
              ].join(' ')
            }
          >
            {t.label}
          </NavLink>
        ))}
      </div>

      <div className="p-5">
        <Routes>
          <Route index element={<Navigate to="config" replace />} />
          <Route
            path="config"
            element={
              <ProviderConfigTab providerId={provider.id} providerType={provider.type} />
            }
          />
          <Route path="versions" element={<ProviderVersionsTab providerId={provider.id} />} />
          <Route path="*" element={<Navigate to="config" replace />} />
        </Routes>
      </div>
    </Card>
  )
}
