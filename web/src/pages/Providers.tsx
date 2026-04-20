import { useEffect, useState } from 'react'
import { useProviders } from '../api/hooks/useProviders'
import { NewProviderModal } from '../components/NewProviderModal'
import { ProviderCard } from '../components/ProviderCard'
import { Card, EmptyState, SectionHeader, Spinner } from '../components/ui'

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
            <div className="grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
              {data.map((p) => (
                <ProviderCard key={p.id} provider={p} />
              ))}
            </div>
          )}
        </div>
      </Card>
      <NewProviderModal open={modalOpen} onClose={() => setModalOpen(false)} />
    </div>
  )
}
