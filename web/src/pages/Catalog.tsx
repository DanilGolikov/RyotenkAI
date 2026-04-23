import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useAllPlugins } from '../api/hooks/usePlugins'
import { useConfigPresets } from '../api/hooks/useConfigPresets'
import { PluginCatalogCard } from '../components/PluginCatalogCard'
import { PluginInfoModal } from '../components/PluginInfoModal'
import { Card, EmptyState, SectionHeader, Spinner } from '../components/ui'
import type { PluginKind, PluginManifest } from '../api/types'

const KIND_FILTERS: { key: PluginKind | 'presets' | 'all'; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'validation', label: 'Validation' },
  { key: 'reward', label: 'Reward' },
  { key: 'evaluation', label: 'Evaluation' },
  { key: 'reports', label: 'Reports' },
  { key: 'presets', label: 'Presets' },
]

type FilterKey = typeof KIND_FILTERS[number]['key']

/**
 * Settings → Catalog. Read-only browsing of every plugin + preset on
 * disk, grouped by kind with search and URL-synced filters.
 *
 * Two user journeys this serves:
 *   1. "What plugins do I even have?" — before planning a run.
 *   2. "What parameters does plugin X accept?" — before editing its
 *      config in a project. Click the ``i`` button on the card to open
 *      :class:`PluginInfoModal` with the full JSON-Schema-driven view.
 *
 * Mirrors the Providers page shell (Card + SectionHeader + EmptyState)
 * so Settings feels like one product, not a grab-bag of tabs.
 */
export function CatalogPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const activeFilter: FilterKey = (searchParams.get('kind') as FilterKey) || 'all'
  const query = searchParams.get('q') || ''

  const { byKind, isLoading, error } = useAllPlugins()
  const { data: presetData } = useConfigPresets()

  const [selected, setSelected] = useState<PluginManifest | null>(null)

  const setFilter = (key: FilterKey) => {
    const next = new URLSearchParams(searchParams)
    if (key === 'all') next.delete('kind')
    else next.set('kind', key)
    setSearchParams(next, { replace: true })
  }

  const setQuery = (q: string) => {
    const next = new URLSearchParams(searchParams)
    if (!q) next.delete('q')
    else next.set('q', q)
    setSearchParams(next, { replace: true })
  }

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    const matches = (p: PluginManifest) =>
      !q
      || p.id.toLowerCase().includes(q)
      || (p.name ?? '').toLowerCase().includes(q)
      || (p.description ?? '').toLowerCase().includes(q)
      || (p.category ?? '').toLowerCase().includes(q)
    return {
      validation: byKind.validation.filter(matches),
      evaluation: byKind.evaluation.filter(matches),
      reward: byKind.reward.filter(matches),
      reports: byKind.reports.filter(matches),
    }
  }, [byKind, query])

  const presets = useMemo(() => {
    const q = query.trim().toLowerCase()
    const all = presetData?.presets ?? []
    if (!q) return all
    return all.filter(
      (p) =>
        p.name.toLowerCase().includes(q)
        || (p.display_name ?? '').toLowerCase().includes(q)
        || (p.description ?? '').toLowerCase().includes(q),
    )
  }, [presetData, query])

  const pluginSections: Array<{ kind: PluginKind; label: string; plugins: PluginManifest[] }> = [
    { kind: 'validation', label: 'Validation', plugins: filtered.validation },
    { kind: 'reward', label: 'Reward', plugins: filtered.reward },
    { kind: 'evaluation', label: 'Evaluation', plugins: filtered.evaluation },
    { kind: 'reports', label: 'Reports', plugins: filtered.reports },
  ]

  const visiblePluginSections = pluginSections.filter(
    (s) => activeFilter === 'all' || activeFilter === s.kind,
  )
  const showPresets = activeFilter === 'all' || activeFilter === 'presets'
  const totalVisible =
    visiblePluginSections.reduce((n, s) => n + s.plugins.length, 0)
    + (showPresets ? presets.length : 0)

  return (
    <div className="space-y-4">
      <Card padding="p-0">
        <div className="px-4 pt-4 pb-3 space-y-3 border-b border-line-1">
          <SectionHeader
            title="Plugin & preset catalog"
            subtitle="Everything available in the community/ directory. Click the info button on a card to see the full schema."
          />
          <div className="flex items-center gap-2 flex-wrap">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by name, id, or description…"
              className="input h-8 text-xs flex-1 min-w-48"
              aria-label="Search catalog"
            />
            <div className="flex gap-1 flex-wrap">
              {KIND_FILTERS.map((f) => (
                <button
                  key={f.key}
                  type="button"
                  onClick={() => setFilter(f.key)}
                  className={[
                    'px-2.5 h-8 rounded-md text-2xs border transition',
                    activeFilter === f.key
                      ? 'bg-surface-2 border-brand text-ink-1'
                      : 'border-line-2 text-ink-3 hover:text-ink-1 hover:border-brand-alt',
                  ].join(' ')}
                >
                  {f.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="p-4 space-y-6">
          {error ? (
            <div className="px-3 py-4 text-sm text-err">{(error as Error).message}</div>
          ) : isLoading ? (
            <div className="px-3 py-4 text-sm text-ink-3 flex items-center gap-2">
              <Spinner /> loading
            </div>
          ) : totalVisible === 0 ? (
            <EmptyState
              title="Nothing matches"
              hint={query ? `No plugins or presets match "${query}".` : 'The catalog is empty.'}
            />
          ) : (
            <>
              {visiblePluginSections.map((section) =>
                section.plugins.length === 0 ? null : (
                  <div key={section.kind} className="space-y-2">
                    <div className="flex items-baseline gap-2">
                      <div className="text-xs font-semibold text-ink-1">{section.label}</div>
                      <div className="text-[0.6rem] text-ink-4">{section.plugins.length}</div>
                    </div>
                    <div className="grid gap-2 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
                      {section.plugins.map((p) => (
                        <PluginCatalogCard
                          key={p.id}
                          plugin={p}
                          onInfoClick={() => setSelected(p)}
                        />
                      ))}
                    </div>
                  </div>
                ),
              )}
              {showPresets && presets.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-baseline gap-2">
                    <div className="text-xs font-semibold text-ink-1">Presets</div>
                    <div className="text-[0.6rem] text-ink-4">{presets.length}</div>
                  </div>
                  <div className="grid gap-2 grid-cols-1 sm:grid-cols-2">
                    {presets.map((preset) => (
                      <div key={preset.name} className="card p-3 space-y-1">
                        <div className="flex items-baseline gap-2 flex-wrap">
                          <div className="text-xs font-semibold text-ink-1 truncate">
                            {preset.display_name || preset.name}
                          </div>
                          {preset.size_tier && (
                            <span className="inline-flex items-center rounded border border-line-2 bg-surface-2 px-1.5 py-0 text-[0.6rem] text-ink-3 uppercase">
                              {preset.size_tier}
                            </span>
                          )}
                        </div>
                        <div className="text-2xs text-ink-3 font-mono truncate">{preset.name}</div>
                        {preset.description && (
                          <div className="text-2xs text-ink-3 leading-snug line-clamp-2">
                            {preset.description}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </Card>

      {selected && (
        <PluginInfoModal plugin={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  )
}
