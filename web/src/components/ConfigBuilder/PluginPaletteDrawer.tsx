import { useDraggable } from '@dnd-kit/core'
import { useEffect, useMemo, useState } from 'react'
import type { PluginKind, PluginManifest } from '../../api/types'
import { useAllPlugins } from '../../api/hooks/usePlugins'

interface Props {
  /** Pluin ids already attached per kind so the palette can grey them
   *  out (reports) or flag them as "already in project" (others). */
  attachedIdsByKind: Record<PluginKind, Set<string>>
  /** Currently active strategy types in the project (``grpo``,
   *  ``sft``…). Used to dim reward plugins whose
   *  ``supported_strategies`` don't include any of these. */
  activeStrategyTypes: Set<string>
  onInfoClick?: (plugin: PluginManifest) => void
}

// Project-wide kind order: Validation → Reward → Evaluation → Reports.
// Mirrors ``KIND_SECTIONS`` in ``PluginsTab`` so the palette reads in
// the same sequence as the drop zones on the left.
const KIND_ORDER: PluginKind[] = ['validation', 'reward', 'evaluation', 'reports']
const KIND_LABELS: Record<PluginKind, string> = {
  validation: 'Validation',
  reward: 'Reward',
  evaluation: 'Evaluation',
  reports: 'Reports',
}

/**
 * Right-column plugin palette for the Project → Plugins tab. Each chip
 * is a ``@dnd-kit`` draggable whose ``data`` payload tells the parent's
 * ``onDragEnd`` handler which kind + plugin id is being dropped.
 *
 * Filtering rules applied inside the palette (not in the drag handler)
 * so the visual hints (greyed out, tooltip) and the actual behaviour
 * agree:
 *   - Reports single-instance: already-attached ids are hidden.
 *   - Reward: plugins whose ``supported_strategies`` doesn't match any
 *     current phase strategy are rendered dimmed with a tooltip
 *     explaining why.
 *   - No draggable state when a plugin is marked unusable.
 */
export function PluginPaletteDrawer({
  attachedIdsByKind,
  activeStrategyTypes,
  onInfoClick,
}: Props) {
  const { byKind, isLoading, error } = useAllPlugins()
  const [query, setQuery] = useState('')
  // Collapsed by default — all four accordion groups closed so the
  // palette doesn't dump 30+ chips on the user at once. A non-empty
  // search auto-opens the matching groups (see effect below).
  const [openKinds, setOpenKinds] = useState<Set<PluginKind>>(new Set())

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    const out: Record<PluginKind, PluginManifest[]> = {
      validation: [],
      evaluation: [],
      reward: [],
      reports: [],
    }
    for (const kind of KIND_ORDER) {
      const list = byKind[kind] ?? []
      out[kind] = list.filter((p) => {
        // Reports: drop already-attached ids outright (single-instance).
        if (kind === 'reports' && attachedIdsByKind.reports.has(p.id)) return false
        if (!q) return true
        return (
          p.id.toLowerCase().includes(q)
          || (p.name ?? '').toLowerCase().includes(q)
          || (p.description ?? '').toLowerCase().includes(q)
        )
      })
    }
    return out
  }, [byKind, query, attachedIdsByKind])

  // Search auto-expands groups that have any match — otherwise typing
  // into the search box would look broken with everything collapsed.
  useEffect(() => {
    if (!query.trim()) return
    const toOpen = new Set<PluginKind>()
    for (const kind of KIND_ORDER) {
      if (filtered[kind].length > 0) toOpen.add(kind)
    }
    setOpenKinds(toOpen)
  }, [query, filtered])

  function toggleKind(kind: PluginKind) {
    setOpenKinds((prev) => {
      const next = new Set(prev)
      if (next.has(kind)) next.delete(kind)
      else next.add(kind)
      return next
    })
  }

  // Card-style palette — ``rounded-md border border-line-1`` matches
  // the project's ``.card`` token so the aside reads as part of the
  // same design language as the Providers / Catalog cards. The earlier
  // ``border-l only`` treatment looked like an un-themed side panel.
  //
  // ``self-start + sticky top-0`` pins the palette to the top of the
  // scrolling tab panel. ``top-0`` because the scrolling ancestor is
  // ``<div className="p-5 overflow-y-auto">`` in ProjectDetail, not
  // the viewport; any offset above 0 just leaves a gap.
  return (
    <aside className="w-64 shrink-0 rounded-md border border-line-1 bg-surface-1 shadow-card flex flex-col max-h-[calc(100vh-9rem)] self-start sticky top-0 overflow-hidden">
      <div className="px-3 py-2.5 border-b border-line-1">
        <div className="flex items-baseline justify-between">
          <div className="text-xs font-semibold text-ink-1">Palette</div>
          <div className="text-[0.6rem] text-ink-4">drag to attach</div>
        </div>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search plugins…"
          className="input h-7 text-xs w-full mt-2"
          aria-label="Search palette"
        />
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {error && (
          <div className="text-xs text-err">{(error as Error).message}</div>
        )}
        {isLoading && <div className="text-2xs text-ink-3">Loading…</div>}

        {!isLoading && KIND_ORDER.map((kind) => {
          const list = filtered[kind]
          const isOpen = openKinds.has(kind)
          return (
            <div key={kind} className="space-y-1">
              <button
                type="button"
                onClick={() => toggleKind(kind)}
                className="w-full flex items-center justify-between px-1 py-1 rounded hover:bg-surface-2 transition text-left"
                aria-expanded={isOpen}
                aria-controls={`palette-${kind}`}
              >
                <div className="flex items-center gap-1.5">
                  <span
                    className={`text-[0.65rem] text-ink-4 transition-transform ${isOpen ? 'rotate-90' : ''}`}
                    aria-hidden="true"
                  >
                    ▸
                  </span>
                  <span className="text-[0.65rem] font-semibold text-ink-2 uppercase tracking-wide">
                    {KIND_LABELS[kind]}
                  </span>
                </div>
                <span className="text-[0.6rem] text-ink-4">{list.length}</span>
              </button>
              {isOpen && (
                <div id={`palette-${kind}`} className="space-y-1 pl-1">
                  {list.length === 0 ? (
                    <div className="text-[0.65rem] text-ink-4 px-1 py-1">
                      {query ? 'No matches' : 'Nothing to add — everything is attached or the catalog is empty.'}
                    </div>
                  ) : (
                    list.map((p) => {
                      const attached = attachedIdsByKind[kind].has(p.id)
                      const rewardMismatch =
                        kind === 'reward'
                        && p.supported_strategies
                        && p.supported_strategies.length > 0
                        && !p.supported_strategies.some((s) =>
                          activeStrategyTypes.has(s.toLowerCase()),
                        )
                      return (
                        <PalettePluginChip
                          key={p.id}
                          plugin={p}
                          kind={kind}
                          disabled={!!rewardMismatch}
                          disabledReason={
                            rewardMismatch
                              ? `Incompatible with current strategies (${
                                [...activeStrategyTypes].join(', ') || 'none'
                              }). Supported: ${p.supported_strategies?.join(', ')}`
                              : undefined
                          }
                          attachedHint={attached && kind !== 'reports'
                            ? 'Already attached — drop again to add another instance.'
                            : undefined}
                          onInfoClick={onInfoClick ? () => onInfoClick(p) : undefined}
                        />
                      )
                    })
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </aside>
  )
}

function PalettePluginChip({
  plugin,
  kind,
  disabled,
  disabledReason,
  attachedHint,
  onInfoClick,
}: {
  plugin: PluginManifest
  kind: PluginKind
  disabled: boolean
  disabledReason?: string
  attachedHint?: string
  onInfoClick?: () => void
}) {
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({
    id: `palette:${kind}:${plugin.id}`,
    data: { source: 'palette', kind, pluginId: plugin.id },
    disabled,
  })

  return (
    <div
      ref={setNodeRef}
      {...(disabled ? {} : attributes)}
      {...(disabled ? {} : listeners)}
      className={[
        'flex items-center gap-2 rounded border px-2 py-1 bg-surface-0',
        disabled
          ? 'opacity-40 border-line-1 cursor-not-allowed'
          : 'border-line-1 hover:border-brand-alt cursor-grab active:cursor-grabbing',
        isDragging ? 'ring-2 ring-brand-alt/40 opacity-70' : '',
      ].join(' ')}
      title={disabledReason ?? attachedHint ?? plugin.description ?? undefined}
      aria-label={`Plugin ${plugin.id}${disabled ? ' (unavailable)' : ''}`}
    >
      <div className="flex-1 min-w-0">
        <div className="text-2xs font-mono text-ink-1 truncate">{plugin.id}</div>
        {plugin.name && plugin.name !== plugin.id && (
          <div className="text-[0.6rem] text-ink-3 truncate">{plugin.name}</div>
        )}
      </div>
      {onInfoClick && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation()
            onInfoClick()
          }}
          className="w-4 h-4 rounded-full border border-line-2 text-ink-3 hover:text-ink-1 hover:border-brand-alt text-[0.6rem] font-semibold flex items-center justify-center shrink-0"
          aria-label={`Details for ${plugin.id}`}
          title="Show details"
        >
          i
        </button>
      )}
    </div>
  )
}
