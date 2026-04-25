import { useCallback, useMemo, useState } from 'react'
import {
  DndContext,
  DragOverlay,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  pointerWithin,
  useDndContext,
  useDroppable,
  useSensor,
  useSensors,
  type DragEndEvent,
  type DragStartEvent,
} from '@dnd-kit/core'
import {
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import {
  useProjectConfig,
  useSaveProjectConfig,
} from '../../api/hooks/useProjects'
import { useAllPlugins } from '../../api/hooks/usePlugins'
import { useReportDefaults } from '../../api/hooks/useReportDefaults'
import type { PluginKind, PluginManifest } from '../../api/types'
import { dumpYaml } from '../../lib/yaml'
import { Spinner } from '../ui'
import { PluginInfoModal } from '../PluginInfoModal'
import { PluginConfigModal } from '../ConfigBuilder/PluginConfigModal'
import { PluginPaletteDrawer } from '../ConfigBuilder/PluginPaletteDrawer'
import { PluginInstanceRow } from './PluginInstanceRow'
import {
  addInstance,
  isRecord,
  readInstanceDetails,
  readInstances,
  rewardBroadcastTargets,
  removeInstance,
  renameInstance,
  reorderInstances,
  writeInstanceDetails,
  type PluginInstanceDetails,
} from './pluginInstances'

interface Props {
  projectId: string
}

// Validation kind lives under the Datasets project tab now — every
// dataset has its own plugin attachments. See
// `web/src/components/Datasets/ValidationPluginsSection.tsx`. The
// generic Plugins tab keeps reward / evaluation / reports only.
const KIND_SECTIONS: {
  kind: PluginKind
  label: string
  help: string
  sortable: boolean
}[] = [
  { kind: 'reward', label: 'Reward', help: 'Reward plugin for GRPO/SAPO/DPO/ORPO strategies. Matched against each phase\'s strategy_type.', sortable: false },
  { kind: 'evaluation', label: 'Evaluation', help: 'Post-training model evaluators.', sortable: true },
  { kind: 'reports', label: 'Reports', help: 'Sections rendered in the run report — drag to reorder.', sortable: true },
]

/**
 * Project → Plugins tab. Schema v3 UX:
 *   - Four vertical kind sections (Validation / Evaluation / Reward /
 *     Reports), each a ``SortableContext`` over the project's current
 *     plugin instance list for that kind.
 *   - Right-side palette (``PluginPaletteDrawer``) as the DnD source
 *     for adding new instances. Same plugin dropped twice becomes two
 *     instances with unique ``id_2`` / ``id_3`` suffixes (validation,
 *     evaluation). Reward and reports are single-instance per id.
 *   - Drag within a section = reorder. Drag from palette into a kind
 *     section = add instance. Dragging across kinds is rejected by
 *     the drop handler (kind mismatch).
 *
 * State is managed through ``useProjectConfig`` → pure helper mutation
 * → ``useSaveProjectConfig``. No local duplicate state; optimistic
 * updates come from react-query's mutation cache on save.
 */
export function PluginsTab({ projectId }: Props) {
  const configQuery = useProjectConfig(projectId)
  const pluginsAll = useAllPlugins()
  const reportDefaultsQuery = useReportDefaults()
  const saveMut = useSaveProjectConfig(projectId)
  const parsed = configQuery.data?.parsed_json ?? {}

  const [infoPlugin, setInfoPlugin] = useState<PluginManifest | null>(null)
  const [configuring, setConfiguring] = useState<
    | { kind: PluginKind; instanceId: string; pluginId: string }
    | null
  >(null)
  const [activeDrag, setActiveDrag] = useState<
    | { source: 'palette'; kind: PluginKind; pluginId: string }
    | { source: 'instance'; kind: PluginKind; instanceId: string }
    | null
  >(null)

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: { distance: 4 },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  )

  // ---------- derived data ----------

  // When the pipeline config omits ``reports.sections`` the backend
  // falls back to ``DEFAULT_REPORT_SECTIONS``. Showing an empty Reports
  // section here would be misleading ("I thought I had reports?") and
  // makes drag-to-reorder impossible. We mirror the backend fallback:
  // render defaults when sections is empty, and materialize them into
  // the config on the first mutation (see ``maybeMaterializeReports``
  // below).
  const reportsInstancesMaterialized = useMemo(() => {
    const explicit = readInstances('reports', parsed)
    if (explicit.length > 0) return explicit
    const defaults = reportDefaultsQuery.data?.sections ?? []
    return defaults.map((id) => ({ instanceId: id, pluginId: id }))
  }, [parsed, reportDefaultsQuery.data])

  const reportsAreMaterialized = useMemo(() => {
    const explicit = readInstances('reports', parsed)
    return explicit.length > 0
  }, [parsed])

  const instancesByKind = useMemo(() => {
    const out: Record<PluginKind, { instanceId: string; pluginId: string; enabled?: boolean }[]> = {
      validation: readInstances('validation', parsed),
      evaluation: readInstances('evaluation', parsed),
      reward: readInstances('reward', parsed),
      reports: reportsInstancesMaterialized,
    }
    return out
  }, [parsed, reportsInstancesMaterialized])

  /** Returns a parsed config with the rendered (possibly default) reports
   *  list written into ``reports.sections`` so the first add/remove/
   *  reorder doesn't silently lose the defaults the user was staring at. */
  const materializeReportsIfNeeded = useCallback(
    (source: Record<string, unknown>): Record<string, unknown> => {
      if (reportsAreMaterialized) return source
      const defaults = reportDefaultsQuery.data?.sections
      if (!defaults || defaults.length === 0) return source
      const next = structuredClone(source) as Record<string, unknown>
      const reports = (next.reports && typeof next.reports === 'object' && !Array.isArray(next.reports))
        ? (next.reports as Record<string, unknown>)
        : {}
      reports.sections = [...defaults]
      next.reports = reports
      return next
    },
    [reportsAreMaterialized, reportDefaultsQuery.data],
  )

  const attachedIdsByKind = useMemo(() => {
    const out: Record<PluginKind, Set<string>> = {
      validation: new Set(instancesByKind.validation.map((x) => x.pluginId)),
      evaluation: new Set(instancesByKind.evaluation.map((x) => x.pluginId)),
      reward: new Set(instancesByKind.reward.map((x) => x.pluginId)),
      reports: new Set(instancesByKind.reports.map((x) => x.pluginId)),
    }
    return out
  }, [instancesByKind])

  const manifestById = useMemo(() => {
    const map = new Map<string, PluginManifest>()
    for (const kind of ['validation', 'evaluation', 'reward', 'reports'] as const) {
      for (const p of pluginsAll.byKind[kind] ?? []) map.set(p.id, p)
    }
    return map
  }, [pluginsAll.byKind])

  const activeStrategyTypes = useMemo(() => {
    const training = isRecord(parsed.training) ? parsed.training : {}
    const strategies = Array.isArray(training.strategies) ? (training.strategies as unknown[]) : []
    const out = new Set<string>()
    for (const s of strategies) {
      if (!isRecord(s)) continue
      const t = typeof s.strategy_type === 'string' ? s.strategy_type.toLowerCase() : ''
      if (t) out.add(t)
    }
    return out
  }, [parsed])

  // ---------- mutations ----------

  const commit = useCallback(
    async (nextParsed: Record<string, unknown>) => {
      await saveMut.mutateAsync(dumpYaml(nextParsed))
    },
    [saveMut],
  )

  const handleAdd = useCallback(
    async (kind: PluginKind, pluginId: string) => {
      const manifest = manifestById.get(pluginId)
      if (!manifest) return
      const base = kind === 'reports' ? materializeReportsIfNeeded(parsed) : parsed
      const { next } = addInstance(kind, base, manifest)
      await commit(next)
    },
    [commit, manifestById, parsed, materializeReportsIfNeeded],
  )

  const handleRemove = useCallback(
    async (kind: PluginKind, instanceId: string) => {
      const base = kind === 'reports' ? materializeReportsIfNeeded(parsed) : parsed
      const next = removeInstance(kind, base, instanceId)
      await commit(next)
    },
    [commit, parsed, materializeReportsIfNeeded],
  )

  const handleReorder = useCallback(
    async (kind: PluginKind, orderedIds: string[]) => {
      const base = kind === 'reports' ? materializeReportsIfNeeded(parsed) : parsed
      const next = reorderInstances(kind, base, orderedIds)
      await commit(next)
    },
    [commit, parsed, materializeReportsIfNeeded],
  )

  const handleSaveConfigure = useCallback(
    async (kind: PluginKind, original: PluginInstanceDetails, edited: PluginInstanceDetails) => {
      let stage = parsed
      if (edited.instanceId !== original.instanceId) {
        const renamed = renameInstance(kind, stage, original.instanceId, edited.instanceId)
        if (!renamed) throw new Error(`Instance id "${edited.instanceId}" is already taken.`)
        stage = renamed
      }
      const next = writeInstanceDetails(kind, stage, edited)
      await commit(next)
    },
    [commit, parsed],
  )

  /** Drop ``reports.sections`` from the config so the backend falls
   *  back to ``DEFAULT_REPORT_SECTIONS`` again. Used by the reports
   *  empty-state "Reset to defaults" CTA: removing the key (not
   *  writing defaults literally) means future releases shipping new
   *  built-in sections automatically flow to this user. */
  const handleResetReports = useCallback(async () => {
    const next = structuredClone(parsed) as Record<string, unknown>
    if (next.reports && typeof next.reports === 'object' && !Array.isArray(next.reports)) {
      const reports = next.reports as Record<string, unknown>
      delete reports.sections
      if (Object.keys(reports).length === 0) delete next.reports
    }
    await commit(next)
  }, [commit, parsed])

  // ---------- DnD glue ----------

  const onDragStart = useCallback((event: DragStartEvent) => {
    const data = event.active.data.current as {
      source?: 'palette' | 'instance'
      kind?: PluginKind
      pluginId?: string
      instanceId?: string
    } | undefined
    if (!data) return
    if (data.source === 'palette' && data.kind && data.pluginId) {
      setActiveDrag({ source: 'palette', kind: data.kind, pluginId: data.pluginId })
    } else if (data.source === 'instance' && data.kind && data.instanceId) {
      setActiveDrag({ source: 'instance', kind: data.kind, instanceId: data.instanceId })
    }
  }, [])

  const onDragEnd = useCallback(
    (event: DragEndEvent) => {
      setActiveDrag(null)
      const { active, over } = event
      if (!over) return
      const activeData = active.data.current as {
        source?: 'palette' | 'instance'
        kind?: PluginKind
        pluginId?: string
        instanceId?: string
      } | undefined
      const overData = over.data.current as {
        source?: 'instance' | 'container'
        kind?: PluginKind
        instanceId?: string
      } | undefined
      if (!activeData || !overData) return
      const targetKind = overData.kind
      const sourceKind = activeData.kind
      if (!targetKind || sourceKind !== targetKind) return // kind guard

      if (activeData.source === 'palette' && activeData.pluginId) {
        void handleAdd(targetKind, activeData.pluginId)
        return
      }

      if (activeData.source === 'instance' && activeData.instanceId) {
        // Reorder within the same kind. Use the current order as base,
        // move the dragged instance to the position of the ``over``.
        const current = instancesByKind[targetKind].map((x) => x.instanceId)
        const fromIdx = current.indexOf(activeData.instanceId)
        const toInstanceId =
          overData.source === 'instance' ? overData.instanceId : undefined
        const toIdx = toInstanceId ? current.indexOf(toInstanceId) : current.length - 1
        if (fromIdx < 0 || toIdx < 0 || fromIdx === toIdx) return
        const next = [...current]
        next.splice(fromIdx, 1)
        next.splice(toIdx, 0, activeData.instanceId)
        void handleReorder(targetKind, next)
      }
    },
    [handleAdd, handleReorder, instancesByKind],
  )

  // ---------- render ----------

  // Gate the tab on BOTH config AND the plugin catalog. Without the
  // second condition we render rows with an empty ``manifestById`` for
  // ~1 frame on first mount — every row lights up amber (``isStale``)
  // until the plugin queries resolve. Waiting another beat on the
  // spinner is preferable to a golden flash. ``reportDefaultsQuery``
  // also contributes: reports.sections falls back to the server's
  // defaults, which we don't want to briefly render as empty either.
  if (configQuery.isLoading || pluginsAll.isLoading || reportDefaultsQuery.isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-ink-3">
        <Spinner /> loading config
      </div>
    )
  }

  return (
    <DndContext
      sensors={sensors}
      collisionDetection={(args) => {
        // Prefer precise pointer-in for palette → section drops; fall
        // back to closestCenter for sortable reorders inside one list.
        const hits = pointerWithin(args)
        if (hits.length > 0) return hits
        return closestCenter(args)
      }}
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
      onDragCancel={() => setActiveDrag(null)}
    >
      {/* Save errors block — rendered only when non-empty, so it
          doesn't push the first section down when everything is fine.
          In-flight "Saving…" / "Saved" state was dropped entirely: it
          flickered for ~100ms per action and added no value beyond
          what the row-level optimistic update already conveys. */}
      {saveMut.error && (
        <div className="rounded-md border border-err/40 bg-err/10 text-err text-xs px-3 py-2 mb-3">
          {(saveMut.error as Error).message}
        </div>
      )}
      <div className="flex gap-4 items-start">
        <div className="flex-1 min-w-0 space-y-5">
          {KIND_SECTIONS.map((section) => (
            <KindSection
              key={section.kind}
              kind={section.kind}
              label={section.label}
              help={section.help}
              sortable={section.sortable}
              instances={instancesByKind[section.kind]}
              manifestById={manifestById}
              activeStrategyTypes={activeStrategyTypes}
              onRemove={(id) => void handleRemove(section.kind, id)}
              onConfigure={(inst) => setConfiguring({
                kind: section.kind,
                instanceId: inst.instanceId,
                pluginId: inst.pluginId,
              })}
              onInfo={(inst) => {
                const m = manifestById.get(inst.pluginId)
                if (m) setInfoPlugin(m)
              }}
              onResetToDefaults={
                section.kind === 'reports' && reportsAreMaterialized
                  ? () => void handleResetReports()
                  : undefined
              }
            />
          ))}
        </div>
        <PluginPaletteDrawer
          attachedIdsByKind={attachedIdsByKind}
          activeStrategyTypes={activeStrategyTypes}
          onInfoClick={(p) => setInfoPlugin(p)}
        />
      </div>

      <DragOverlay dropAnimation={null}>
        {activeDrag?.source === 'palette' ? (
          <div className="rounded border border-brand-alt bg-surface-1 px-2 py-1 text-2xs font-mono text-ink-1 shadow-card">
            {activeDrag.pluginId}
          </div>
        ) : activeDrag?.source === 'instance' ? (
          <div className="rounded border border-brand bg-surface-1 px-2 py-1 text-2xs font-mono text-ink-1 shadow-card">
            {activeDrag.instanceId}
          </div>
        ) : null}
      </DragOverlay>

      {infoPlugin && (
        <PluginInfoModal plugin={infoPlugin} onClose={() => setInfoPlugin(null)} />
      )}

      {configuring && (() => {
        const manifest = manifestById.get(configuring.pluginId)
        const details = readInstanceDetails(configuring.kind, parsed, configuring.instanceId)
        if (!manifest || !details) {
          // Race: either the manifest disappeared or the instance was
          // removed before the modal opened. Silently close.
          setConfiguring(null)
          return null
        }
        const takenIds = instancesByKind[configuring.kind].map((i) => i.instanceId)
        return (
          <PluginConfigModal
            kind={configuring.kind}
            manifest={manifest}
            initial={details}
            takenInstanceIds={takenIds}
            projectId={projectId}
            broadcastTargets={
              configuring.kind === 'reward'
                ? rewardBroadcastTargets(parsed, configuring.instanceId)
                : undefined
            }
            onCancel={() => setConfiguring(null)}
            onSave={async (edited) => {
              await handleSaveConfigure(configuring.kind, details, edited)
            }}
          />
        )
      })()}
    </DndContext>
  )
}

export function KindSection({
  kind,
  label,
  help,
  sortable,
  instances,
  manifestById,
  activeStrategyTypes,
  onRemove,
  onConfigure,
  onInfo,
  onResetToDefaults,
}: {
  kind: PluginKind
  label: string
  help: string
  sortable: boolean
  instances: { instanceId: string; pluginId: string; enabled?: boolean }[]
  manifestById: Map<string, PluginManifest>
  activeStrategyTypes: Set<string>
  onRemove: (instanceId: string) => void
  onConfigure: (instance: { instanceId: string; pluginId: string }) => void
  onInfo: (instance: { instanceId: string; pluginId: string }) => void
  /** Optional "Reset to defaults" action — only wired for ``reports``,
   *  where emptying the list is valid but the user usually wants the
   *  built-in section order back. Other kinds don't have defaults. */
  onResetToDefaults?: () => void
}) {
  const { setNodeRef } = useDroppable({
    id: `container:${kind}`,
    data: { source: 'container', kind },
  })
  // The active drag's payload — used to decide whether THIS section is
  // a valid drop target and to compute "is hover". ``useDroppable.isOver``
  // only fires when the cursor is literally over the container element;
  // it stays ``false`` while the cursor hovers one of the sortable
  // child rows, because the nearest droppable then is the row itself.
  // We derive ``isOverSection`` from ``dnd.over`` instead, which
  // exposes the true resolved target — either our container or any
  // instance belonging to our kind.
  const dnd = useDndContext()
  const activeData = dnd.active?.data?.current as
    | { source?: 'palette' | 'instance'; kind?: PluginKind }
    | undefined
  const overData = dnd.over?.data?.current as
    | { source?: 'container' | 'instance'; kind?: PluginKind }
    | undefined
  const isOverSection =
    !!dnd.over
    && (dnd.over.id === `container:${kind}` || overData?.kind === kind)
  const canAccept = isOverSection && (!activeData || activeData.kind === kind)
  const willReject = isOverSection && !!activeData && activeData.kind !== kind
  const itemIds = instances.map((i) => `instance:${kind}:${i.instanceId}`)

  return (
    <section className="space-y-2">
      <div className="flex items-baseline gap-2">
        <div className="text-xs font-semibold text-ink-1">{label}</div>
        <div className="text-[0.6rem] text-ink-4">{instances.length}</div>
      </div>
      <div className="text-[0.65rem] text-ink-3 leading-snug">{help}</div>
      <SortableContext items={itemIds} strategy={verticalListSortingStrategy}>
        <div
          ref={setNodeRef}
          className={[
            'rounded-md border-2 border-dashed bg-surface-1/50 p-2 space-y-1.5 min-h-[64px] transition',
            canAccept
              ? 'border-brand-alt bg-brand-alt/5'
              : willReject
                ? 'border-err/60 bg-err/5 cursor-not-allowed'
                : 'border-line-1',
          ].join(' ')}
          aria-disabled={willReject || undefined}
        >
          {instances.length === 0 ? (
            <div className="px-2 py-3 text-center space-y-2">
              <div className="text-[0.65rem] text-ink-3">
                {onResetToDefaults
                  ? 'No sections — the report will render empty.'
                  : `Drop a ${label.toLowerCase()} plugin here from the palette.`}
              </div>
              {onResetToDefaults && (
                <button
                  type="button"
                  onClick={onResetToDefaults}
                  className="btn-ghost h-7 text-[0.65rem] px-2"
                  title="Restore the built-in section order."
                >
                  ↺ Reset to defaults
                </button>
              )}
            </div>
          ) : (
            instances.map((inst) => {
              const manifest = manifestById.get(inst.pluginId) ?? null
              const warning = rewardIncompatibilityWarning(kind, manifest, activeStrategyTypes)
              return (
                <PluginInstanceRow
                  key={inst.instanceId}
                  instanceId={inst.instanceId}
                  pluginId={inst.pluginId}
                  manifest={manifest}
                  kind={kind}
                  sortable={sortable}
                  enabled={inst.enabled}
                  onRemove={() => onRemove(inst.instanceId)}
                  onConfigure={manifest ? () => onConfigure(inst) : undefined}
                  onInfo={manifest ? () => onInfo(inst) : undefined}
                  warning={warning}
                />
              )
            })
          )}
        </div>
      </SortableContext>
    </section>
  )
}

/** Returns a user-facing message when the attached reward plugin is
 *  incompatible with any currently-active strategy phase, or ``undefined``
 *  when everything is fine. */
export function rewardIncompatibilityWarning(
  kind: PluginKind,
  manifest: PluginManifest | null,
  activeStrategyTypes: Set<string>,
): string | undefined {
  if (kind !== 'reward' || !manifest) return undefined
  const supported = manifest.supported_strategies ?? []
  if (supported.length === 0) return undefined
  const match = supported.some((s) => activeStrategyTypes.has(s.toLowerCase()))
  if (match) return undefined
  return `Incompatible with current strategies (${
    [...activeStrategyTypes].join(', ') || 'none'
  }). Supports: ${supported.join(', ')}.`
}
