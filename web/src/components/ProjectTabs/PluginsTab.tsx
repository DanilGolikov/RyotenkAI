import { useCallback, useMemo, useState } from 'react'
import {
  DndContext,
  DragOverlay,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  pointerWithin,
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
  removeInstance,
  renameInstance,
  reorderInstances,
  writeInstanceDetails,
  type PluginInstanceDetails,
} from './pluginInstances'

interface Props {
  projectId: string
}

const KIND_SECTIONS: {
  kind: PluginKind
  label: string
  help: string
  sortable: boolean
}[] = [
  { kind: 'validation', label: 'Validation', help: 'Dataset pre-flight checks.', sortable: true },
  { kind: 'evaluation', label: 'Evaluation', help: 'Post-training model evaluators.', sortable: true },
  { kind: 'reward', label: 'Reward', help: 'Reward plugin for GRPO/SAPO/DPO/ORPO strategies. Matched against each phase\'s strategy_type.', sortable: false },
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

  const instancesByKind = useMemo(() => {
    const out: Record<PluginKind, { instanceId: string; pluginId: string }[]> = {
      validation: readInstances('validation', parsed),
      evaluation: readInstances('evaluation', parsed),
      reward: readInstances('reward', parsed),
      reports: readInstances('reports', parsed),
    }
    return out
  }, [parsed])

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
      const { next } = addInstance(kind, parsed, manifest)
      await commit(next)
    },
    [commit, manifestById, parsed],
  )

  const handleRemove = useCallback(
    async (kind: PluginKind, instanceId: string) => {
      const next = removeInstance(kind, parsed, instanceId)
      await commit(next)
    },
    [commit, parsed],
  )

  const handleReorder = useCallback(
    async (kind: PluginKind, orderedIds: string[]) => {
      const next = reorderInstances(kind, parsed, orderedIds)
      await commit(next)
    },
    [commit, parsed],
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

  if (configQuery.isLoading) {
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
      <div className="flex gap-4">
        <div className="flex-1 min-w-0 space-y-5">
          <div className="text-2xs text-ink-3">
            {saveMut.isPending ? 'Saving…' : saveMut.isSuccess ? 'Saved' : ''}
          </div>
          {saveMut.error && (
            <div className="rounded-md border border-err/40 bg-err/10 text-err text-xs px-3 py-2">
              {(saveMut.error as Error).message}
            </div>
          )}

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

function KindSection({
  kind,
  label,
  help,
  sortable,
  instances,
  manifestById,
  activeStrategyTypes,
  onRemove,
  onConfigure,
}: {
  kind: PluginKind
  label: string
  help: string
  sortable: boolean
  instances: { instanceId: string; pluginId: string }[]
  manifestById: Map<string, PluginManifest>
  activeStrategyTypes: Set<string>
  onRemove: (instanceId: string) => void
  onConfigure: (instance: { instanceId: string; pluginId: string }) => void
}) {
  const { isOver, setNodeRef } = useDroppable({
    id: `container:${kind}`,
    data: { source: 'container', kind },
  })
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
            isOver ? 'border-brand-alt bg-brand-alt/5' : 'border-line-1',
          ].join(' ')}
        >
          {instances.length === 0 ? (
            <div className="text-[0.65rem] text-ink-3 px-2 py-3 text-center">
              Drop a {label.toLowerCase()} plugin here from the palette.
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
                  onRemove={() => onRemove(inst.instanceId)}
                  onConfigure={manifest ? () => onConfigure(inst) : undefined}
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
function rewardIncompatibilityWarning(
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
