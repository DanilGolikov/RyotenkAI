/**
 * Per-dataset validation plugins — reuses the same DnD / info /
 * configure UX as the project-wide Plugins tab, but scoped to a single
 * dataset key. See `KindSection` in `../ProjectTabs/PluginsTab` for
 * the interaction contract.
 *
 * Keys that change from the PluginsTab version:
 *   - add / remove / reorder / read-details / write-details / rename
 *     are bound to a specific `datasetKey` via the helpers in
 *     ./datasetValidationInstances.
 *   - PluginPaletteDrawer is filtered to `onlyKinds={['validation']}`.
 *   - The outer DndContext lives here because DatasetDetail is not
 *     wrapped in one — keeping it local keeps the blast radius
 *     contained to this section.
 */

import {
  DndContext,
  DragOverlay,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  pointerWithin,
  useSensor,
  useSensors,
  type DragEndEvent,
  type DragStartEvent,
} from '@dnd-kit/core'
import { sortableKeyboardCoordinates } from '@dnd-kit/sortable'
import { useCallback, useMemo, useState } from 'react'
import { useAllPlugins } from '../../api/hooks/usePlugins'
import type { PluginKind, PluginManifest } from '../../api/types'
import { PluginConfigModal } from '../ConfigBuilder/PluginConfigModal'
import { PluginPaletteDrawer } from '../ConfigBuilder/PluginPaletteDrawer'
import { PluginInfoModal } from '../PluginInfoModal'
import { KindSection } from '../ProjectTabs/PluginsTab'
import type { PluginInstanceDetails } from '../ProjectTabs/pluginInstances'
import {
  addValidationInstanceFor,
  readValidationInstanceDetailsFor,
  readValidationInstancesFor,
  removeValidationInstanceFor,
  renameValidationInstanceFor,
  reorderValidationInstancesFor,
  writeValidationInstanceDetailsFor,
} from './datasetValidationInstances'

interface Props {
  parsed: Record<string, unknown>
  datasetKey: string
  /** Owning project — forwarded to the Configure modal so the env
   *  block can read/write project env.json. */
  projectId: string
  persist: (next: Record<string, unknown>) => Promise<void>
  saving: boolean
}

const KIND: PluginKind = 'validation'

export function ValidationPluginsSection({ parsed, datasetKey, projectId, persist }: Props) {
  const { byKind } = useAllPlugins()

  const instances = useMemo(
    () => readValidationInstancesFor(parsed, datasetKey),
    [parsed, datasetKey],
  )

  const manifestById = useMemo(() => {
    const map = new Map<string, PluginManifest>()
    for (const p of byKind.validation ?? []) map.set(p.id, p)
    return map
  }, [byKind.validation])

  // attachedIdsByKind shape required by PluginPaletteDrawer — only
  // `validation` entry matters here since we pass `onlyKinds`.
  const attachedIdsByKind = useMemo<Record<PluginKind, Set<string>>>(
    () => ({
      validation: new Set(instances.map((i) => i.pluginId)),
      evaluation: new Set(),
      reward: new Set(),
      reports: new Set(),
    }),
    [instances],
  )
  // `activeStrategyTypes` is only used for reward-plugin compatibility
  // warnings — validation has no such constraint, so an empty set is
  // fine here.
  const activeStrategyTypes = useMemo(() => new Set<string>(), [])

  // ---------- modals ----------
  const [infoPlugin, setInfoPlugin] = useState<PluginManifest | null>(null)
  const [configuring, setConfiguring] = useState<
    | { instanceId: string; pluginId: string }
    | null
  >(null)
  const [activeDrag, setActiveDrag] = useState<
    | { source: 'palette'; pluginId: string }
    | { source: 'instance'; instanceId: string }
    | null
  >(null)

  // ---------- mutations ----------
  const handleAdd = useCallback(
    async (pluginId: string) => {
      const manifest = manifestById.get(pluginId)
      if (!manifest) return
      const { next } = addValidationInstanceFor(parsed, manifest, datasetKey)
      await persist(next)
    },
    [parsed, datasetKey, manifestById, persist],
  )

  const handleRemove = useCallback(
    async (instanceId: string) => {
      const next = removeValidationInstanceFor(parsed, instanceId, datasetKey)
      await persist(next)
    },
    [parsed, datasetKey, persist],
  )

  const handleReorder = useCallback(
    async (orderedIds: string[]) => {
      const next = reorderValidationInstancesFor(parsed, orderedIds, datasetKey)
      await persist(next)
    },
    [parsed, datasetKey, persist],
  )

  const handleSaveConfigure = useCallback(
    async (original: PluginInstanceDetails, edited: PluginInstanceDetails) => {
      let stage = parsed
      if (edited.instanceId !== original.instanceId) {
        const renamed = renameValidationInstanceFor(
          stage,
          original.instanceId,
          edited.instanceId,
          datasetKey,
        )
        if (!renamed) throw new Error(`Instance id "${edited.instanceId}" is already taken.`)
        stage = renamed
      }
      const next = writeValidationInstanceDetailsFor(stage, edited, datasetKey)
      await persist(next)
    },
    [parsed, datasetKey, persist],
  )

  // ---------- DnD glue ----------
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  )

  const onDragStart = useCallback((event: DragStartEvent) => {
    const data = event.active.data.current as
      | { source?: 'palette' | 'instance'; kind?: PluginKind; pluginId?: string; instanceId?: string }
      | undefined
    if (!data || data.kind !== KIND) return
    if (data.source === 'palette' && data.pluginId) {
      setActiveDrag({ source: 'palette', pluginId: data.pluginId })
    } else if (data.source === 'instance' && data.instanceId) {
      setActiveDrag({ source: 'instance', instanceId: data.instanceId })
    }
  }, [])

  const onDragEnd = useCallback(
    (event: DragEndEvent) => {
      setActiveDrag(null)
      const { active, over } = event
      if (!over) return
      const activeData = active.data.current as
        | { source?: 'palette' | 'instance'; kind?: PluginKind; pluginId?: string; instanceId?: string }
        | undefined
      const overData = over.data.current as
        | { source?: 'container' | 'instance'; kind?: PluginKind; instanceId?: string }
        | undefined
      if (!activeData || !overData) return
      if (activeData.kind !== KIND || overData.kind !== KIND) return

      if (activeData.source === 'palette' && activeData.pluginId) {
        void handleAdd(activeData.pluginId)
        return
      }

      if (activeData.source === 'instance' && activeData.instanceId) {
        const current = instances.map((x) => x.instanceId)
        const fromIdx = current.indexOf(activeData.instanceId)
        const toInstanceId = overData.source === 'instance' ? overData.instanceId : undefined
        const toIdx = toInstanceId ? current.indexOf(toInstanceId) : current.length - 1
        if (fromIdx < 0 || toIdx < 0 || fromIdx === toIdx) return
        const next = [...current]
        next.splice(fromIdx, 1)
        next.splice(toIdx, 0, activeData.instanceId)
        void handleReorder(next)
      }
    },
    [handleAdd, handleReorder, instances],
  )

  return (
    <DndContext
      sensors={sensors}
      collisionDetection={(args) => {
        const hits = pointerWithin(args)
        if (hits.length > 0) return hits
        return closestCenter(args)
      }}
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
      onDragCancel={() => setActiveDrag(null)}
    >
      <div className="flex gap-4 items-start">
        <div className="flex-1 min-w-0">
          <KindSection
            kind={KIND}
            label="Validation plugins"
            help="Drop a validation plugin from the palette. Each dataset has its own plugin list — the same catalog entry can appear on multiple datasets with independent params."
            sortable
            instances={instances}
            manifestById={manifestById}
            activeStrategyTypes={activeStrategyTypes}
            onRemove={(id) => void handleRemove(id)}
            onConfigure={(inst) => setConfiguring({ instanceId: inst.instanceId, pluginId: inst.pluginId })}
            onInfo={(inst) => {
              const m = manifestById.get(inst.pluginId)
              if (m) setInfoPlugin(m)
            }}
          />
        </div>
        <PluginPaletteDrawer
          attachedIdsByKind={attachedIdsByKind}
          activeStrategyTypes={activeStrategyTypes}
          onInfoClick={(p) => setInfoPlugin(p)}
          onlyKinds={[KIND]}
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
        const details = readValidationInstanceDetailsFor(parsed, configuring.instanceId, datasetKey)
        if (!manifest || !details) {
          setConfiguring(null)
          return null
        }
        const takenIds = instances.map((i) => i.instanceId)
        return (
          <PluginConfigModal
            kind={KIND}
            manifest={manifest}
            initial={details}
            takenInstanceIds={takenIds}
            projectId={projectId}
            onCancel={() => setConfiguring(null)}
            onSave={async (edited) => {
              await handleSaveConfigure(details, edited)
            }}
          />
        )
      })()}
    </DndContext>
  )
}
