import { useSortable } from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import type { PluginManifest, PluginKind } from '../../api/types'

interface Props {
  instanceId: string
  pluginId: string
  /** When ``null`` the plugin is missing from the catalog (stale
   *  reference) — row goes into warning state. */
  manifest: PluginManifest | null
  kind: PluginKind
  /** Reward rows are not sortable (single-instance per strategy-phase). */
  sortable: boolean
  /** False = instance is attached but will be skipped at run time.
   *  Renders as a desaturated / muted row so the user can see at a
   *  glance which items are dormant without opening Configure. */
  enabled?: boolean
  onRemove: () => void
  onConfigure?: () => void
  onInfo?: () => void
  /** Extra warning text (e.g. reward incompatible with current
   *  strategy). Rendered inline below the row. */
  warning?: string
}

/**
 * One row inside a kind-section of the project's Plugins tab. The row
 * is a ``@dnd-kit/sortable`` item when ``sortable`` is true — drag the
 * ≡ handle on the left to reorder within the same section. Dropping a
 * row on an empty drop-zone in another section is disallowed at the
 * top-level collision detector, not here.
 *
 * Shows: drag-handle · instance id · plugin-id reference · Configure
 * button (wired to a modal in a later phase) · Remove button.
 *
 * Missing-manifest fallback: when ``manifest`` is null we assume the
 * plugin was removed from ``community/`` while the config still
 * references it. The row renders an amber warning state, Configure is
 * disabled, but Remove still works so the user can clean up.
 */
export function PluginInstanceRow({
  instanceId,
  pluginId,
  manifest,
  kind,
  sortable,
  enabled = true,
  onRemove,
  onConfigure,
  onInfo,
  warning,
}: Props) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({
    id: `instance:${kind}:${instanceId}`,
    data: { source: 'instance', kind, instanceId },
    disabled: !sortable,
  })

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
  }

  const isStale = !manifest
  const isDisabled = enabled === false
  const rowCls = [
    'flex items-center gap-2 rounded-md border px-2 py-1.5 transition',
    // Desaturated + dashed-border look when disabled — conveys
    // "attached but skipped" without hiding the row. Hover still
    // highlights so the user can Configure / Remove / Enable.
    isDisabled
      ? 'bg-surface-1 border-dashed border-line-2 opacity-60 grayscale'
      : 'bg-surface-0',
    !isDisabled && (isStale
      ? 'border-warn/50'
      : warning
        ? 'border-err/50'
        : 'border-line-1 hover:border-line-2'),
    isDragging ? 'opacity-60 ring-2 ring-brand-alt/40' : '',
  ].filter(Boolean).join(' ')

  return (
    <div>
      <div ref={setNodeRef} style={style} className={rowCls}>
        {sortable && (
          <button
            type="button"
            {...attributes}
            {...listeners}
            className="text-ink-4 hover:text-ink-1 cursor-grab active:cursor-grabbing px-1 leading-none"
            aria-label={`Reorder ${instanceId}`}
            title="Drag to reorder"
          >
            ≡
          </button>
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-2 min-w-0">
            <div className="text-xs font-mono text-ink-1 truncate">{instanceId}</div>
            {instanceId !== pluginId && (
              <div className="text-[0.65rem] text-ink-3 font-mono truncate" title={`Plugin reference: ${pluginId}`}>
                → {pluginId}
              </div>
            )}
            {isDisabled && (
              <span
                className="inline-flex items-center rounded border border-line-2 bg-surface-2 px-1.5 py-0 text-[0.6rem] uppercase tracking-wide text-ink-3"
                title="Disabled — attached but will be skipped on next run. Toggle in Configure."
              >
                disabled
              </span>
            )}
          </div>
          {isStale ? (
            <div className="text-[0.65rem] text-warn mt-0.5">
              Plugin not found in catalog — remove or reinstall.
            </div>
          ) : warning ? (
            <div className="text-[0.65rem] text-err mt-0.5">{warning}</div>
          ) : manifest?.name && manifest.name !== pluginId ? (
            <div className="text-[0.65rem] text-ink-3 truncate">{manifest.name}</div>
          ) : null}
        </div>
        {onInfo && (
          <button
            type="button"
            disabled={isStale}
            onClick={onInfo}
            className="w-6 h-6 rounded-full border border-line-2 text-ink-3 hover:text-ink-1 hover:border-brand-alt text-[0.7rem] font-semibold flex items-center justify-center shrink-0 disabled:opacity-40 disabled:cursor-not-allowed"
            title={isStale ? 'Plugin missing — no catalog entry' : 'Show plugin details'}
            aria-label={`Show details for ${pluginId}`}
          >
            i
          </button>
        )}
        <button
          type="button"
          disabled={isStale}
          onClick={onConfigure}
          className="btn-ghost h-7 text-[0.65rem] px-2 disabled:opacity-40 disabled:cursor-not-allowed"
          title={isStale ? 'Plugin missing — cannot configure' : 'Configure this instance'}
        >
          Configure
        </button>
        <button
          type="button"
          onClick={onRemove}
          className="btn-ghost h-7 text-[0.65rem] px-2 text-err hover:bg-err/10"
          title="Remove from project"
          aria-label={`Remove ${instanceId}`}
        >
          Remove
        </button>
      </div>
    </div>
  )
}
