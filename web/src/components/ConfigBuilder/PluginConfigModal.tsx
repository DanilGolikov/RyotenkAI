import { useEffect, useMemo, useRef, useState } from 'react'
import { useClickOutside } from '../../hooks/useClickOutside'
import type { PipelineJsonSchema, JsonSchemaNode } from '../../api/hooks/useConfigSchema'
import type { PluginKind, PluginManifest } from '../../api/types'
import type { PluginInstanceDetails } from '../ProjectTabs/pluginInstances'
import { ObjectFields } from './FieldRenderer'
import { HelpTooltip } from './HelpTooltip'

interface Props {
  kind: PluginKind
  manifest: PluginManifest
  initial: PluginInstanceDetails
  /** Ids already in use within this kind's list — used for client-side
   *  uniqueness validation when the user renames the instance. */
  takenInstanceIds: string[]
  onCancel: () => void
  onSave: (next: PluginInstanceDetails) => Promise<void>
}

const INSTRUCTIONS_BY_KIND: Record<PluginKind, string> = {
  validation: 'Adjust how this dataset validator runs and its threshold knobs.',
  evaluation: 'Fine-tune the evaluator parameters and thresholds; toggle whether the run keeps its per-sample report.',
  reward: 'Reward plugins apply to every matching training phase. Changes here propagate to all grpo/sapo/dpo/orpo strategies in the project.',
  reports: 'Report sections have no per-instance parameters — ordering is handled by drag-and-drop in the Plugins tab.',
}

/**
 * Configure-instance modal. Three collapsible sections, each rendered
 * only when it has content:
 *
 *   1. **Instance** — id, enabled, kind-specific toggles
 *      (apply_to / fail_on_error / save_report).
 *   2. **Parameters** — :class:`FieldRenderer` over
 *      ``manifest.params_schema`` (already JSON Schema shaped after
 *      Phase 1's ``ui_manifest`` rewrite).
 *   3. **Thresholds** — same mechanism with ``thresholds_schema``.
 *
 * Two reset actions:
 *   - Per-field ``↺ default`` is handled inside ``FieldRenderer`` (the
 *     schema's own ``default`` gets treated as a placeholder).
 *   - Per-instance "Reset to suggested" at the footer flood-fills
 *     ``suggested_params`` / ``suggested_thresholds`` — the values the
 *     plugin author recommends, which may differ from each field's
 *     individual default.
 *
 * No save-on-blur — the user commits via the Save button. Cancel
 * discards unsaved edits after a confirm when dirty.
 */
export function PluginConfigModal({
  kind,
  manifest,
  initial,
  takenInstanceIds,
  onCancel,
  onSave,
}: Props) {
  const [draft, setDraft] = useState<PluginInstanceDetails>(initial)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)

  const panelRef = useRef<HTMLDivElement | null>(null)
  const firstFieldRef = useRef<HTMLInputElement | null>(null)
  useClickOutside(panelRef, true, () => handleCancel())

  useEffect(() => {
    firstFieldRef.current?.focus()
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') handleCancel()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft])

  const dirty = useMemo(
    () => JSON.stringify(draft) !== JSON.stringify(initial),
    [draft, initial],
  )

  const takenOthers = useMemo(
    () => takenInstanceIds.filter((id) => id !== initial.instanceId),
    [takenInstanceIds, initial.instanceId],
  )
  const idCollision = takenOthers.includes(draft.instanceId)

  // Schemas come pre-shaped as JSON Schema `{type: object, properties, required}`.
  // FieldRenderer expects a PipelineJsonSchema root with $defs; our
  // plugin schema is flat so we can safely cast.
  const paramsSchema = manifest.params_schema as unknown as JsonSchemaNode
  const thresholdsSchema = manifest.thresholds_schema as unknown as JsonSchemaNode
  const paramsRoot = paramsSchema as unknown as PipelineJsonSchema
  const thresholdsRoot = thresholdsSchema as unknown as PipelineJsonSchema

  const paramCount = Object.keys(
    (paramsSchema as { properties?: object }).properties ?? {},
  ).length
  const thresholdCount = Object.keys(
    (thresholdsSchema as { properties?: object }).properties ?? {},
  ).length

  function handleCancel() {
    if (dirty && !confirm('Discard unsaved changes?')) return
    onCancel()
  }

  async function handleSave() {
    if (idCollision) return
    setSaving(true)
    setSaveError(null)
    try {
      await onSave(draft)
      onCancel()
    } catch (err) {
      setSaveError((err as Error).message)
    } finally {
      setSaving(false)
    }
  }

  function resetToSuggested() {
    setDraft({
      ...draft,
      params: { ...(manifest.suggested_params ?? {}) },
      thresholds: { ...(manifest.suggested_thresholds ?? {}) },
    })
  }

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="plugin-config-title"
    >
      <div
        ref={panelRef}
        className="w-full max-w-3xl max-h-[88vh] overflow-hidden rounded-xl border border-line-2 bg-surface-1 shadow-card flex flex-col"
      >
        {/* Header */}
        <div className="px-4 py-3 border-b border-line-1 flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div id="plugin-config-title" className="text-sm font-semibold text-ink-1 truncate">
              Configure {manifest.name || manifest.id}
            </div>
            <div className="text-2xs text-ink-3 flex items-center gap-2 flex-wrap mt-0.5">
              <span className="uppercase tracking-wide">{kind}</span>
              <span className="text-ink-4">·</span>
              <code className="font-mono text-ink-2">{manifest.id}</code>
              <span className="text-ink-4">·</span>
              <span>v{manifest.version}</span>
            </div>
          </div>
          <button
            type="button"
            onClick={handleCancel}
            className="text-ink-3 hover:text-ink-1 text-xs"
          >
            close
          </button>
        </div>

        <div className="px-4 py-2 border-b border-line-1 text-2xs text-ink-3 leading-snug">
          {INSTRUCTIONS_BY_KIND[kind]}
        </div>

        {/* Body */}
        <div className="p-4 text-xs space-y-5 overflow-y-auto">
          {/* Instance settings — flat rows, no group heading. The
              ``Parameters`` / ``Thresholds`` blocks below get their own
              counted headers; for instance-level knobs the context is
              obvious from the modal title. */}
          <div className="space-y-2">
            {kind !== 'reports' && kind !== 'reward' && (
              <Field
                label="Instance id"
                help="Used to identify this attachment in logs and reports. Must be unique within this kind."
              >
                <input
                  ref={firstFieldRef}
                  type="text"
                  className={`input w-full text-xs ${idCollision ? 'border-err ring-1 ring-err' : ''}`}
                  value={draft.instanceId}
                  onChange={(e) => setDraft({ ...draft, instanceId: e.target.value })}
                />
                {idCollision && (
                  <div className="text-[0.65rem] text-err mt-1">
                    This id is already used by another instance.
                  </div>
                )}
              </Field>
            )}

            {kind === 'reward' && (
              <div className="text-2xs text-ink-3">
                Reward instance id is fixed to the plugin id —{' '}
                <code className="font-mono text-ink-2">{draft.instanceId}</code>.
              </div>
            )}

            {draft.enabled !== undefined && (
              <Field
                label="Enabled"
                help="Uncheck to keep this attachment in the config but skip it on next run."
              >
                <label className="inline-flex items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    checked={draft.enabled}
                    onChange={(e) => setDraft({ ...draft, enabled: e.target.checked })}
                  />
                  <span className="text-ink-2">{draft.enabled ? 'yes' : 'no'}</span>
                </label>
              </Field>
            )}

            {kind === 'validation' && (
              <>
                <Field
                  label="Apply to"
                  help="Which dataset phases this validator runs against. Empty = both."
                >
                  <div className="flex gap-3 text-xs">
                    {['train', 'eval'].map((phase) => {
                      const checked = draft.apply_to?.includes(phase) ?? true
                      return (
                        <label key={phase} className="inline-flex items-center gap-1.5">
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={(e) => {
                              const current = new Set(draft.apply_to ?? ['train', 'eval'])
                              if (e.target.checked) current.add(phase)
                              else current.delete(phase)
                              setDraft({ ...draft, apply_to: [...current] })
                            }}
                          />
                          {phase}
                        </label>
                      )
                    })}
                  </div>
                </Field>
                <Field
                  label="Fail on error"
                  help="Abort the run when this validator reports a FAIL issue."
                >
                  <label className="inline-flex items-center gap-2 text-xs">
                    <input
                      type="checkbox"
                      checked={draft.fail_on_error ?? false}
                      onChange={(e) => setDraft({ ...draft, fail_on_error: e.target.checked })}
                    />
                    <span className="text-ink-2">{draft.fail_on_error ? 'yes' : 'no'}</span>
                  </label>
                </Field>
              </>
            )}

            {kind === 'evaluation' && (
              <Field
                label="Save per-sample report"
                help="Writes a Markdown breakdown next to the run — useful for debugging threshold failures."
              >
                <label className="inline-flex items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    checked={draft.save_report ?? true}
                    onChange={(e) => setDraft({ ...draft, save_report: e.target.checked })}
                  />
                  <span className="text-ink-2">{draft.save_report ? 'yes' : 'no'}</span>
                </label>
              </Field>
            )}
          </div>

          {/* Parameters — no optional-field collapse; plugins usually have
              few knobs and users expect to see them all when configuring. */}
          {paramCount > 0 && (
            <SchemaGroup title="Parameters" count={paramCount}>
              <ObjectFields
                root={paramsRoot}
                node={paramsSchema}
                value={draft.params}
                onChange={(next) => setDraft({ ...draft, params: (next as Record<string, unknown>) ?? {} })}
                depth={0}
                pathPrefix={`__plugin_params__/${manifest.id}`}
                forceExpandOptional
              />
            </SchemaGroup>
          )}

          {/* Thresholds */}
          {thresholdCount > 0 && (
            <SchemaGroup title="Thresholds" count={thresholdCount}>
              <ObjectFields
                root={thresholdsRoot}
                node={thresholdsSchema}
                value={draft.thresholds}
                onChange={(next) => setDraft({ ...draft, thresholds: (next as Record<string, unknown>) ?? {} })}
                depth={0}
                pathPrefix={`__plugin_thresholds__/${manifest.id}`}
                forceExpandOptional
              />
            </SchemaGroup>
          )}

          {paramCount === 0 && thresholdCount === 0 && (
            <div className="rounded-md border border-dashed border-line-2 px-3 py-2 text-2xs text-ink-3">
              This plugin has no configurable parameters or thresholds.
            </div>
          )}

          {saveError && (
            <div className="rounded-md border border-err/40 bg-err/10 text-err text-xs px-3 py-2">
              {saveError}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-line-1 flex items-center justify-between gap-2">
          <button
            type="button"
            onClick={resetToSuggested}
            className="btn-ghost h-8 text-xs"
            title="Restore the values the plugin author recommends (suggested_params / suggested_thresholds)."
            disabled={paramCount === 0 && thresholdCount === 0}
          >
            Reset to suggested
          </button>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleCancel}
              className="btn-ghost h-8 text-xs"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleSave}
              disabled={saving || idCollision || !dirty}
              className="btn-primary h-8 text-xs px-3 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {saving ? 'Saving…' : 'Save'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

/** Group header for the schema-driven Parameters / Thresholds blocks.
 *  Shows ``Title · N`` so users can see at a glance how many knobs are
 *  coming; a thin hairline under the header keeps the group visually
 *  distinct without a full card chrome. */
function SchemaGroup({
  title,
  count,
  children,
}: {
  title: string
  count: number
  children: React.ReactNode
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-baseline gap-2 border-b border-line-1 pb-1">
        <div className="text-xs font-semibold text-ink-1">{title}</div>
        <div className="text-[0.65rem] text-ink-4">·</div>
        <div className="text-[0.65rem] text-ink-3">{count}</div>
      </div>
      <div className="space-y-2">{children}</div>
    </div>
  )
}

/** One row for an instance-level scalar: label on the left with a
 *  ``?``-tooltip for long explanations, input on the right. Replaces
 *  the old inline-under-label description block — the help text now
 *  lives inside ``HelpTooltip`` so instance rows don't dominate the
 *  modal height. */
function Field({
  label,
  help,
  children,
}: {
  label: string
  help?: string
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:gap-3">
      <div className="sm:w-40 shrink-0 flex items-center gap-1.5">
        <div className="text-xs text-ink-2">{label}</div>
        {help && <HelpTooltip text={help} label={`Help for ${label}`} />}
      </div>
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  )
}
