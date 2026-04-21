import { useState } from 'react'
import { api } from '../../api/client'
import { useProviders, useProviderTypes } from '../../api/hooks/useProviders'
import type {
  ConfigResponse,
  ProviderSummary,
  ProviderTypeInfo,
} from '../../api/types'
import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { FieldRenderer } from './FieldRenderer'

export interface GroupRendererProps {
  root: PipelineJsonSchema
  node: JsonSchemaNode
  value: unknown
  onChange: (value: unknown) => void
  labelKey: string
  required?: boolean
  /** The whole form value — used when the group needs to cross-edit sibling groups (e.g. `training.provider`). */
  rootValue?: Record<string, unknown>
  onRootChange?: (next: Record<string, unknown>) => void
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

export function ProviderPickerField({
  value,
  onChange,
  required,
  rootValue,
  onRootChange,
}: GroupRendererProps) {
  const providersQuery = useProviders()
  const typesQuery = useProviderTypes()
  const [expanded, setExpanded] = useState<string | null>(null)
  const [pickOpen, setPickOpen] = useState<'settings' | 'custom' | null>(null)
  const [customType, setCustomType] = useState('')
  const [customName, setCustomName] = useState('')
  const [attachError, setAttachError] = useState<string | null>(null)

  const attached = isRecord(value) ? value : {}
  const attachedIds = Object.keys(attached)

  const registered: ProviderSummary[] = providersQuery.data ?? []
  const registeredById = new Map(registered.map((p) => [p.id, p]))
  const types: ProviderTypeInfo[] = typesQuery.data?.types ?? []
  const typeById = new Map(types.map((t) => [t.id, t]))

  function updateAttached(next: Record<string, unknown>) {
    onChange(next)
  }

  async function attachFromSettings(provider: ProviderSummary) {
    try {
      setAttachError(null)
      const config = await api.get<ConfigResponse>(
        `/providers/${encodeURIComponent(provider.id)}/config`,
      )
      const parsed = config.parsed_json ?? {}
      const next = {
        ...attached,
        [provider.id]: { type: provider.type, ...parsed },
      }
      updateAttached(next)

      // Auto-wire `training.provider` if empty
      if (rootValue && onRootChange) {
        const training = isRecord(rootValue.training) ? rootValue.training : {}
        if (!training.provider) {
          onRootChange({
            ...rootValue,
            training: { ...training, provider: provider.id },
            providers: next,
          })
        }
      }
      setPickOpen(null)
    } catch (exc) {
      setAttachError((exc as Error).message)
    }
  }

  function attachCustom() {
    if (!customType || !customName.trim()) return
    if (attached[customName.trim()]) {
      setAttachError(`already attached: ${customName.trim()}`)
      return
    }
    const info = typeById.get(customType)
    if (!info) return
    const next = { ...attached, [customName.trim()]: { type: customType } }
    updateAttached(next)
    setCustomName('')
    setCustomType('')
    setPickOpen(null)
    setAttachError(null)
  }

  function detach(id: string) {
    const next = { ...attached }
    delete next[id]
    updateAttached(next)
    if (expanded === id) setExpanded(null)

    // clear training.provider if it pointed at removed entry
    if (rootValue && onRootChange) {
      const training = isRecord(rootValue.training) ? rootValue.training : {}
      if (training.provider === id) {
        onRootChange({ ...rootValue, training: { ...training, provider: '' }, providers: next })
      }
    }
  }

  return (
    <div className="rounded-md border border-line-1 bg-surface-1 p-4 space-y-4">
      <div>
        <div className="text-sm font-medium text-ink-1">
          Providers {required && <span className="ml-1 text-brand-warm">*</span>}
        </div>
        <div className="text-2xs text-ink-3">
          Named compute providers. Pick one of your saved Settings providers or
          define a new one.
        </div>
      </div>

      {attachedIds.length === 0 ? (
        <div className="text-xs text-ink-3 rounded-md border border-dashed border-line-1 px-3 py-4 text-center">
          No providers attached yet.
        </div>
      ) : (
        <div className="rounded-md border border-line-1 overflow-hidden">
          <table className="w-full text-xs">
            <thead className="bg-surface-2 text-2xs uppercase tracking-wide text-ink-3">
              <tr>
                <th className="text-left font-medium px-3 py-1.5">id</th>
                <th className="text-left font-medium px-3 py-1.5">type</th>
                <th className="text-left font-medium px-3 py-1.5">source</th>
                <th className="text-right font-medium px-3 py-1.5">actions</th>
              </tr>
            </thead>
            <tbody>
              {attachedIds.map((id) => {
                const entry = isRecord(attached[id]) ? (attached[id] as Record<string, unknown>) : {}
                const type = typeof entry.type === 'string' ? entry.type : '?'
                const typeInfo = typeById.get(type)
                const knownSource = registeredById.get(id)
                const isExpanded = expanded === id
                return (
                  <>
                    <tr
                      key={id}
                      className="border-t border-line-1 hover:bg-surface-2/40 cursor-pointer"
                      onClick={() => setExpanded(isExpanded ? null : id)}
                    >
                      <td className="px-3 py-2 font-mono text-ink-1">{id}</td>
                      <td className="px-3 py-2 text-brand-alt">{typeInfo?.label ?? type}</td>
                      <td className="px-3 py-2 text-ink-3">
                        {knownSource ? `from settings (${knownSource.name})` : 'inline'}
                      </td>
                      <td className="px-3 py-2 text-right">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation()
                            detach(id)
                          }}
                          className="text-err/80 hover:text-err text-2xs"
                        >
                          detach
                        </button>
                      </td>
                    </tr>
                    {isExpanded && (
                      <tr key={`${id}-body`} className="border-t border-line-1/60 bg-surface-0/50">
                        <td colSpan={4} className="px-3 py-3">
                          {typeInfo ? (
                            <FieldRenderer
                              root={typeInfo.json_schema as PipelineJsonSchema}
                              node={typeInfo.json_schema as unknown as JsonSchemaNode}
                              value={entry}
                              onChange={(next) => {
                                const nextAll = { ...attached }
                                nextAll[id] = isRecord(next) ? { type, ...next } : next
                                updateAttached(nextAll)
                              }}
                              labelKey={id}
                              depth={1}
                            />
                          ) : (
                            <pre className="text-[0.65rem] font-mono text-ink-3 whitespace-pre-wrap break-words">
                              {JSON.stringify(entry, null, 2)}
                            </pre>
                          )}
                        </td>
                      </tr>
                    )}
                  </>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {attachError && (
        <div className="rounded-md border border-err/40 bg-err/10 text-err text-2xs px-3 py-2">
          {attachError}
        </div>
      )}

      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setPickOpen(pickOpen === 'settings' ? null : 'settings')}
          className="btn-primary px-3 py-1.5 text-xs"
          disabled={providersQuery.isLoading}
        >
          + Add from Settings
        </button>
        <button
          type="button"
          onClick={() => setPickOpen(pickOpen === 'custom' ? null : 'custom')}
          className="rounded-md border border-line-1 px-3 py-1.5 text-xs text-ink-2 hover:text-ink-1 hover:border-line-2"
        >
          + Add custom
        </button>
      </div>

      {pickOpen === 'settings' && (
        <div className="rounded-md border border-line-1 bg-surface-0 p-3 space-y-2">
          {registered.length === 0 ? (
            <div className="text-2xs text-ink-3">No providers registered yet.</div>
          ) : (
            registered.map((p) => {
              const already = attachedIds.includes(p.id)
              return (
                <button
                  key={p.id}
                  type="button"
                  disabled={already}
                  onClick={() => attachFromSettings(p)}
                  className="w-full flex items-center gap-3 rounded-md px-3 py-2 text-left text-xs border border-line-1 hover:bg-surface-2 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <div className="min-w-0 flex-1">
                    <div className="text-ink-1 font-medium">{p.name}</div>
                    <div className="text-[0.65rem] font-mono text-ink-3">{p.id}</div>
                  </div>
                  <span className="text-[0.65rem] text-brand-alt">
                    {typeById.get(p.type)?.label ?? p.type}
                  </span>
                  {already && <span className="text-[0.6rem] text-ink-4">attached</span>}
                </button>
              )
            })
          )}
          <a
            href="/settings/providers#new"
            className="block rounded-md border border-dashed border-brand-alt/60 bg-brand-alt/5 px-3 py-2 text-xs text-brand-alt hover:bg-brand-alt/10 transition"
          >
            + Create new provider in Settings →
          </a>
        </div>
      )}

      {pickOpen === 'custom' && (
        <div className="rounded-md border border-line-1 bg-surface-0 p-3 space-y-2">
          <div className="grid grid-cols-[160px_1fr_auto] gap-2">
            <select
              value={customType}
              onChange={(e) => setCustomType(e.target.value)}
              className="rounded-md bg-surface-2 border border-line-1 px-2 py-1.5 text-xs focus:outline-none focus:border-brand"
            >
              <option value="">type…</option>
              {types.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
            <input
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              placeholder="id (e.g. runpod-experimental)"
              className="rounded-md bg-surface-2 border border-line-1 px-2 py-1.5 text-xs font-mono focus:outline-none focus:border-brand"
            />
            <button
              type="button"
              onClick={attachCustom}
              disabled={!customType || !customName.trim()}
              className="btn-primary px-3 py-1.5 text-xs disabled:opacity-50"
            >
              Attach
            </button>
          </div>
          <div className="text-[0.65rem] text-ink-4">
            Defines an inline provider block in this project only. For reusable
            configs, create one in Settings → Providers.
          </div>
        </div>
      )}
    </div>
  )
}
