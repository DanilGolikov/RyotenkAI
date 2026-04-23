import { useEffect, useRef } from 'react'
import { useClickOutside } from '../hooks/useClickOutside'
import type { PluginManifest } from '../api/types'

interface Props {
  plugin: PluginManifest
  onClose: () => void
}

type JsonSchemaObject = {
  type?: string
  properties?: Record<string, Record<string, unknown>>
  required?: string[]
  additionalProperties?: boolean
}

/** Tiny schema-viewer — renders the ``params_schema`` /
 *  ``thresholds_schema`` JSON Schema object that the backend emits for
 *  each plugin. Not a form: this is read-only information so the user
 *  can understand what a plugin accepts before adding it to a project.
 *  Each field shows its type, default, description, constraints, and
 *  a ``required`` badge when applicable. */
function SchemaTable({ schema, emptyHint }: { schema: JsonSchemaObject; emptyHint: string }) {
  const props = schema.properties ?? {}
  const required = new Set(schema.required ?? [])
  const keys = Object.keys(props)

  if (keys.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-line-2 px-3 py-2 text-2xs text-ink-3">
        {emptyHint}
      </div>
    )
  }

  return (
    <div className="rounded-md border border-line-1 bg-surface-0 divide-y divide-line-1">
      {keys.map((key) => {
        const field = props[key]
        const isRequired = required.has(key)
        return (
          <div key={key} className="px-3 py-2 space-y-1">
            <div className="flex items-baseline gap-2 flex-wrap">
              <code className="font-mono text-xs text-ink-1">{key}</code>
              <span className="text-[0.6rem] uppercase tracking-wide text-ink-3">
                {renderType(field)}
              </span>
              {isRequired && (
                <span
                  className="text-[0.6rem] uppercase tracking-wide text-err"
                  title="This field has no default — the user must fill it in."
                >
                  required
                </span>
              )}
              {field['x-secret'] === true && (
                <span
                  className="text-[0.6rem] uppercase tracking-wide text-warn"
                  title="Rendered as a password input; never logged."
                >
                  secret
                </span>
              )}
            </div>
            {typeof field.title === 'string' && field.title && (
              <div className="text-xs text-ink-2">{field.title}</div>
            )}
            {typeof field.description === 'string' && field.description && (
              <div className="text-2xs text-ink-3 leading-snug">{field.description}</div>
            )}
            {renderConstraints(field)}
            {field.default !== undefined && (
              <div className="text-2xs text-ink-3">
                <span className="text-ink-4">default:</span>{' '}
                <code className="font-mono text-ink-2">{formatValue(field.default)}</code>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function renderType(field: Record<string, unknown>): string {
  const enumList = field.enum as unknown[] | undefined
  if (Array.isArray(enumList) && enumList.length > 0) return `enum (${enumList.length})`
  return String(field.type ?? 'unknown')
}

function renderConstraints(field: Record<string, unknown>) {
  const parts: string[] = []
  if (typeof field.minimum === 'number') parts.push(`min ${field.minimum}`)
  if (typeof field.maximum === 'number') parts.push(`max ${field.maximum}`)
  const enumList = field.enum as unknown[] | undefined
  if (Array.isArray(enumList)) {
    parts.push(`one of: ${enumList.map((v) => formatValue(v)).join(' · ')}`)
  }
  if (parts.length === 0) return null
  return <div className="text-2xs text-ink-4 font-mono">{parts.join('  ·  ')}</div>
}

function formatValue(v: unknown): string {
  if (typeof v === 'string') return JSON.stringify(v)
  return String(v)
}

const STABILITY_CLS: Record<string, string> = {
  stable: 'text-ok border-ok/40 bg-ok/10',
  beta: 'text-warn border-warn/40 bg-warn/10',
  experimental: 'text-err border-err/40 bg-err/10',
}

/**
 * Modal shown from the Settings/Catalog page when the user clicks the
 * info button on a plugin card. Read-only: schema tables for params +
 * thresholds, description, metadata, example YAML snippet.
 */
export function PluginInfoModal({ plugin, onClose }: Props) {
  const panelRef = useRef<HTMLDivElement | null>(null)
  useClickOutside(panelRef, true, onClose)

  // Trap focus entry point on mount; Esc close comes from useClickOutside
  // plus a dedicated keydown listener.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const paramsSchema = (plugin.params_schema ?? {}) as JsonSchemaObject
  const thresholdsSchema = (plugin.thresholds_schema ?? {}) as JsonSchemaObject

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="plugin-info-title"
    >
      <div
        ref={panelRef}
        className="w-full max-w-2xl max-h-[85vh] overflow-hidden rounded-xl border border-line-2 bg-surface-1 shadow-card flex flex-col"
      >
        <div className="px-4 py-3 border-b border-line-1 flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div id="plugin-info-title" className="text-sm font-semibold text-ink-1 truncate">
              {plugin.name || plugin.id}
            </div>
            <div className="text-2xs text-ink-3 flex items-center gap-2 flex-wrap mt-0.5">
              <code className="font-mono text-ink-2">{plugin.id}</code>
              <span className="text-ink-4">·</span>
              <span>{plugin.kind}</span>
              <span className="text-ink-4">·</span>
              <span>v{plugin.version}</span>
              {plugin.category && (
                <>
                  <span className="text-ink-4">·</span>
                  <span>{plugin.category}</span>
                </>
              )}
              {plugin.stability && (
                <span
                  className={`inline-flex items-center rounded border px-1.5 py-0 text-[0.6rem] uppercase tracking-wide ${STABILITY_CLS[plugin.stability] ?? 'text-ink-3 border-line-2'}`}
                  title={`Stability: ${plugin.stability}`}
                >
                  {plugin.stability}
                </span>
              )}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-ink-3 hover:text-ink-1 text-xs"
            title="Close (Esc)"
          >
            close
          </button>
        </div>

        <div className="p-4 text-xs space-y-4 overflow-y-auto">
          {plugin.description && (
            <div className="text-xs text-ink-2 leading-snug">{plugin.description}</div>
          )}

          {plugin.kind === 'reward' && (
            <div className="rounded-md border border-brand-alt/30 bg-brand-alt/10 px-3 py-2 space-y-1">
              <div className="text-2xs font-semibold text-brand-alt">Compatible strategies</div>
              {plugin.supported_strategies && plugin.supported_strategies.length > 0 ? (
                <div className="flex flex-wrap gap-1">
                  {plugin.supported_strategies.map((s) => (
                    <code
                      key={s}
                      className="inline-flex items-center rounded border border-brand-alt/30 bg-surface-1 px-1.5 py-0.5 text-[0.65rem] font-mono text-brand-alt"
                    >
                      {s}
                    </code>
                  ))}
                </div>
              ) : (
                <div className="text-2xs text-ink-3">none declared</div>
              )}
            </div>
          )}

          <Section title="Parameters" count={Object.keys(paramsSchema.properties ?? {}).length}>
            <SchemaTable
              schema={paramsSchema}
              emptyHint="This plugin accepts no parameters."
            />
          </Section>

          <Section
            title="Thresholds"
            count={Object.keys(thresholdsSchema.properties ?? {}).length}
          >
            <SchemaTable
              schema={thresholdsSchema}
              emptyHint="This plugin has no threshold fields."
            />
          </Section>

          {(Object.keys(plugin.suggested_params ?? {}).length > 0
            || Object.keys(plugin.suggested_thresholds ?? {}).length > 0) && (
            <Section title="Suggested defaults" count={null}>
              <pre className="rounded-md border border-line-1 bg-surface-0 p-3 text-2xs font-mono whitespace-pre overflow-x-auto">
                {buildYamlSnippet(plugin)}
              </pre>
            </Section>
          )}
        </div>

        <div className="px-4 py-3 border-t border-line-1 flex items-center justify-end">
          <button
            type="button"
            onClick={onClose}
            className="btn-ghost h-8 text-xs"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

function Section({
  title,
  count,
  children,
}: {
  title: string
  count: number | null
  children: React.ReactNode
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline gap-2 border-b border-line-1 pb-1">
        <span className="text-xs font-semibold text-ink-1">
          {title}
          {count !== null && (
            <>
              <span className="text-ink-4"> · </span>
              <span className="text-ink-3">{count}</span>
            </>
          )}
        </span>
      </div>
      {children}
    </div>
  )
}

/** Tiny YAML-ish snippet to show users how a plugin is referenced in a
 *  pipeline config. Intentionally lo-fi — readable format without
 *  pulling in a YAML stringifier. */
function buildYamlSnippet(plugin: PluginManifest): string {
  const lines: string[] = [`plugin: ${plugin.id}`]
  const sp = plugin.suggested_params ?? {}
  const st = plugin.suggested_thresholds ?? {}
  if (Object.keys(sp).length > 0) {
    lines.push('params:')
    for (const [k, v] of Object.entries(sp)) {
      lines.push(`  ${k}: ${yamlValue(v)}`)
    }
  }
  if (Object.keys(st).length > 0) {
    lines.push('thresholds:')
    for (const [k, v] of Object.entries(st)) {
      lines.push(`  ${k}: ${yamlValue(v)}`)
    }
  }
  return lines.join('\n')
}

function yamlValue(v: unknown): string {
  if (typeof v === 'string') return v
  if (v === null || v === undefined) return 'null'
  if (typeof v === 'number' || typeof v === 'boolean') return String(v)
  return JSON.stringify(v)
}
