import { useEffect, useMemo, useState } from 'react'
import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { FieldRenderer } from './FieldRenderer'
import { FieldSearchOmniBox } from './FieldSearchOmniBox'
import type { GroupRendererProps } from './ProviderPickerField'
import { topLevelLabel, visibleTopLevelKeys } from './schemaUtils'
import { TocRail, type GroupValidity } from './TocRail'

export interface ConfigBuilderProps {
  schema: PipelineJsonSchema
  value: Record<string, unknown>
  onChange: (next: Record<string, unknown>) => void
  /** Validity dot per top-level group, sourced from the last validation result. */
  groupValidity?: Partial<Record<string, GroupValidity>>
  /** Which hash namespace to sync (so multiple builders can coexist). */
  hashPrefix?: string
  /** Per-top-level-group custom renderers (e.g. providers → ProviderPickerField). */
  groupRenderers?: Partial<Record<string, React.ComponentType<GroupRendererProps>>>
}

function readInitialGroup(keys: string[], hashPrefix: string): string {
  if (typeof window === 'undefined' || !window.location.hash) {
    return keys[0] ?? ''
  }
  const raw = window.location.hash.slice(1)
  const prefix = hashPrefix ? `${hashPrefix}:` : ''
  if (prefix && !raw.startsWith(prefix)) return keys[0] ?? ''
  const tail = prefix ? raw.slice(prefix.length) : raw
  const [group] = tail.split('.')
  return keys.includes(group) ? group : keys[0] ?? ''
}

function readDottedTrailer(hashPrefix: string): string | null {
  if (typeof window === 'undefined' || !window.location.hash) return null
  const raw = window.location.hash.slice(1)
  const prefix = hashPrefix ? `${hashPrefix}:` : ''
  if (prefix && !raw.startsWith(prefix)) return null
  const tail = prefix ? raw.slice(prefix.length) : raw
  return tail.includes('.') ? tail : null
}

export function ConfigBuilder({
  schema,
  value,
  onChange,
  groupValidity,
  hashPrefix = '',
  groupRenderers,
}: ConfigBuilderProps) {
  const topProps = (schema.properties ?? {}) as Record<string, JsonSchemaNode>
  const topRequired = new Set<string>(Array.isArray(schema.required) ? schema.required : [])
  const keys = useMemo(() => visibleTopLevelKeys(Object.keys(topProps)), [topProps])

  const [active, setActive] = useState<string>(() => readInitialGroup(keys, hashPrefix))

  // Keep `active` in sync with history navigation (e.g. deep-link, back button).
  useEffect(() => {
    function onHash() {
      setActive(readInitialGroup(keys, hashPrefix))
      // If the hash has a dotted trailer like "training.lora.r", try to scroll
      // to the matching FieldAnchor once the group is mounted.
      const trailer = readDottedTrailer(hashPrefix)
      if (trailer) {
        setTimeout(() => {
          const anchor = document.querySelector(`[data-field-path="${CSS.escape(trailer)}"]`)
          if (anchor) anchor.scrollIntoView({ behavior: 'smooth', block: 'center' })
        }, 100)
      }
    }
    window.addEventListener('hashchange', onHash)
    onHash()
    return () => window.removeEventListener('hashchange', onHash)
  }, [keys, hashPrefix])

  function selectGroup(key: string) {
    setActive(key)
    const prefix = hashPrefix ? `${hashPrefix}:` : ''
    const nextHash = `#${prefix}${key}`
    if (window.location.hash !== nextHash) {
      // Use replaceState so we don't spam history stack.
      history.replaceState(null, '', nextHash)
    }
  }

  if (keys.length === 0) {
    return <div className="text-xs text-ink-3">Empty schema.</div>
  }

  const activeKey = keys.includes(active) ? active : keys[0]
  const activeNode = topProps[activeKey]
  const activeNodeLabelled: JsonSchemaNode = activeNode
    ? { ...activeNode, title: topLevelLabel(activeKey, (activeNode as { title?: string }).title) }
    : activeNode

  // Negative margins bleed the rail's darker surface-1 all the way to the
  // outer Card edge — ProjectDetail.tsx wraps this in a `<div class="p-5">`,
  // so -m-5 cancels that padding. If the outer padding changes, update here.
  return (
    <div className="grid grid-cols-[208px_1fr] -mx-5 -my-5">
      <aside className="min-w-0 bg-surface-1 border-r border-line-1 px-3 py-5">
        <TocRail
          schema={schema}
          active={activeKey}
          onSelect={selectGroup}
          validity={groupValidity}
        />
      </aside>
      <div className="min-w-0 space-y-4 px-5 py-5">
        <section id={`cfg-${activeKey}`} className="min-w-0 scroll-mt-24">
          {renderActive()}
        </section>
        <FieldSearchOmniBox schema={schema} hashPrefix={hashPrefix} />
      </div>
    </div>
  )

  function renderActive() {
    const Custom = groupRenderers?.[activeKey]
    const setKey = (next: unknown) => {
      const copy = { ...value }
      if (next === undefined) delete copy[activeKey]
      else copy[activeKey] = next
      onChange(copy)
    }
    if (Custom) {
      return (
        <Custom
          root={schema}
          node={activeNodeLabelled}
          value={value?.[activeKey]}
          labelKey={activeKey}
          required={topRequired.has(activeKey)}
          onChange={setKey}
          rootValue={value}
          onRootChange={onChange}
        />
      )
    }
    return (
      <FieldRenderer
        root={schema}
        node={activeNodeLabelled}
        value={value?.[activeKey]}
        labelKey={activeKey}
        required={topRequired.has(activeKey)}
        depth={0}
        onChange={setKey}
        path={activeKey}
        hashPrefix={hashPrefix}
      />
    )
  }
}
