import { useEffect, useMemo, useState } from 'react'
import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { FieldRenderer } from './FieldRenderer'
import type { GroupRendererProps } from './ProviderPickerField'
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
  const keys = useMemo(() => Object.keys(topProps), [topProps])

  const [active, setActive] = useState<string>(() => readInitialGroup(keys, hashPrefix))

  // Keep `active` in sync with history navigation (e.g. deep-link, back button).
  useEffect(() => {
    function onHash() {
      setActive(readInitialGroup(keys, hashPrefix))
    }
    window.addEventListener('hashchange', onHash)
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

  return (
    <div className="grid grid-cols-[220px_1fr] gap-6">
      <TocRail
        schema={schema}
        active={activeKey}
        onSelect={selectGroup}
        validity={groupValidity}
      />

      <section id={`cfg-${activeKey}`} className="min-w-0 scroll-mt-24">
        {renderActive()}
      </section>
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
          node={activeNode}
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
        node={activeNode}
        value={value?.[activeKey]}
        labelKey={activeKey}
        required={topRequired.has(activeKey)}
        depth={0}
        onChange={setKey}
      />
    )
  }
}
