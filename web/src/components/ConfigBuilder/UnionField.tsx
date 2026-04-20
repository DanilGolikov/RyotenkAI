import { useMemo, useState } from 'react'
import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { prettifyTitle, resolveRef } from './schemaUtils'
import { SelectField } from './SelectField'

function branchLabel(node: JsonSchemaNode, fallbackIndex: number): string {
  const typeProp = (node.properties as Record<string, JsonSchemaNode> | undefined)?.type
  const c = typeProp && 'const' in typeProp ? (typeProp as { const?: unknown }).const : undefined
  if (typeof c === 'string' && c) return c
  if (typeof node.title === 'string' && node.title) return prettifyTitle(node.title)
  return `Option ${fallbackIndex + 1}`
}

function branchDiscriminatorValue(node: JsonSchemaNode): string | undefined {
  const typeProp = (node.properties as Record<string, JsonSchemaNode> | undefined)?.type
  if (!typeProp || !('const' in typeProp)) return undefined
  const c = (typeProp as { const?: unknown }).const
  return typeof c === 'string' ? c : undefined
}

function pickInitialBranch(branches: JsonSchemaNode[], value: unknown): number {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return 0
  const record = value as Record<string, unknown>
  const currentType = typeof record.type === 'string' ? record.type : undefined
  if (currentType) {
    const i = branches.findIndex((b) => branchDiscriminatorValue(b) === currentType)
    if (i >= 0) return i
  }
  const keys = Object.keys(record)
  if (keys.length === 0) return 0
  let best = 0
  let bestScore = -1
  branches.forEach((b, idx) => {
    const props = (b.properties as Record<string, unknown> | undefined) ?? {}
    const score = keys.filter((k) => k in props).length
    if (score > bestScore) {
      bestScore = score
      best = idx
    }
  })
  return best
}

export interface UnionFieldProps {
  root: PipelineJsonSchema
  branches: JsonSchemaNode[]
  value: unknown
  onChange: (value: unknown) => void
  label: string
  required?: boolean
  renderBranch: (branch: JsonSchemaNode) => React.ReactNode
}

export function UnionField({
  root,
  branches,
  value,
  onChange,
  label,
  required,
  renderBranch,
}: UnionFieldProps) {
  const resolved = useMemo(() => branches.map((b) => resolveRef(root, b)), [root, branches])
  const activeIdx = pickInitialBranch(resolved, value)
  const [open, setOpen] = useState(false)

  function switchBranch(nextIdx: number) {
    const nextBranch = resolved[nextIdx]
    const current = (value && typeof value === 'object' && !Array.isArray(value)
      ? { ...(value as Record<string, unknown>) }
      : {}) as Record<string, unknown>
    const nextProps = (nextBranch.properties as Record<string, JsonSchemaNode> | undefined) ?? {}
    const preserved: Record<string, unknown> = {}
    for (const key of Object.keys(nextProps)) {
      if (key in current) preserved[key] = current[key]
    }
    const disc = branchDiscriminatorValue(nextBranch)
    if (disc) preserved.type = disc
    onChange(preserved)
  }

  return (
    <div className="relative rounded border border-line-1 bg-surface-1">
      {open && (
        <div
          aria-hidden
          className="absolute left-0 inset-y-1 w-0.5 bg-gradient-brand rounded-full pointer-events-none"
        />
      )}
      <div
        role="button"
        tabIndex={-1}
        onClick={(e) => {
          // Only toggle when the header background itself (or the label
          // button) is clicked — not the inner branch selector.
          if ((e.target as HTMLElement).closest('[data-no-toggle]')) return
          setOpen((v) => !v)
        }}
        className="flex items-center gap-3 px-4 py-2.5 cursor-pointer hover:bg-surface-2/60 transition-colors"
      >
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation()
            setOpen((v) => !v)
          }}
          aria-expanded={open}
          className="flex items-center gap-2 text-left"
        >
          <span
            aria-hidden
            className={`text-ink-3 text-[10px] transition-transform ${open ? 'rotate-90' : ''}`}
          >
            ▸
          </span>
          <span className="text-xs text-ink-1 font-medium">
            {label}
            {required ? <span className="text-brand ml-1">*</span> : null}
          </span>
        </button>
        <div className="ml-auto" data-no-toggle>
          <SelectField
            value={String(activeIdx)}
            options={resolved.map((b, idx) => ({
              value: String(idx),
              label: branchLabel(b, idx),
            }))}
            onChange={(next) => switchBranch(Number.parseInt(next, 10))}
          />
        </div>
      </div>
      {open && (
        <div className="px-4 pb-3 pt-2 border-t border-line-1 space-y-3">
          {renderBranch(resolved[activeIdx])}
        </div>
      )}
    </div>
  )
}
