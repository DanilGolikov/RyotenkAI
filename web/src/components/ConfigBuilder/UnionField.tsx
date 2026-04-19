import { useMemo } from 'react'
import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'
import { resolveRef } from './schemaUtils'

function branchLabel(node: JsonSchemaNode, fallbackIndex: number): string {
  const typeProp = (node.properties as Record<string, JsonSchemaNode> | undefined)?.type
  const c = typeProp && 'const' in typeProp ? (typeProp as { const?: unknown }).const : undefined
  if (typeof c === 'string' && c) return c
  if (typeof node.title === 'string' && node.title) return node.title
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
    <div className="rounded-md border border-line-1 bg-surface-1 p-4 space-y-3">
      <div className="flex items-baseline gap-3">
        <div className="text-2xs text-ink-2 font-medium">
          {label}
          {required ? <span className="text-brand ml-1">*</span> : null}
        </div>
        <select
          value={activeIdx}
          onChange={(e) => switchBranch(Number.parseInt(e.target.value, 10))}
          className="ml-auto rounded-md bg-surface-2 border border-line-1 px-2 py-1 text-xs font-mono focus:outline-none focus:border-brand"
        >
          {resolved.map((b, idx) => (
            <option key={idx} value={idx}>
              {branchLabel(b, idx)}
            </option>
          ))}
        </select>
      </div>
      <div className="pt-2 border-t border-line-1">{renderBranch(resolved[activeIdx])}</div>
    </div>
  )
}
