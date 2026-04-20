import type { JsonSchemaNode, PipelineJsonSchema } from '../../api/hooks/useConfigSchema'

/**
 * Resolve a single ``$ref`` step. Supports only JSON-pointer refs into the
 * schema root (``#/$defs/Foo`` and the Pydantic alias ``#/definitions/Foo``).
 * Returns the target node or the original when not resolvable.
 */
export function resolveRef(root: PipelineJsonSchema, node: JsonSchemaNode): JsonSchemaNode {
  if (!node || typeof node !== 'object') return node
  const ref = node.$ref
  if (typeof ref !== 'string' || !ref.startsWith('#/')) return node
  const segments = ref.slice(2).split('/')
  let cursor: unknown = root
  for (const seg of segments) {
    if (cursor && typeof cursor === 'object' && seg in (cursor as Record<string, unknown>)) {
      cursor = (cursor as Record<string, unknown>)[seg]
    } else {
      return node
    }
  }
  if (cursor && typeof cursor === 'object') {
    return { ...(cursor as JsonSchemaNode), ...node, $ref: undefined }
  }
  return node
}

export type FieldKind =
  | 'string'
  | 'number'
  | 'integer'
  | 'boolean'
  | 'enum'
  | 'object'
  | 'array'
  | 'union'
  | 'unknown'

export function detectKind(node: JsonSchemaNode): FieldKind {
  if (!node) return 'unknown'
  if (Array.isArray(node.enum) && node.enum.length > 0) return 'enum'
  const type = Array.isArray(node.type) ? node.type[0] : node.type
  if (type === 'object') return 'object'
  if (type === 'array') return 'array'
  if (type === 'integer') return 'integer'
  if (type === 'number') return 'number'
  if (type === 'boolean') return 'boolean'
  if (type === 'string') return 'string'
  if (Array.isArray(node.anyOf) && node.anyOf.length > 0) {
    const consts = node.anyOf.filter((n) => 'const' in n)
    if (consts.length === node.anyOf.length) return 'enum'
    return 'union'
  }
  return 'unknown'
}

export function titleOrKey(node: JsonSchemaNode, key: string): string {
  return typeof node?.title === 'string' && node.title.length > 0 ? node.title : key
}

export function getDefault(node: JsonSchemaNode): unknown {
  return Object.prototype.hasOwnProperty.call(node ?? {}, 'default') ? node.default : undefined
}

/**
 * Preferred ordering for PipelineConfig top-level groups in the UI. Keys
 * not listed here fall through alphabetically at the end so unknown
 * future groups still render. Applied by ConfigBuilder + GroupSubtabs +
 * TocRail via {@link orderTopLevelKeys}.
 */
const TOP_LEVEL_ORDER = [
  'model',
  'training',
  'inference',
  'evaluation',
  'experiment_tracking',
  'datasets',
  'providers',
]

export function orderTopLevelKeys(keys: string[]): string[] {
  const index = new Map(TOP_LEVEL_ORDER.map((k, i) => [k, i]))
  return [...keys].sort((a, b) => {
    const ai = index.get(a) ?? TOP_LEVEL_ORDER.length
    const bi = index.get(b) ?? TOP_LEVEL_ORDER.length
    if (ai !== bi) return ai - bi
    return a.localeCompare(b)
  })
}
