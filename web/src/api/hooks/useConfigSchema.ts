import { useQuery } from '@tanstack/react-query'
import { api } from '../client'
import { qk } from '../queryKeys'

export interface JsonSchemaNode {
  type?: string | string[]
  title?: string
  description?: string
  default?: unknown
  enum?: unknown[]
  properties?: Record<string, JsonSchemaNode>
  required?: string[]
  items?: JsonSchemaNode
  anyOf?: JsonSchemaNode[]
  oneOf?: JsonSchemaNode[]
  allOf?: JsonSchemaNode[]
  $ref?: string
  [key: string]: unknown
}

export interface PipelineJsonSchema extends JsonSchemaNode {
  $defs?: Record<string, JsonSchemaNode>
  definitions?: Record<string, JsonSchemaNode>
}

export function useConfigSchema() {
  return useQuery({
    queryKey: qk.configSchema(),
    queryFn: () => api.get<PipelineJsonSchema>('/config/schema'),
    staleTime: 60 * 60 * 1000, // 1h — schema rarely changes at runtime
  })
}
