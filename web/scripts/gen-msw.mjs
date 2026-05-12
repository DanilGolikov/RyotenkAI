#!/usr/bin/env node
/**
 * MSW handler codegen for the FE contract layer (Phase 3, Contract
 * Testing Matrix, deliverable G2).
 *
 * Reads ``web/src/api/openapi.json`` and emits
 * ``web/src/api/msw_handlers.ts`` — a default-exported array of MSW v2
 * handlers covering every operation in the spec. Each handler returns
 * a deterministic, contract-shaped 200 response synthesised from the
 * response schema (``example`` → ``default`` → type-keyed zero value).
 *
 * Why a custom generator rather than ``msw-auto-mock``?
 * - ``msw-auto-mock``'s MSW v2 support has lagged the v2 release line
 *   and brings in faker as a runtime dep, which would make every test
 *   non-deterministic. Our synthesiser is small, deterministic, and
 *   never imports faker.
 * - The point isn't realistic data — it's contract-shaped placeholders
 *   for tests. A few hundred lines of generated zeros + ``[]`` is far
 *   cheaper to review than a faker pin + matching plugin churn.
 *
 * The output is byte-stable: running this twice produces an identical
 * file. CI's ``make verify-api-sync`` relies on that.
 */

import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const WEB_DIR = resolve(__dirname, '..')
const INPUT = resolve(WEB_DIR, 'src/api/openapi.json')
const OUTPUT = resolve(WEB_DIR, 'src/api/msw_handlers.ts')

const spec = JSON.parse(readFileSync(INPUT, 'utf8'))
const components = spec.components?.schemas ?? {}

/**
 * Resolve a ``$ref`` string against the spec. We only follow
 * ``#/components/schemas/<Name>`` refs — anything else falls back to
 * the empty object (synthesised as ``null``).
 */
function resolveRef(ref) {
  const prefix = '#/components/schemas/'
  if (!ref.startsWith(prefix)) return {}
  const name = ref.slice(prefix.length)
  return components[name] ?? {}
}

/**
 * Synthesise a deterministic example value for a JSON Schema fragment.
 *
 * Resolution order, mirroring how openapi-typescript and msw-auto-mock
 * resolve example sources, but stripped down to what we need:
 *   1. ``example`` — explicit example provided by the API author.
 *   2. ``default`` — the schema's documented default.
 *   3. ``const``  — single-allowed-value schemas.
 *   4. First entry of ``enum`` — closest thing to a "canonical" value.
 *   5. ``anyOf`` / ``oneOf`` first branch — recurse so e.g. nullable
 *      strings produce ``""``, not ``null``, when ``""`` is on the
 *      schema's "happy path".
 *   6. ``$ref`` — recurse into the referenced schema.
 *   7. Type-keyed zero value: ``string`` → ``""``, ``number`` → ``0``,
 *      ``integer`` → ``0``, ``boolean`` → ``false``, ``array`` → ``[]``,
 *      ``object`` → recurse over ``properties`` (only ``required`` keys
 *      so the synthesised object stays minimal).
 *   8. Fallback: ``null``.
 *
 * ``depth`` caps recursion at a small bound — some OpenAPI specs have
 * cycles (e.g. tree structures) and we'd otherwise infinite-loop. At
 * the limit we return ``null`` rather than synthesise garbage.
 */
function synth(schema, depth = 0) {
  if (!schema || typeof schema !== 'object' || depth > 6) return null

  if (schema.example !== undefined) return schema.example
  if (schema.default !== undefined) return schema.default
  if (schema.const !== undefined) return schema.const
  if (Array.isArray(schema.enum) && schema.enum.length > 0) return schema.enum[0]

  if (schema.$ref) return synth(resolveRef(schema.$ref), depth + 1)

  if (Array.isArray(schema.anyOf) && schema.anyOf.length > 0) {
    // Prefer the first non-null branch so nullable fields synthesise
    // a real value (the contract still allows null, but tests reading
    // ``response.field.something`` work without extra guards).
    const nonNull = schema.anyOf.find(
      (s) => !(s && typeof s === 'object' && s.type === 'null'),
    )
    return synth(nonNull ?? schema.anyOf[0], depth + 1)
  }
  if (Array.isArray(schema.oneOf) && schema.oneOf.length > 0) {
    return synth(schema.oneOf[0], depth + 1)
  }
  if (Array.isArray(schema.allOf) && schema.allOf.length > 0) {
    return Object.assign(
      {},
      ...schema.allOf.map((s) => synth(s, depth + 1) ?? {}),
    )
  }

  // ``type`` may be a string or an array (OpenAPI 3.1). Pick the first
  // non-null type so we get a concrete shape.
  const types = Array.isArray(schema.type) ? schema.type : schema.type ? [schema.type] : []
  const primaryType = types.find((t) => t !== 'null') ?? types[0]

  switch (primaryType) {
    case 'string':
      if (schema.format === 'date-time') return '2026-01-01T00:00:00Z'
      if (schema.format === 'date') return '2026-01-01'
      if (schema.format === 'uuid') return '00000000-0000-0000-0000-000000000000'
      return ''
    case 'number':
    case 'integer':
      return 0
    case 'boolean':
      return false
    case 'array':
      return []
    case 'object': {
      const out = {}
      const props = schema.properties ?? {}
      const required = Array.isArray(schema.required) ? schema.required : Object.keys(props)
      for (const key of required) {
        if (props[key]) out[key] = synth(props[key], depth + 1)
      }
      return out
    }
    case 'null':
      return null
    default:
      // Schema-less or unknown — return ``null`` so callers know it's a
      // gap rather than a real value.
      return null
  }
}

/**
 * Pick the success response body schema for an operation. Prefers
 * ``200``, then ``201``/``202``/``204`` (in that order), then
 * ``default``, finally the first response key whatever it is.
 */
function pickResponseSchema(operation) {
  const responses = operation.responses ?? {}
  const preferred = ['200', '201', '202', '204', 'default']
  let response
  for (const code of preferred) {
    if (responses[code]) {
      response = responses[code]
      break
    }
  }
  if (!response) {
    const firstCode = Object.keys(responses)[0]
    if (firstCode) response = responses[firstCode]
  }
  if (!response) return null
  return response.content?.['application/json']?.schema ?? null
}

/**
 * Turn ``/api/v1/projects/{project_id}``-style paths into MSW v2's
 * ``:project_id`` placeholders so the handler matches the actual
 * fetched URL.
 */
function toMswPath(path) {
  return path.replace(/\{([^}]+)\}/g, ':$1')
}

const HTTP_METHODS = new Set(['get', 'post', 'put', 'patch', 'delete', 'options', 'head'])

const handlers = []
for (const [path, item] of Object.entries(spec.paths ?? {})) {
  for (const [method, op] of Object.entries(item)) {
    if (!HTTP_METHODS.has(method)) continue
    const schema = pickResponseSchema(op)
    const body = synth(schema)
    handlers.push({
      method,
      path: toMswPath(path),
      operationId: op.operationId ?? `${method.toUpperCase()} ${path}`,
      body,
    })
  }
}

const BANNER = `/* eslint-disable */
// AUTO-GENERATED — DO NOT EDIT BY HAND.
// Regenerate with: \`make regen-msw\` (or \`cd web && npm run gen:msw\`).
// Source: web/src/api/openapi.json (produced by scripts/sync_openapi.py).
//
// Each handler returns a deterministic, contract-shaped 200 response
// synthesised from the response schema (example > default > zero value).
// Override per-test by passing an additional handler to MSW's
// \`server.use(...)\` — the latest registration wins.
`

const fileLines = [
  BANNER,
  `import { http, HttpResponse } from 'msw'`,
  '',
  `export const handlers = [`,
]

for (const h of handlers) {
  const bodyLiteral = JSON.stringify(h.body)
  fileLines.push(`  // ${h.operationId}`)
  fileLines.push(
    `  http.${h.method}(${JSON.stringify(h.path)}, () => HttpResponse.json(${bodyLiteral})),`,
  )
}

fileLines.push(`]`)
fileLines.push('')

writeFileSync(OUTPUT, fileLines.join('\n'), 'utf8')
console.log(`[gen-msw] wrote ${OUTPUT} (${handlers.length} handlers)`)
