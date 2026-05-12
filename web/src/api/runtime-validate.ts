/**
 * Runtime contract validation for API responses.
 *
 * Phase 3 (Contract Testing Matrix) deliverable D5. Wraps a Zod schema's
 * ``safeParse`` so callers can validate inbound payloads against the
 * generated schema (``zod.ts``) without writing ad-hoc guards.
 *
 * Failure semantics:
 * - Throws ``ApiContractError`` containing the Zod issues so the caller
 *   (e.g. React Query) surfaces a typed boundary rather than a generic
 *   ``TypeError`` deep in render code.
 * - Logs to ``console.error`` in development (``import.meta.env.DEV``)
 *   so the violation shows up in the browser console alongside the
 *   request — handy when the backend ships a contract-breaking change.
 *
 * The guard is intentionally *not* enabled by default for every endpoint.
 * Wire it in at call sites where a contract mismatch should be surfaced
 * loudly (high-traffic GETs, anything driving navigation). See the
 * ``useProjects`` hook for the Phase 3 demo wiring.
 */

import type { ZodIssue, ZodType } from 'zod'

export class ApiContractError extends Error {
  readonly issues: ReadonlyArray<ZodIssue>

  constructor(message: string, issues: ReadonlyArray<ZodIssue>) {
    super(message)
    this.name = 'ApiContractError'
    this.issues = issues
  }
}

interface ImportMetaEnv {
  readonly DEV?: boolean
}

interface ImportMetaWithEnv {
  readonly env?: ImportMetaEnv
}

function isDev(): boolean {
  // ``import.meta.env`` is provided by Vite; in vitest/node it may be
  // ``undefined``. Guard so this module loads in plain Node too (for
  // codegen smoke tests that import it without bundling).
  try {
    const meta = import.meta as ImportMetaWithEnv
    return Boolean(meta.env?.DEV)
  } catch {
    return false
  }
}

/**
 * Validate ``data`` against ``schema``.
 *
 * Returns the parsed value (typed as the schema's inferred type) on
 * success, throws ``ApiContractError`` on failure. The schema's output
 * type is preferred because Zod transforms may widen the input.
 */
export function validateResponse<S extends ZodType>(
  schema: S,
  data: unknown,
  context = 'response',
): ReturnType<S['parse']> {
  const result = schema.safeParse(data)
  if (result.success) {
    return result.data as ReturnType<S['parse']>
  }
  if (isDev()) {
    // eslint-disable-next-line no-console
    console.error(
      `[runtime-validate] ${context} failed contract validation:`,
      result.error.issues,
    )
  }
  throw new ApiContractError(
    `${context} did not match contract: ${result.error.issues.length} issue(s)`,
    result.error.issues,
  )
}
