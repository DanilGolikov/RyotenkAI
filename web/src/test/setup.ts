/**
 * MSW test-environment setup (Phase 3, Contract Testing Matrix, G2).
 *
 * Boots an MSW Node server with the auto-generated, contract-shaped
 * handlers from ``src/api/msw_handlers.ts``. Component tests that fetch
 * via the existing ``api`` client get deterministic, schema-conformant
 * responses with no extra ceremony.
 *
 * Per-test overrides are encouraged: register additional handlers via
 * ``server.use(...)`` inside ``beforeEach`` / individual tests. MSW
 * picks the most-recently-registered handler that matches a request,
 * so overrides win cleanly. ``afterEach(server.resetHandlers)`` undoes
 * those overrides so tests can't leak state into each other.
 *
 * This file is referenced ADDITIVELY by ``vitest.config.ts`` alongside
 * the original ``vitest.setup.ts`` — we don't want to drop the
 * jest-dom matcher setup or the ``matchMedia`` stub.
 */

import { afterAll, afterEach, beforeAll } from 'vitest'
import { setupServer } from 'msw/node'
import { handlers } from '../api/msw_handlers'

export const server = setupServer(...handlers)

beforeAll(() => {
  // ``onUnhandledRequest: 'error'`` would be ideal but the existing
  // smoke test (``harness.test.tsx``) makes no network calls, so any
  // mode is fine. We choose ``warn`` to make missed-coverage gaps
  // visible without failing unrelated tests.
  server.listen({ onUnhandledRequest: 'warn' })
})

afterEach(() => {
  server.resetHandlers()
})

afterAll(() => {
  server.close()
})
