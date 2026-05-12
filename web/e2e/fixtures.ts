/**
 * Phase-6 Playwright fixtures.
 *
 * `webE2eStack` — pytest fixture-style helper that wraps the Python
 * orchestrator. When the env var ``RYOTENKAI_E2E_USE_STACK=1`` is set,
 * we shell out to ``python -m tests._harness.stack.orchestrator start
 * --profile=dev`` and parse the resulting control-plane URL; otherwise
 * we return the vite dev server URL (started by playwright.config's
 * ``webServer``).
 *
 * For the common laptop developer flow the MSW fallback is plenty —
 * the hermetic stack matters when we record traces that must survive
 * across releases (the replay corpus).
 */

import { test as base, expect } from '@playwright/test'
import { execSync } from 'node:child_process'

type Fixtures = {
  webE2eStack: { baseUrl: string }
}

export const test = base.extend<Fixtures>({
  webE2eStack: async ({ baseURL }, use) => {
    if (process.env.RYOTENKAI_E2E_USE_STACK === '1') {
      const output = execSync(
        'python -m tests._harness.stack.orchestrator start --profile=dev --emit-control-plane-url',
        { encoding: 'utf8', cwd: '..', timeout: 90_000 },
      )
      const url = output.trim().split('\n').filter(Boolean).pop() ?? ''
      if (!url.startsWith('http')) {
        throw new Error(`stack orchestrator returned unexpected output: ${output}`)
      }
      await use({ baseUrl: url })
      return
    }
    await use({ baseUrl: baseURL ?? 'http://127.0.0.1:5173' })
  },
})

export { expect }
