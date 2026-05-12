/**
 * Playwright configuration — phase-6 E2E + replay corpus.
 *
 * Two ways to run:
 *
 *   1) Against the hermetic stack (`make start-stack`):
 *      BASE_URL=http://127.0.0.1:<port> npm run test:e2e
 *
 *   2) Against the vite dev server with MSW handlers (fallback when
 *      booting the stack is too heavyweight — e.g. on a laptop
 *      without docker daemon). The dev server already wires MSW.
 *      The `webServer` block below brings it up automatically.
 *
 * Only Chromium is enabled. Firefox/WebKit roughly triple CI time
 * for marginal coverage gain; we'll add them when a real customer
 * needs WebKit support.
 */

import { defineConfig, devices } from '@playwright/test'

const baseURL = process.env.BASE_URL ?? 'http://127.0.0.1:5173'

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : [['list']],
  use: {
    baseURL,
    // trace-on-first-retry — local runs stay fast; CI gets a trace on
    // the second attempt so flakes are reproducible without paying
    // the cost on every green run.
    trace: 'on-first-retry',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  // Phase-6 fallback: spin up `vite` against MSW if no external BASE_URL
  // is set. The hermetic-stack path overrides this by passing BASE_URL
  // directly to `playwright test`.
  webServer: process.env.BASE_URL
    ? undefined
    : {
        command: 'npm run dev -- --port=5173 --strictPort',
        port: 5173,
        reuseExistingServer: !process.env.CI,
        timeout: 60_000,
      },
})
