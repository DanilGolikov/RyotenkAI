/**
 * E2E: user opens the log dock for a running run.
 */

import path from 'node:path'
import { test, expect } from './fixtures'

const TRACE_PATH = path.join('..', 'tests', 'replay', 'corpus', 'view_run_logs.zip')

test('user can render the SPA root', async ({ page, context, webE2eStack }) => {
  await context.tracing.start({ screenshots: true, snapshots: true, sources: true })

  await page.goto(webE2eStack.baseUrl)
  await expect(page.locator('html')).toBeVisible()

  await context.tracing.stop({ path: TRACE_PATH })
})
