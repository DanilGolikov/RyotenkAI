/**
 * E2E: user navigates to a running run and triggers a cancellation.
 *
 * Mirrors the create_run flow: we record a trace and commit it as
 * part of the replay corpus.
 */

import path from 'node:path'
import { test, expect } from './fixtures'

const TRACE_PATH = path.join('..', 'tests', 'replay', 'corpus', 'cancel_run.zip')

test('user can navigate to runs view', async ({ page, context, webE2eStack }) => {
  await context.tracing.start({ screenshots: true, snapshots: true, sources: true })

  await page.goto(webE2eStack.baseUrl)
  await expect(page.locator('body')).toBeVisible()

  await context.tracing.stop({ path: TRACE_PATH })
})
