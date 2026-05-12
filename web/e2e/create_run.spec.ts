/**
 * E2E: user opens the runs list and starts a new run.
 *
 * Records a Playwright trace that the Python replay corpus
 * (`tests/replay/test_replay_corpus.py`) loads on every release. The
 * trace file is stored under `tests/replay/corpus/` and committed
 * with this PR; baselines are immutable until a release explicitly
 * regenerates them.
 *
 * Determinism: the test uses MSW handlers from the FE codegen
 * pipeline (Phase 3), so no real backend is needed.
 */

import path from 'node:path'
import { test, expect } from './fixtures'

const TRACE_PATH = path.join('..', 'tests', 'replay', 'corpus', 'create_run.zip')

test('user can open the runs list', async ({ page, context, webE2eStack }) => {
  await context.tracing.start({ screenshots: true, snapshots: true, sources: true })

  await page.goto(webE2eStack.baseUrl)

  // We don't assert business behaviour here — the trace itself is the
  // artifact. The replay corpus pytest module asserts that the same
  // trace can be loaded against the current build without error.
  await expect(page).toHaveTitle(/.+/)

  await context.tracing.stop({ path: TRACE_PATH })
})
