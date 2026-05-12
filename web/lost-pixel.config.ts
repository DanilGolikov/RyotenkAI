/**
 * lost-pixel configuration — visual regression for Storybook stories.
 *
 * Phase 6 status: **non-blocking warning**. Per Decision 7 in the
 * structured-hopping-starfish plan, visual diffs are tracked as PR
 * comments for the first 6 months while baselines stabilise. Promotion
 * to blocking happens once we have a stable 6-month rolling window
 * with <1% false-positive rate.
 *
 *   * shotConcurrency: 2 — CI runners are small, two parallel headless
 *     Chromium tabs keeps memory usage <1.5 GB.
 *   * threshold: 0.01 — 1% pixel-diff tolerance. Conservative. Antialias
 *     and subpixel rendering drift between Linux and macOS sits around
 *     0.3%, so 1% leaves headroom without masking real diffs.
 *   * mask: timestamp regions inside RunRow and dynamic IDs in
 *     DeleteProjectModal. Volatile regions are marked with the
 *     `[data-lostpixel-mask]` attribute on the source element; we list
 *     the same selector here so the diff engine ignores them.
 */

import type { CustomProjectConfig } from 'lost-pixel'

export const config: CustomProjectConfig = {
  storybookShots: {
    storybookUrl: 'http://localhost:6006',
    mask: [
      // Volatile regions: anything tagged with this attribute is
      // ignored on every story. Components that render timestamps,
      // dynamic ids, or stochastic placeholders should add the
      // attribute on the relevant DOM node.
      { selector: '[data-lostpixel-mask="true"]' },
      // RunRow's relative-time line — drifts every minute.
      { selector: '.run-row-time' },
    ],
  },
  shotConcurrency: 2,
  imagePathBaseline: './.lost-pixel/baseline',
  imagePathCurrent: './.lost-pixel/current',
  imagePathDifference: './.lost-pixel/difference',
  threshold: 0.01,
  // Phase-6 attitude — never fail the build on visual diff. The
  // GitHub Action layer (`.github/workflows/visual-regression.yml`)
  // posts the diffs as a PR comment instead. Six months from now we
  // flip this back to default `error` exit.
  failOnDifference: false,
}

export default config
