/**
 * axe-core accessibility scan on the key pages.
 *
 * We assert *no critical or serious* violations. "Moderate" and
 * "minor" violations are tracked but not failed — phase-6 baseline,
 * tightened later (Decision 7 schedule).
 */

import AxeBuilder from '@axe-core/playwright'
import { test, expect } from './fixtures'

const PAGES = [
  { path: '/', label: 'root' },
  { path: '/runs', label: 'runs' },
  { path: '/projects', label: 'projects' },
] as const

for (const page of PAGES) {
  test(`axe: ${page.label} has no critical/serious violations`, async ({
    page: playwright,
    webE2eStack,
  }) => {
    await playwright.goto(`${webE2eStack.baseUrl}${page.path}`)
    const results = await new AxeBuilder({ page: playwright })
      .withTags(['wcag2a', 'wcag2aa'])
      .analyze()

    const blocking = results.violations.filter((v) =>
      ['critical', 'serious'].includes(v.impact ?? ''),
    )

    if (blocking.length > 0) {
      // Pretty-print so the failure message is actionable.
      const summary = blocking
        .map(
          (v) =>
            `  - [${v.impact}] ${v.id}: ${v.help} (${v.nodes.length} nodes)`,
        )
        .join('\n')
      throw new Error(
        `Found ${blocking.length} critical/serious axe violation(s) on ${page.path}:\n${summary}`,
      )
    }

    expect(blocking).toEqual([])
  })
}
