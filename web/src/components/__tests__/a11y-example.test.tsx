/**
 * jest-axe component-level accessibility example (Phase 6).
 *
 * The end-to-end axe scan in `web/e2e/a11y.spec.ts` catches whole-page
 * violations against the real DOM; this jest-axe variant catches them
 * at the component level, where the iteration loop is faster.
 *
 * Phase-6 ships exactly one example to document the pattern. New
 * components SHOULD add an a11y test alongside the regular RTL tests,
 * with `expect(await axe(container)).toHaveNoViolations()` as the
 * usual assertion.
 *
 * The example below uses a tiny inline component so the test stays
 * decoupled from production-code import paths — the point is to
 * demonstrate the jest-axe pattern, not to assert against a specific
 * component.
 */

import { describe, expect, it } from 'vitest'
import { axe, toHaveNoViolations } from 'jest-axe'
import { render } from '@testing-library/react'

expect.extend(toHaveNoViolations)

function SamplePanel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section aria-labelledby="panel-heading">
      <h2 id="panel-heading">{title}</h2>
      <p>{children}</p>
      <button type="button" aria-label="acknowledge">
        OK
      </button>
    </section>
  )
}

describe('jest-axe a11y example', () => {
  it('a heading + paragraph + button passes WCAG2 AA', async () => {
    const { container } = render(
      <SamplePanel title="Run completed">
        The trainer exited successfully after 45 minutes.
      </SamplePanel>,
    )
    const results = await axe(container)
    expect(results).toHaveNoViolations()
  })

  it('a panel with only a labelled button still passes', async () => {
    const { container } = render(
      <button type="button" aria-label="dismiss notification">
        x
      </button>,
    )
    const results = await axe(container)
    expect(results).toHaveNoViolations()
  })
})
