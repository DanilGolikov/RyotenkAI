/**
 * Smoke test for the Vitest + jsdom + @testing-library/react harness.
 *
 * Guards against the kind of subtle config-drift that turns "I added
 * a new test file with `render()` and got SyntaxError on JSX" into a
 * 30-minute debugging session. By rendering a trivial component and
 * making one user-event interaction, we exercise:
 *
 * - JSX compilation in the test transform (Vite's React plugin),
 * - jsdom DOM availability (``screen.getByRole`` walks the DOM),
 * - the jest-dom matcher extension (``toBeInTheDocument``),
 * - and ``user-event``'s implicit pointer setup.
 *
 * If THIS file fails, every component test below it would also fail
 * for an unrelated reason — the smoke test is the canary.
 */

import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'

function Counter() {
  const [n, setN] = useState(0)
  return (
    <button onClick={() => setN((x) => x + 1)} type="button">
      clicked {n}
    </button>
  )
}

describe('component test harness', () => {
  it('renders a React component into jsdom', () => {
    render(<Counter />)
    expect(screen.getByRole('button')).toBeInTheDocument()
    expect(screen.getByRole('button')).toHaveTextContent('clicked 0')
  })

  it('handles user-event clicks with state updates', async () => {
    const user = userEvent.setup()
    render(<Counter />)
    await user.click(screen.getByRole('button'))
    await user.click(screen.getByRole('button'))
    expect(screen.getByRole('button')).toHaveTextContent('clicked 2')
  })
})
