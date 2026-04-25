/**
 * Vitest test-environment setup. Loaded once per worker before the
 * first test file runs.
 *
 * Side effects:
 * - Extends ``expect`` with the @testing-library/jest-dom matchers
 *   (``toBeInTheDocument``, ``toHaveTextContent``, ``toBeDisabled``,
 *   …) so component tests read like the React docs.
 * - Stubs ``window.matchMedia`` because some headless UI widgets
 *   (CodeMirror's theme detection, react-router devtools) call it on
 *   mount; jsdom doesn't ship one.
 */

import '@testing-library/jest-dom/vitest'

// Stub matchMedia for components that probe prefers-color-scheme on
// mount. Returning a deterministic "no" prevents flaky tests where
// the dark-mode branch only fires sometimes.
if (typeof window !== 'undefined' && !window.matchMedia) {
  window.matchMedia = (query: string): MediaQueryList => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  })
}
