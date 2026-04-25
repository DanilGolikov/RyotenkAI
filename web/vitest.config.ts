/**
 * Vitest configuration — separate from ``vite.config.ts`` so the dev
 * server config stays focused on serving the SPA.
 *
 * The default test environment is ``jsdom`` so component tests built
 * on ``@testing-library/react`` work without per-file overrides. Pure
 * logic tests (e.g. ``pluginInstances.test.ts``) don't touch the DOM
 * and run identically in jsdom — the slight startup overhead is in
 * the noise compared to running the app in CI.
 *
 * ``setupFiles`` extends ``expect`` with @testing-library/jest-dom's
 * matchers (``toBeInTheDocument`` etc.) so component tests read
 * naturally.
 */

import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'node:path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./vitest.setup.ts'],
    css: false,
  },
})
