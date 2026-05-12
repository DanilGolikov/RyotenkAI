#!/usr/bin/env node
/**
 * Zod codegen for the FE contract layer (Phase 3, Contract Testing Matrix).
 *
 * Reads ``web/src/api/openapi.json`` (regenerated from the FastAPI app by
 * ``scripts/sync_openapi.py``) and writes ``web/src/api/zod.ts`` using the
 * upstream ``openapi-zod-client`` CLI. The output is the schema-only file,
 * so consumers can ``import { ... } from '@/api/zod'`` and call ``parse``
 * / ``safeParse`` on response payloads.
 *
 * The script is idempotent: running it twice produces a byte-identical
 * file. CI's ``make verify-api-sync`` relies on that to detect drift.
 *
 * Why a small wrapper rather than ``npx openapi-zod-client`` directly?
 * - We need to inject a banner comment with the regen command so reviewers
 *   immediately know the file is auto-generated.
 * - We want to handle relative paths consistently regardless of where
 *   the user invokes the script from (repo root vs. ``web/``).
 */

import { execFileSync } from 'node:child_process'
import { readFileSync, writeFileSync, existsSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const WEB_DIR = resolve(__dirname, '..')
const INPUT = resolve(WEB_DIR, 'src/api/openapi.json')
const OUTPUT = resolve(WEB_DIR, 'src/api/zod.ts')

if (!existsSync(INPUT)) {
  console.error(`[gen-zod] missing ${INPUT}. Run \`make sync-openapi\` first.`)
  process.exit(1)
}

// Locate the CLI binary. ``node_modules/.bin/openapi-zod-client`` is the
// canonical install path; falling back to ``npx`` keeps this script usable
// before ``npm install`` has populated ``.bin``.
const CLI_BIN = resolve(WEB_DIR, 'node_modules/.bin/openapi-zod-client')
const useLocalBin = existsSync(CLI_BIN)

const args = [
  // The CLI takes the input as a positional argument; ``-i`` is unsupported.
  INPUT,
  '-o',
  OUTPUT,
  // Emit all ``#/components/schemas`` so callers can import them by name
  // (e.g. ``import { ProjectSummary } from '@/api/zod'``). The endpoint
  // ``api`` const is also generated but we don't use it directly — the
  // existing ``client.ts`` fetch wrapper stays in charge.
  '--export-schemas',
]

const cmd = useLocalBin ? CLI_BIN : 'npx'
const cmdArgs = useLocalBin ? args : ['--yes', 'openapi-zod-client@1.18.3', ...args]

try {
  execFileSync(cmd, cmdArgs, { stdio: 'inherit', cwd: WEB_DIR })
} catch (err) {
  console.error('[gen-zod] codegen failed:', err.message)
  process.exit(1)
}

// Prepend an unmistakable auto-generated banner so reviewers don't try to
// hand-edit the file. The banner is appended *after* the codegen so the
// CLI's own header (if any) is preserved.
const BANNER = `/* eslint-disable */
// AUTO-GENERATED — DO NOT EDIT BY HAND.
// Regenerate with: \`make regen-zod\` (or \`cd web && npm run gen:zod\`).
// Source: web/src/api/openapi.json (produced by scripts/sync_openapi.py).
`

const existing = readFileSync(OUTPUT, 'utf8')
if (!existing.startsWith('/* eslint-disable */\n// AUTO-GENERATED')) {
  writeFileSync(OUTPUT, BANNER + existing, 'utf8')
}

console.log(`[gen-zod] wrote ${OUTPUT}`)
