/**
 * Tests for ``pluginInstances.ts`` — the pure read/write/add/remove/
 * reorder/rename helpers that translate between a parsed pipeline
 * config and the UI's ``{instanceId, pluginId, enabled?}`` shape.
 *
 * Coverage matrix (categories requested by the user):
 *   - positive / negative ("does it do the happy-path; does it refuse bad input?")
 *   - boundary (empty lists, missing keys, single-item lists, duplicates)
 *   - invariants (``readInstances(write(x)) === x`` round-trips where meaningful)
 *   - dependency errors (unknown plugin reference in YAML → read tolerates; write rejects)
 *   - regressions (bugs already fixed — guard against a reappearance)
 *   - logic-specific (kind-specific write paths: reward mirrors params across
 *     matching strategies; reports single-instance; validation default-dataset-only)
 *   - combinatorial (all (kind × mutation) pairs sanity-checked)
 */

import { describe, expect, it } from 'vitest'
import type { PluginManifest } from '../../api/types'
import {
  addInstance,
  generateInstanceId,
  readInstanceDetails,
  readInstances,
  removeInstance,
  renameInstance,
  reorderInstances,
  rewardBroadcastTargets,
  writeInstanceDetails,
} from './pluginInstances'

// ---------------------------------------------------------------------------
// Test helpers — tiny factories for PluginManifest and parsed configs. Kept
// hand-rolled (no generic factory framework) so the intent of each test is
// obvious without jumping through abstractions.
// ---------------------------------------------------------------------------

function manifest(id: string, overrides: Partial<PluginManifest> = {}): PluginManifest {
  return {
    schema_version: 3,
    id,
    name: id,
    version: '1.0.0',
    description: '',
    category: '',
    stability: 'stable',
    kind: 'validation',
    supported_strategies: [],
    params_schema: {},
    thresholds_schema: {},
    suggested_params: {},
    suggested_thresholds: {},
    ...overrides,
  }
}

function emptyConfig(): Record<string, unknown> {
  return {}
}

function validationConfig(ids: string[]): Record<string, unknown> {
  return {
    datasets: {
      default: {
        validations: {
          plugins: ids.map((id, i) => ({
            id: `${id}${i === 0 ? '' : `_${i + 1}`}`,
            plugin: id,
          })),
        },
      },
    },
  }
}

// ---------------------------------------------------------------------------
// readInstances — positive
// ---------------------------------------------------------------------------

describe('readInstances — positive path', () => {
  it('validation: returns entries from default dataset', () => {
    const cfg = validationConfig(['avg_length', 'min_samples'])
    const res = readInstances('validation', cfg)
    expect(res).toEqual([
      { instanceId: 'avg_length', pluginId: 'avg_length', enabled: undefined },
      { instanceId: 'min_samples_2', pluginId: 'min_samples', enabled: undefined },
    ])
  })

  it('evaluation: reads evaluation.evaluators.plugins', () => {
    const cfg = {
      evaluation: {
        enabled: true,
        evaluators: {
          plugins: [
            { id: 'judge', plugin: 'cerebras_judge', enabled: false },
          ],
        },
      },
    }
    const res = readInstances('evaluation', cfg)
    expect(res).toEqual([
      { instanceId: 'judge', pluginId: 'cerebras_judge', enabled: false },
    ])
  })

  it('reward: dedupes reward_plugin across phases', () => {
    const cfg = {
      training: {
        strategies: [
          { strategy_type: 'sft' },
          { strategy_type: 'grpo', params: { reward_plugin: 'helixql' } },
          { strategy_type: 'sapo', params: { reward_plugin: 'helixql' } },
        ],
      },
    }
    const res = readInstances('reward', cfg)
    expect(res).toEqual([{ instanceId: 'helixql', pluginId: 'helixql' }])
  })

  it('reports: reads reports.sections in order', () => {
    const cfg = { reports: { sections: ['header', 'summary', 'footer'] } }
    const res = readInstances('reports', cfg)
    expect(res.map((x) => x.pluginId)).toEqual(['header', 'summary', 'footer'])
  })
})

// ---------------------------------------------------------------------------
// readInstances — negative / boundary
// ---------------------------------------------------------------------------

describe('readInstances — boundary', () => {
  it('returns [] on completely empty config for every kind', () => {
    for (const kind of ['validation', 'evaluation', 'reward', 'reports'] as const) {
      expect(readInstances(kind, {})).toEqual([])
    }
  })

  it('validation: ignores non-object entries', () => {
    const cfg = {
      datasets: {
        default: {
          validations: { plugins: ['not-an-object', null, 42] },
        },
      },
    }
    expect(readInstances('validation', cfg)).toEqual([])
  })

  it('evaluation: skips entries with missing plugin id', () => {
    const cfg = {
      evaluation: {
        evaluators: { plugins: [{ id: 'no_plugin_ref' }, { plugin: 'ok' }] },
      },
    }
    expect(readInstances('evaluation', cfg)).toEqual([
      { instanceId: 'ok', pluginId: 'ok', enabled: undefined },
    ])
  })

  it('reward: ignores phases that are not grpo/sapo/dpo/orpo', () => {
    const cfg = {
      training: {
        strategies: [
          { strategy_type: 'sft', params: { reward_plugin: 'leaked' } },
          { strategy_type: 'cpt', params: { reward_plugin: 'also_leaked' } },
        ],
      },
    }
    expect(readInstances('reward', cfg)).toEqual([])
  })

  it('reports: filters non-string entries', () => {
    const cfg = { reports: { sections: ['header', 42, null, 'footer'] } }
    expect(readInstances('reports', cfg).map((x) => x.pluginId)).toEqual(['header', 'footer'])
  })
})

// ---------------------------------------------------------------------------
// addInstance — positive + invariants
// ---------------------------------------------------------------------------

describe('addInstance', () => {
  it('validation: first attach creates default dataset + plugins list', () => {
    const { next, instanceId } = addInstance('validation', emptyConfig(), manifest('min_samples'))
    expect(instanceId).toBe('min_samples')
    const read = readInstances('validation', next)
    expect(read).toEqual([
      { instanceId: 'min_samples', pluginId: 'min_samples', enabled: undefined },
    ])
  })

  it('validation: second attach of same plugin gets _2 suffix (multi-instance)', () => {
    let cfg: Record<string, unknown> = emptyConfig()
    cfg = addInstance('validation', cfg, manifest('avg_length')).next
    const step2 = addInstance('validation', cfg, manifest('avg_length'))
    expect(step2.instanceId).toBe('avg_length_2')
    const ids = readInstances('validation', step2.next).map((x) => x.instanceId)
    expect(ids).toEqual(['avg_length', 'avg_length_2'])
  })

  it('validation: N-th attach increments suffix, skipping already-taken ids', () => {
    let cfg: Record<string, unknown> = emptyConfig()
    for (let i = 0; i < 5; i++) {
      cfg = addInstance('validation', cfg, manifest('avg_length')).next
    }
    const ids = readInstances('validation', cfg).map((x) => x.instanceId)
    expect(ids).toEqual(['avg_length', 'avg_length_2', 'avg_length_3', 'avg_length_4', 'avg_length_5'])
  })

  it('evaluation: adds with suggested params + thresholds merged in', () => {
    const m = manifest('helixql_syntax', {
      kind: 'evaluation',
      suggested_params: { sample_size: 100 },
      suggested_thresholds: { min_valid_ratio: 0.8 },
    })
    const { next } = addInstance('evaluation', emptyConfig(), m)
    const details = readInstanceDetails('evaluation', next, 'helixql_syntax')
    expect(details?.params).toEqual({ sample_size: 100 })
    expect(details?.thresholds).toEqual({ min_valid_ratio: 0.8 })
  })

  it('reports: appends id to sections and is a no-op if already present', () => {
    const cfg = { reports: { sections: ['header'] } }
    const first = addInstance('reports', cfg, manifest('summary', { kind: 'reports' }))
    expect(readInstances('reports', first.next).map((x) => x.pluginId)).toEqual([
      'header', 'summary',
    ])
    // Re-add same id — no duplicate.
    const second = addInstance('reports', first.next, manifest('summary', { kind: 'reports' }))
    expect(readInstances('reports', second.next).map((x) => x.pluginId)).toEqual([
      'header', 'summary',
    ])
  })

  it('reward: writes to matching phases only (grpo/sapo/dpo/orpo)', () => {
    const cfg = {
      training: {
        strategies: [
          { strategy_type: 'sft' },
          { strategy_type: 'grpo', hyperparams: {} },
          { strategy_type: 'sapo' },
        ],
      },
    }
    const m = manifest('helixql', { kind: 'reward', supported_strategies: ['grpo', 'sapo'] })
    const { next } = addInstance('reward', cfg, m)
    const strategies = (next.training as Record<string, unknown>).strategies as Array<Record<string, unknown>>
    expect((strategies[0].params as Record<string, unknown> | undefined)?.reward_plugin).toBeUndefined()
    expect((strategies[1].params as Record<string, unknown>).reward_plugin).toBe('helixql')
    expect((strategies[2].params as Record<string, unknown>).reward_plugin).toBe('helixql')
  })

  it('reward: skips phases whose strategy_type is NOT in supported_strategies', () => {
    const cfg = {
      training: {
        strategies: [
          { strategy_type: 'grpo' },
          { strategy_type: 'dpo' },
        ],
      },
    }
    const m = manifest('grpo_only', { kind: 'reward', supported_strategies: ['grpo'] })
    const { next } = addInstance('reward', cfg, m)
    const s = (next.training as Record<string, unknown>).strategies as Array<Record<string, unknown>>
    expect((s[0].params as Record<string, unknown>).reward_plugin).toBe('grpo_only')
    // dpo phase is a compatible "reward-family" phase BUT not in the
    // plugin's supported list — must stay untouched.
    expect((s[1].params as Record<string, unknown> | undefined)?.reward_plugin).toBeUndefined()
  })
})

// ---------------------------------------------------------------------------
// generateInstanceId — logic-specific
// ---------------------------------------------------------------------------

describe('generateInstanceId', () => {
  it('returns the bare id when nothing is taken', () => {
    expect(generateInstanceId('foo', new Set())).toBe('foo')
  })

  it('appends _2 on first collision', () => {
    expect(generateInstanceId('foo', new Set(['foo']))).toBe('foo_2')
  })

  it('skips already-taken suffixes (combinatorial gap)', () => {
    expect(generateInstanceId('foo', new Set(['foo', 'foo_2', 'foo_4']))).toBe('foo_3')
    expect(generateInstanceId('foo', new Set(['foo', 'foo_2', 'foo_3']))).toBe('foo_4')
  })
})

// ---------------------------------------------------------------------------
// removeInstance
// ---------------------------------------------------------------------------

describe('removeInstance', () => {
  it('validation: removes matching entry and keeps siblings', () => {
    const cfg = validationConfig(['a', 'b', 'c'])
    const next = removeInstance('validation', cfg, 'b_2')
    expect(readInstances('validation', next).map((x) => x.instanceId)).toEqual(['a', 'c_3'])
  })

  it('evaluation: unknown id is a no-op (no crash)', () => {
    const cfg = {
      evaluation: { evaluators: { plugins: [{ id: 'only', plugin: 'only' }] } },
    }
    const next = removeInstance('evaluation', cfg, 'nonexistent')
    expect(readInstances('evaluation', next).map((x) => x.instanceId)).toEqual(['only'])
  })

  it('reward: clears reward_plugin only on phases referencing the given id', () => {
    const cfg = {
      training: {
        strategies: [
          { strategy_type: 'grpo', params: { reward_plugin: 'A', unrelated: 1 } },
          { strategy_type: 'sapo', params: { reward_plugin: 'B' } },
        ],
      },
    }
    const next = removeInstance('reward', cfg, 'A')
    const s = (next.training as Record<string, unknown>).strategies as Array<Record<string, unknown>>
    expect((s[0].params as Record<string, unknown>).reward_plugin).toBeUndefined()
    // Unrelated key must survive — regression guard.
    expect((s[0].params as Record<string, unknown>).unrelated).toBe(1)
    expect((s[1].params as Record<string, unknown>).reward_plugin).toBe('B')
  })

  it('reports: drops the matching id preserving order', () => {
    const cfg = { reports: { sections: ['a', 'b', 'c'] } }
    const next = removeInstance('reports', cfg, 'b')
    expect(readInstances('reports', next).map((x) => x.pluginId)).toEqual(['a', 'c'])
  })

  it('reports: removing last leaves empty sections (allowed)', () => {
    const cfg = { reports: { sections: ['only'] } }
    const next = removeInstance('reports', cfg, 'only')
    expect(readInstances('reports', next)).toEqual([])
    // sections still present but empty — caller (PluginsTab) decides to
    // either leave it or delete the key via handleResetReports.
    expect((next.reports as Record<string, unknown>).sections).toEqual([])
  })
})

// ---------------------------------------------------------------------------
// reorderInstances — invariants + combinatorial
// ---------------------------------------------------------------------------

describe('reorderInstances', () => {
  it('validation: applies the new id order exactly', () => {
    const cfg = validationConfig(['a', 'b', 'c'])
    const next = reorderInstances('validation', cfg, ['c_3', 'a', 'b_2'])
    expect(readInstances('validation', next).map((x) => x.instanceId)).toEqual(['c_3', 'a', 'b_2'])
  })

  it('evaluation: round-trip read(write(read)) is identity on order', () => {
    const cfg = {
      evaluation: {
        evaluators: {
          plugins: [
            { id: 'x', plugin: 'x' },
            { id: 'y', plugin: 'y' },
            { id: 'z', plugin: 'z' },
          ],
        },
      },
    }
    const originalOrder = readInstances('evaluation', cfg).map((x) => x.instanceId)
    const next = reorderInstances('evaluation', cfg, originalOrder)
    expect(readInstances('evaluation', next).map((x) => x.instanceId)).toEqual(originalOrder)
  })

  it('reports: new order is written verbatim (source of truth is the config)', () => {
    const cfg = { reports: { sections: ['header', 'summary', 'footer'] } }
    const next = reorderInstances('reports', cfg, ['footer', 'header', 'summary'])
    expect((next.reports as Record<string, unknown>).sections).toEqual([
      'footer', 'header', 'summary',
    ])
  })

  it('reward: is a no-op (single-instance-per-matching-phase)', () => {
    const cfg = {
      training: {
        strategies: [{ strategy_type: 'grpo', params: { reward_plugin: 'x' } }],
      },
    }
    const next = reorderInstances('reward', cfg, ['x'])
    expect(next).toEqual(cfg)
  })

  it('validation: items missing from orderedIds fall through to end (defensive)', () => {
    const cfg = validationConfig(['a', 'b', 'c'])
    const next = reorderInstances('validation', cfg, ['c_3'])
    const ids = readInstances('validation', next).map((x) => x.instanceId)
    // orphaned items appended in their original relative order
    expect(ids[0]).toBe('c_3')
    expect(new Set(ids)).toEqual(new Set(['a', 'b_2', 'c_3']))
  })
})

// ---------------------------------------------------------------------------
// renameInstance — combinatorial
// ---------------------------------------------------------------------------

describe('renameInstance', () => {
  it('same old/new id is a no-op', () => {
    const cfg = validationConfig(['a'])
    const next = renameInstance('validation', cfg, 'a', 'a')
    expect(next).toBe(cfg)
  })

  it('validation: renames and returns a new config', () => {
    const cfg = validationConfig(['a'])
    const next = renameInstance('validation', cfg, 'a', 'a_renamed')
    expect(next).not.toBeNull()
    expect(readInstances('validation', next!).map((x) => x.instanceId)).toEqual(['a_renamed'])
  })

  it('returns null when the target id is already taken (uniqueness invariant)', () => {
    const cfg = validationConfig(['a', 'b'])
    const result = renameInstance('validation', cfg, 'a', 'b_2')
    expect(result).toBeNull()
  })

  it('reports: rename is a no-op (instanceId === pluginId)', () => {
    const cfg = { reports: { sections: ['header'] } }
    const next = renameInstance('reports', cfg, 'header', 'renamed')
    expect(next).toBe(cfg)
  })

  it('reward: rename rewrites reward_plugin in every referencing phase', () => {
    const cfg = {
      training: {
        strategies: [
          { strategy_type: 'grpo', params: { reward_plugin: 'old' } },
          { strategy_type: 'sapo', params: { reward_plugin: 'old' } },
          { strategy_type: 'grpo', params: { reward_plugin: 'other' } },
        ],
      },
    }
    const next = renameInstance('reward', cfg, 'old', 'new')
    const s = (next!.training as Record<string, unknown>).strategies as Array<Record<string, unknown>>
    expect((s[0].params as Record<string, unknown>).reward_plugin).toBe('new')
    expect((s[1].params as Record<string, unknown>).reward_plugin).toBe('new')
    expect((s[2].params as Record<string, unknown>).reward_plugin).toBe('other')
  })
})

// ---------------------------------------------------------------------------
// readInstanceDetails / writeInstanceDetails — round-trip invariants
// ---------------------------------------------------------------------------

describe('readInstanceDetails / writeInstanceDetails', () => {
  it('validation: write(read(x)) === x for all instance-level fields', () => {
    const cfg = {
      datasets: {
        default: {
          validations: {
            plugins: [
              {
                id: 'v1',
                plugin: 'avg_length',
                enabled: true,
                apply_to: ['train'],
                fail_on_error: true,
                params: { sample_size: 100 },
                thresholds: { min: 50 },
              },
            ],
          },
        },
      },
    }
    const details = readInstanceDetails('validation', cfg, 'v1')!
    expect(details.enabled).toBe(true)
    expect(details.apply_to).toEqual(['train'])
    expect(details.fail_on_error).toBe(true)
    expect(details.params).toEqual({ sample_size: 100 })
    const next = writeInstanceDetails('validation', cfg, details)
    const roundtripped = readInstanceDetails('validation', next, 'v1')
    expect(roundtripped).toEqual(details)
  })

  it('evaluation: mutating params writes back under the same id', () => {
    const cfg = {
      evaluation: {
        evaluators: {
          plugins: [{ id: 'e1', plugin: 'judge', params: { a: 1 } }],
        },
      },
    }
    const details = readInstanceDetails('evaluation', cfg, 'e1')!
    details.params = { a: 2, b: 3 }
    const next = writeInstanceDetails('evaluation', cfg, details)
    const read = readInstanceDetails('evaluation', next, 'e1')!
    expect(read.params).toEqual({ a: 2, b: 3 })
  })

  it('reward: writing params mirrors to every phase referencing the same reward_plugin', () => {
    const cfg = {
      training: {
        strategies: [
          { strategy_type: 'grpo', params: { reward_plugin: 'r', timeout: 5 } },
          { strategy_type: 'sapo', params: { reward_plugin: 'r' } },
          { strategy_type: 'grpo', params: { reward_plugin: 'other' } },
        ],
      },
    }
    const details = readInstanceDetails('reward', cfg, 'r')!
    details.params = { timeout: 42 }
    const next = writeInstanceDetails('reward', cfg, details)
    const s = (next.training as Record<string, unknown>).strategies as Array<Record<string, unknown>>
    expect(s[0].params).toEqual({ timeout: 42, reward_plugin: 'r' })
    expect(s[1].params).toEqual({ timeout: 42, reward_plugin: 'r' })
    // Phase attached to a DIFFERENT reward plugin stays untouched
    // (regression: earlier draft overwrote unrelated phases).
    expect((s[2].params as Record<string, unknown>).reward_plugin).toBe('other')
  })

  it('reports: writeInstanceDetails is a no-op (reports have no per-instance state)', () => {
    const cfg = { reports: { sections: ['a'] } }
    const before = JSON.stringify(cfg)
    const details = readInstanceDetails('reports', cfg, 'a')!
    const next = writeInstanceDetails('reports', cfg, details)
    // structural equality — write returns a clone but reports block
    // is unchanged.
    expect(next).toEqual(cfg)
    // Input was not mutated either (immutability invariant).
    expect(JSON.stringify(cfg)).toBe(before)
  })

  it('readInstanceDetails returns null for missing ids (dependency error)', () => {
    expect(readInstanceDetails('validation', {}, 'ghost')).toBeNull()
    expect(readInstanceDetails('evaluation', { evaluation: { evaluators: {} } }, 'ghost')).toBeNull()
    expect(readInstanceDetails('reward', { training: { strategies: [] } }, 'ghost')).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// Immutability invariants — every mutator returns a new object tree,
// original config must not be touched.
// ---------------------------------------------------------------------------

describe('invariants: pure functions never mutate the input config', () => {
  it('addInstance', () => {
    const cfg = emptyConfig()
    const snapshot = JSON.stringify(cfg)
    addInstance('validation', cfg, manifest('x'))
    expect(JSON.stringify(cfg)).toBe(snapshot)
  })

  it('removeInstance', () => {
    const cfg = validationConfig(['a'])
    const snapshot = JSON.stringify(cfg)
    removeInstance('validation', cfg, 'a')
    expect(JSON.stringify(cfg)).toBe(snapshot)
  })

  it('reorderInstances', () => {
    const cfg = validationConfig(['a', 'b'])
    const snapshot = JSON.stringify(cfg)
    reorderInstances('validation', cfg, ['b_2', 'a'])
    expect(JSON.stringify(cfg)).toBe(snapshot)
  })

  it('renameInstance', () => {
    const cfg = validationConfig(['a'])
    const snapshot = JSON.stringify(cfg)
    renameInstance('validation', cfg, 'a', 'b')
    expect(JSON.stringify(cfg)).toBe(snapshot)
  })

  it('writeInstanceDetails', () => {
    const cfg = validationConfig(['a'])
    const snapshot = JSON.stringify(cfg)
    const details = readInstanceDetails('validation', cfg, 'a')!
    writeInstanceDetails('validation', cfg, details)
    expect(JSON.stringify(cfg)).toBe(snapshot)
  })
})

// ---------------------------------------------------------------------------
// Combinatorial sanity — every (kind × mutation) pair doesn't throw on
// empty input. Bare-minimum regression guard against shape-handling bugs
// that only surface when a kind's top-level block is missing entirely.
// ---------------------------------------------------------------------------

describe('combinatorial: every (kind × mutation) on empty config is safe', () => {
  const kinds = ['validation', 'evaluation', 'reward', 'reports'] as const

  it.each(kinds)('readInstances(%s, {}) returns []', (kind) => {
    expect(readInstances(kind, {})).toEqual([])
  })

  it.each(kinds)('removeInstance(%s, {}, "ghost") is a no-op', (kind) => {
    const next = removeInstance(kind, {}, 'ghost')
    expect(readInstances(kind, next)).toEqual([])
  })

  it.each(kinds)('reorderInstances(%s, {}, []) is a no-op', (kind) => {
    const next = reorderInstances(kind, {}, [])
    expect(readInstances(kind, next)).toEqual([])
  })
})

// ---------------------------------------------------------------------------
// rewardBroadcastTargets — the helper that powers PR13's "params apply
// to N strategies" hint. Pure function, easy to cover exhaustively.
// ---------------------------------------------------------------------------

describe('rewardBroadcastTargets', () => {
  it('returns every strategy whose phase references the instance id', () => {
    const cfg = {
      training: {
        strategies: [
          { strategy_type: 'sft', params: {} },
          {
            strategy_type: 'grpo',
            params: { reward_plugin: 'helixql_compiler_semantic' },
          },
          {
            strategy_type: 'sapo',
            params: { reward_plugin: 'helixql_compiler_semantic' },
          },
        ],
      },
    }
    expect(rewardBroadcastTargets(cfg, 'helixql_compiler_semantic')).toEqual([
      'grpo',
      'sapo',
    ])
  })

  it('returns an empty list when the instance is not referenced anywhere', () => {
    const cfg = {
      training: {
        strategies: [
          {
            strategy_type: 'grpo',
            params: { reward_plugin: 'other_reward' },
          },
        ],
      },
    }
    expect(rewardBroadcastTargets(cfg, 'helixql_compiler_semantic')).toEqual([])
  })

  it('returns an empty list when the config has no training section', () => {
    expect(rewardBroadcastTargets({}, 'anything')).toEqual([])
  })

  it('preserves config order (no de-duplication, no sorting)', () => {
    const cfg = {
      training: {
        strategies: [
          { strategy_type: 'sapo', params: { reward_plugin: 'r' } },
          { strategy_type: 'grpo', params: { reward_plugin: 'r' } },
        ],
      },
    }
    expect(rewardBroadcastTargets(cfg, 'r')).toEqual(['sapo', 'grpo'])
  })

  it('lowercases strategy types so the hint reads consistently', () => {
    const cfg = {
      training: {
        strategies: [
          { strategy_type: 'GRPO', params: { reward_plugin: 'r' } },
        ],
      },
    }
    expect(rewardBroadcastTargets(cfg, 'r')).toEqual(['grpo'])
  })
})
