import type { ConfigCheck } from '../../api/types'

/** Best-effort mapping of a validation check to a top-level PipelineConfig group. */
export function groupForCheck(check: ConfigCheck): string | null {
  const haystack = `${check.label} ${check.detail ?? ''}`.toLowerCase()
  if (haystack.includes('dataset')) return 'datasets'
  if (haystack.includes('runpod') || haystack.includes('provider')) return 'providers'
  if (haystack.includes('hf_token') || haystack.includes('huggingface') || haystack.includes('mlflow')) {
    return 'experiment_tracking'
  }
  if (haystack.includes('eval')) return 'evaluation'
  if (haystack.includes('inference')) return 'inference'
  if (haystack.includes('training') || haystack.includes('optimizer') || haystack.includes('strategy')) {
    return 'training'
  }
  if (haystack.includes('model')) return 'model'
  return null
}

export type GroupValidity = 'ok' | 'warn' | 'err' | 'idle'

export function deriveGroupValidity(checks: ConfigCheck[]): Record<string, GroupValidity> {
  const out: Record<string, GroupValidity> = {}
  for (const check of checks) {
    const group = groupForCheck(check)
    if (!group) continue
    const current = out[group] ?? 'idle'
    const next: GroupValidity =
      check.status === 'fail'
        ? 'err'
        : check.status === 'warn'
        ? current === 'err'
          ? 'err'
          : 'warn'
        : current === 'idle'
        ? 'ok'
        : current
    out[group] = next
  }
  return out
}
