import { useValidationCtx } from './ValidationContext'
import { SETTINGS_JUMP_TARGET, groupForCheck } from './validationMap'

interface Props {
  /** Click → navigate to Settings. Passed in by parent (ConfigTab owns
   *  routing). If omitted, the chip is non-interactive. */
  onOpenSettings?: () => void
}

/**
 * Small chip next to the provider dropdown that surfaces the current
 * secrets / provider health. It derives its state from the last
 * `/validate` response: any failure that `validationMap` routes to the
 * `__settings` group counts as a secret issue, and any failure routed
 * to `providers` counts as a provider-level problem.
 *
 * This avoids a dedicated `/secrets/health` endpoint: the backend
 * already surfaces missing env vars via validation checks, so we reuse
 * the same signal the banner reads. Trade-off: if validation hasn't
 * run yet (fresh page load, no auto-validate cycle), the chip shows
 * "unknown" rather than a false "ok".
 */
export function ProviderStatusChip({ onOpenSettings }: Props) {
  const ctx = useValidationCtx()
  const result = ctx?.validationResult ?? null

  if (!result) {
    return (
      <span
        className="pill pill-idle text-[0.65rem] leading-none"
        title="Run Validate to check provider secrets"
      >
        unknown
      </span>
    )
  }

  const failures = result.checks.filter((c) => c.status === 'fail')
  const secretFails = failures.filter((c) => groupForCheck(c) === SETTINGS_JUMP_TARGET)
  const providerFails = failures.filter((c) => groupForCheck(c) === 'providers')

  if (secretFails.length > 0) {
    const label =
      secretFails.length === 1
        ? '1 secret missing'
        : `${secretFails.length} secrets missing`
    return (
      <button
        type="button"
        onClick={onOpenSettings}
        className="pill pill-err text-[0.65rem] leading-none hover:brightness-110"
        title="Open Settings to set the required env vars"
      >
        ✗ {label}
      </button>
    )
  }
  if (providerFails.length > 0) {
    return (
      <span
        className="pill pill-warn text-[0.65rem] leading-none"
        title={providerFails[0]?.label ?? 'Provider has issues'}
      >
        provider warning
      </span>
    )
  }
  return (
    <span
      className="pill pill-ok text-[0.65rem] leading-none"
      title="All provider secrets set and validated"
    >
      ✓ key set
    </span>
  )
}
