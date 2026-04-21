import type { Recommendation } from '../../lib/loraRecommendations'

interface Props<V extends object> {
  /** Title shown above the chips. */
  title?: string
  /** Current value of the object being edited. Used to highlight
   *  the active preset (all values in the preset match the current
   *  object — extra fields the user set manually don't disqualify it). */
  currentValue: Record<string, unknown>
  /** Available presets. */
  recommendations: Recommendation<V>[]
  /** Called with the full object the user is editing merged with the
   *  preset values. The parent decides whether to replace (write full
   *  object) or merge. */
  onApply: (next: Record<string, unknown>) => void
}

function shallowSubsetEqual(
  subset: Record<string, unknown>,
  of: Record<string, unknown>,
): boolean {
  for (const [k, v] of Object.entries(subset)) {
    const cur = of[k]
    if (Array.isArray(v) && Array.isArray(cur)) {
      if (v.length !== cur.length) return false
      for (let i = 0; i < v.length; i += 1) if (v[i] !== cur[i]) return false
      continue
    }
    if (cur !== v) return false
  }
  return true
}

/**
 * Generic one-click preset picker. Renders as a wrapped row of
 * "chip" buttons above a form section; clicking a chip writes the
 * preset's fields into the parent value (merged, so fields the preset
 * doesn't touch stay intact).
 *
 * Used by LoRA/QLoRA, GlobalHyperparameters, and per-strategy-phase
 * hyperparameter sections.
 */
export function RecommendationChips<V extends object>({
  title = 'Start from a tested baseline',
  currentValue,
  recommendations,
  onApply,
}: Props<V>) {
  if (recommendations.length === 0) return null

  return (
    <div className="rounded-md border border-line-1 bg-surface-0/50 px-3 py-2 space-y-1.5">
      <div className="text-[0.6rem] uppercase tracking-wide text-ink-3">{title}</div>
      <div className="flex flex-wrap gap-1.5">
        {recommendations.map((rec) => {
          const active = shallowSubsetEqual(
            rec.values as Record<string, unknown>,
            currentValue,
          )
          return (
            <button
              key={rec.id}
              type="button"
              onClick={() =>
                onApply({ ...currentValue, ...(rec.values as Record<string, unknown>) })
              }
              title={rec.description}
              className={[
                'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[0.65rem] transition',
                active
                  ? 'border-brand bg-brand-weak/50 text-brand-strong'
                  : 'border-line-2 text-ink-2 hover:text-ink-1 hover:border-brand-alt',
              ].join(' ')}
            >
              <span className="font-sans font-medium">{rec.label}</span>
              {rec.summary && (
                <span aria-hidden className="opacity-70 font-mono">
                  {rec.summary}
                </span>
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}
