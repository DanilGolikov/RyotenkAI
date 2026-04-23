import { useQuery } from '@tanstack/react-query'
import { api } from '../client'
import { qk } from '../queryKeys'
import type { PresetPreviewResponse } from '../types'

/** Tiny stable-ish hash of the current config so the query key invalidates
 *  when the user edits their config before opening the preview.
 *  Not a cryptographic hash — a string-length-insensitive FNV-like roll
 *  would be overkill; ``JSON.stringify`` is good enough at UI scale. */
function hashConfig(current: Record<string, unknown>): string {
  try {
    return JSON.stringify(current).length.toString(36)
  } catch {
    return 'na'
  }
}

/** Dry-run the backend's preset-apply logic against the user's current
 *  config. The backend returns a structured breakdown:
 *
 *  - ``diff``: per-key changes with ``reason`` = preset_replaced /
 *    preset_added / preset_preserved / no_scope
 *  - ``requirements``: readiness checks (hub_models available, provider
 *    matches, plugins installed, VRAM hint) with ok/missing/warning
 *  - ``placeholders``: paths the user still needs to fill after Apply
 *  - ``warnings``: e.g. legacy v1 "full overwrite" fallback banner
 *
 *  ``enabled`` lets us only fetch when the modal is actually open, so
 *  closing the modal doesn't leave a pending query hanging.
 */
export function usePresetPreview(
  presetId: string | null,
  current: Record<string, unknown>,
) {
  return useQuery({
    queryKey: presetId
      ? qk.configPresetPreview(presetId, hashConfig(current))
      : ['config', 'presets', 'preview', 'idle'],
    queryFn: () =>
      api.post<PresetPreviewResponse>(`/config/presets/${presetId}/preview`, {
        current_config: current,
      }),
    enabled: !!presetId,
    // Preview is a pure function of (preset, current) — cache forever for
    // this combination; React Query will garbage-collect when the modal
    // unmounts and the cursor moves off the key.
    staleTime: Infinity,
  })
}
