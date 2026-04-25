/**
 * Banner that surfaces stale plugin references in the saved config.
 *
 * Stale = the YAML still mentions a plugin id, but the community
 * catalog no longer registers it (manifest deleted, plugin renamed,
 * folder moved). Backend (PR14 in cozy-booping-walrus) populates
 * ``ConfigResponse.stale_plugins`` on every GET; the UI renders this
 * banner alongside the Plugins / Datasets tabs so the user can purge
 * the dead reference with one click instead of failing the run
 * mid-pipeline with a "plugin not found" error.
 *
 * Per-row "Remove from config" calls back into the same
 * ``removeInstance`` helper the rest of the tab uses — one source of
 * truth for "make this reference disappear from YAML".
 */

import type { PluginKind, StalePluginEntry } from '../../api/types'

interface Props {
  entries: readonly StalePluginEntry[]
  /** Called when the user clicks "Remove from config" on one row.
   *  Caller wires this to ``removeInstance(kind, parsed, instanceId)``
   *  + the same commit path normal removes use. */
  onRemove: (kind: PluginKind, instanceId: string) => void
  /** Disables every Remove button while a save is in flight — prevents
   *  double-clicks from racing into the same commit twice. */
  busy?: boolean
}

export function StalePluginsBanner({ entries, onRemove, busy = false }: Props) {
  if (!entries || entries.length === 0) return null

  return (
    <div
      role="alert"
      data-testid="stale-plugins-banner"
      className="rounded-md border border-warn/40 bg-warn/10 text-warn-fg text-xs px-3 py-2 mb-3"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="font-medium">
            {entries.length} stale plugin reference
            {entries.length === 1 ? '' : 's'} in this config
          </div>
          <div className="text-text-2 mt-0.5">
            The plugin{entries.length === 1 ? '' : 's'} below
            {entries.length === 1 ? ' is' : ' are'} no longer in the catalog.
            Remove the reference{entries.length === 1 ? '' : 's'} or restore the
            plugin folder under <code className="font-mono">community/</code>.
          </div>
        </div>
      </div>
      <ul className="mt-2 space-y-1 text-[11px]">
        {entries.map((entry) => (
          <li
            key={`${entry.plugin_kind}:${entry.instance_id}`}
            className="flex items-center justify-between gap-2"
            data-testid="stale-plugin-row"
          >
            <span className="min-w-0 truncate">
              <span className="uppercase tracking-wide text-text-2 mr-1">
                {entry.plugin_kind}
              </span>
              <span className="font-mono">{entry.plugin_name}</span>
              <span className="text-text-3 mx-1">·</span>
              <span className="text-text-2 font-mono">{entry.location}</span>
            </span>
            <button
              type="button"
              onClick={() =>
                onRemove(entry.plugin_kind as PluginKind, entry.instance_id)
              }
              disabled={busy}
              className="btn-ghost h-7 text-[11px] px-2 disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
              data-testid={`stale-plugin-remove-${entry.plugin_kind}-${entry.instance_id}`}
              title={`Drop this reference from the config and save.`}
            >
              Remove from config
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
