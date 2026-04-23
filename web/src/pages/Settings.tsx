import { NavLink, Outlet } from 'react-router-dom'

const SUBTABS: { to: string; label: string; disabled?: boolean; hint?: string }[] = [
  { to: 'providers', label: 'Providers' },
  { to: 'catalog', label: 'Catalog' },
  { to: 'datasets', label: 'Datasets', disabled: true, hint: 'soon' },
  { to: 'models', label: 'Models', disabled: true, hint: 'soon' },
]

export function SettingsPage() {
  return (
    <div className="px-6 py-6 grid grid-cols-[200px_1fr] gap-6">
      <aside className="space-y-0.5">
        <div className="text-2xs uppercase tracking-wide text-ink-3 px-2 mb-1">Settings</div>
        {SUBTABS.map((t) =>
          t.disabled ? (
            <div
              key={t.to}
              className="flex items-center justify-between px-2 py-1.5 rounded-md text-xs text-ink-4 cursor-not-allowed"
            >
              <span>{t.label}</span>
              {t.hint && <span className="text-[0.6rem]">{t.hint}</span>}
            </div>
          ) : (
            <NavLink
              key={t.to}
              to={t.to}
              className={({ isActive }) =>
                [
                  'block px-2 py-1.5 rounded-md text-xs transition',
                  isActive
                    ? 'bg-surface-2 text-ink-1 border-l-2 border-brand -ml-0.5 pl-[0.625rem]'
                    : 'text-ink-3 hover:text-ink-1 hover:bg-surface-2',
                ].join(' ')
              }
            >
              {t.label}
            </NavLink>
          ),
        )}
      </aside>
      <main>
        <Outlet />
      </main>
    </div>
  )
}
