import { Link } from 'react-router-dom'
import { useKpis } from '../api/hooks/useKpis'
import { formatDuration, timeAgo } from '../lib/format'

type HeroTone = 'idle' | 'live' | 'alert'

function pluralise(n: number, singular: string, plural: string) {
  return `${n} ${n === 1 ? singular : plural}`
}

function InlineStats({
  today,
  successRate7d,
  medianDuration7dSec,
}: {
  today: number
  successRate7d: number | null
  medianDuration7dSec: number | null
}) {
  const success =
    successRate7d == null ? '—' : `${Math.round(successRate7d * 100)}%`
  const duration =
    medianDuration7dSec == null ? '—' : formatDuration(medianDuration7dSec)
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-xs text-ink-2">
      <div>
        <span className="text-ink-3">Today </span>
        <span className="font-semibold tabular-nums text-ink-1">{today}</span>
      </div>
      <span className="text-ink-4">·</span>
      <div>
        <span className="text-ink-3">Success 7d </span>
        <span className="font-semibold tabular-nums text-ink-1">{success}</span>
      </div>
      <span className="text-ink-4">·</span>
      <div>
        <span className="text-ink-3">Median 7d </span>
        <span className="font-semibold tabular-nums text-ink-1">{duration}</span>
      </div>
    </div>
  )
}

function HeroStatus({
  tone,
  title,
  subtitle,
  action,
}: {
  tone: HeroTone
  title: React.ReactNode
  subtitle?: React.ReactNode
  action?: React.ReactNode
}) {
  const titleTone =
    tone === 'alert' ? 'text-err' : tone === 'live' ? 'text-ink-1' : 'text-ink-2'
  return (
    <div className="min-w-0">
      <div className="flex items-center gap-3">
        {tone === 'live' && <span className="live-dot" aria-hidden />}
        <h2 className={`text-2xl font-semibold leading-tight truncate ${titleTone}`}>{title}</h2>
      </div>
      {subtitle && <div className="text-xs text-ink-3 mt-1">{subtitle}</div>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  )
}

function LaunchCTA() {
  return (
    <Link to="/launch" className="btn-primary whitespace-nowrap">
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M5 12h14" />
        <path d="M12 5l7 7-7 7" />
      </svg>
      Launch new run
    </Link>
  )
}

export function HealthHero({ onReviewFailures }: { onReviewFailures?: () => void }) {
  const { data, isLoading } = useKpis()

  if (isLoading && !data) {
    return <div className="card-hero h-[116px] animate-pulse" />
  }

  const activeRuns = data?.activeRuns ?? 0
  const failures = data?.failures24h ?? []
  const failuresCount = failures.length

  let tone: HeroTone = 'idle'
  let title: React.ReactNode = 'All idle'
  let subtitle: React.ReactNode = 'no active runs'
  let extraAction: React.ReactNode = null

  if (activeRuns > 0) {
    tone = 'live'
    title = <>{pluralise(activeRuns, 'run live', 'runs live')}</>
    if (failuresCount > 0) {
      subtitle = (
        <>
          {pluralise(failuresCount, 'failure', 'failures')} in last 24h
          {onReviewFailures && (
            <>
              {' · '}
              <button type="button" onClick={onReviewFailures} className="text-err hover:underline">
                review →
              </button>
            </>
          )}
        </>
      )
    } else {
      subtitle = 'pipeline executing'
    }
  } else if (failuresCount > 0) {
    tone = 'alert'
    title = <>{pluralise(failuresCount, 'failure', 'failures')} in last 24h</>
    const first = failures[0]
    subtitle = first ? (
      <>
        latest: <span className="text-ink-2 font-mono">{first.run_id}</span>
        {first.completed_at && <> · {timeAgo(first.completed_at)}</>}
      </>
    ) : null
    if (onReviewFailures) {
      extraAction = (
        <button
          type="button"
          onClick={onReviewFailures}
          className="inline-flex items-center gap-1 text-xs text-err border border-err/40 rounded-full px-2.5 py-0.5 hover:bg-err/10 transition"
        >
          Review failures →
        </button>
      )
    }
  } else {
    const last = data?.mostRecentTerminal
    if (last) {
      subtitle = (
        <>
          last run <span className="text-ink-2 font-mono">{last.run_id}</span>
          {last.completed_at && <> completed {timeAgo(last.completed_at)}</>}
        </>
      )
    } else {
      subtitle = 'no runs yet'
    }
  }

  // Alert tone gets a slightly warmer coral wash over the default hero gradient.
  const heroClasses =
    tone === 'alert'
      ? 'card-hero !bg-none bg-[linear-gradient(180deg,rgba(248,113,113,0.10),rgba(198,48,107,0.04)_55%,transparent),linear-gradient(#1c1a23,#1c1a23)]'
      : 'card-hero'

  return (
    <section className={`${heroClasses} px-5 py-4 flex items-center gap-6 flex-wrap`}>
      <HeroStatus tone={tone} title={title} subtitle={subtitle} action={extraAction} />
      <div className="flex-1 min-w-[240px]" />
      <InlineStats
        today={data?.runsToday ?? 0}
        successRate7d={data?.successRate7d ?? null}
        medianDuration7dSec={data?.medianDuration7dSec ?? null}
      />
      <LaunchCTA />
    </section>
  )
}
