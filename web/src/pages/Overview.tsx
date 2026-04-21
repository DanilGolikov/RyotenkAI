import { useRef } from 'react'
import { ActivityFeed } from '../components/ActivityFeed'
import { HealthHero } from '../components/HealthHero'

export function Overview() {
  const activityRef = useRef<HTMLElement | null>(null)

  const scrollToActivity = () => {
    activityRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <div className="p-5 space-y-5 max-w-[1400px]">
      <section className="space-y-1">
        <h1 className="text-2xl font-semibold text-ink-1">Overview</h1>
        <p className="text-xs text-ink-3">Pipeline fleet at a glance.</p>
      </section>

      <HealthHero onReviewFailures={scrollToActivity} />
      <ActivityFeed ref={activityRef} />
    </div>
  )
}
