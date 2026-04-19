# Overview Page — Monitoring-first Redesign

## Context

Главная страница Overview сейчас ощущается кашей:
две параллельные колонки **Live runs** + **Recent** ([web/src/pages/Overview.tsx](../../web/src/pages/Overview.tsx))
показывают одно и то же через `RunRow`, при этом Live почти всегда пуст —
визуально «половина экрана пустая, вторая дублируется». Наверху 5 KPI
([web/src/components/KpiStrip.tsx](../../web/src/components/KpiStrip.tsx))
all-time, без sparkline-ов, без ответа на вопрос «всё ли ок прямо сейчас?».
Нет единого dominant CTA.

Задача — переделать Overview в **monitoring-first** dashboard по
Shneiderman-мантре *overview → zoom → details*: первое касание отвечает
на «что происходит и всё ли ок»; детали — на /runs.

## Decisions (подтверждено)

- **Focus**: monitoring-first. Big hero health state, Activity feed —
  центральная секция.
- **KPI-полоска удаляется**. Все числа — inline в hero.
- **Running runs получают inline mini stage-timeline** в Activity (8 сегментов × 4px).
- Primary CTA **«Launch new run»** живёт в header'е hero, не в отдельной
  секции.
- **Sparkline-чарты и analytics НЕ в MVP** — данных мало (5 runs), это
  visual bloat.

## Reuse Surface

### Hooks
- [useRuns](../../web/src/api/hooks/useRuns.ts) — polling 5s, база для всего.
- [useKpis](../../web/src/api/hooks/useKpis.ts) — расширяется (новые поля).
- [useStages](../../web/src/api/hooks/useAttempt.ts) — per-attempt stage list,
  3s refresh. Будем запрашивать только для **running runs** (их 0-2 обычно).
- TanStack Query **`useQueries`** — batched queries для массы running runs.

### Components
- [StatusPill](../../web/src/components/StatusPill.tsx) — оставляем, используем в ActivityRow.
- [RunRow](../../web/src/components/RunRow.tsx) — **не используется на Overview** после редизайна.
  Остаётся для /runs.
- [KpiStrip](../../web/src/components/KpiStrip.tsx) — **удаляется** (только в Overview использовался).
- [Card / EmptyState / Spinner / SectionHeader](../../web/src/components/ui.tsx) — переиспользуем.
- Colour tokens: `card-hero` (subtle gradient), `bg-gradient-brand`, `status-*` tokens.

### Types
- `RunSummary.duration_seconds`, `created_ts`, `started_at`, `completed_at`, `error`
  — используем для inline metrics + error excerpt.
- `RunDetail.running_attempt_no` — для Stage timeline.
- `StageRun[]` — рисуем в MiniStageTimeline.

## Implementation Plan

### Architecture

```
<Overview>
  ├── <HealthHero>                 ← new
  │     "2 runs live" | "1 failure" | "All idle"
  │     inline stats · Launch CTA
  ├── <ActivityFeed>               ← new, merges Live + Recent
  │     ├── <ActivityRow running>     ← new, wraps <MiniStageTimeline>
  │     ├── <ActivityRow failed>
  │     └── <ActivityRow completed>
```

### 1. `HealthHero` — `web/src/components/HealthHero.tsx` (new)
- Single card with `card-hero` surface.
- Left: giant status text (2xl, heavy). Logic:
  - `activeRuns > 0` → **"{N} run{s} live"** + live-dot.
  - else if `failures24h > 0` → **"{N} failure{s} in last 24h"** + coral accent + chip "Review →" scrolls to ActivityFeed with filter.
  - else → **"All idle"** + hint "last run {name} {timeago}".
- Right (inline row, text-ink-2 · separator): **Today {n}** · **Success 7d {%}** · **Median 7d {duration}**.
- Far right: `Launch new run` primary gradient button → `/launch`.
- If failures24h > 0 the background tint shifts one notch (subtle coral wash from `card-hero` definition).

### 2. `MiniStageTimeline` — `web/src/components/MiniStageTimeline.tsx` (new)
- Compact 4px-height strip of 8 segments (or len(stages) if <8).
- Segment fill: semantic token by `stage.status` (reuse same tokens as `StageTimeline`).
- Running segment pulses `bg-info/70`.
- Props: `stages: StageRun[]`, `variant?: 'micro' | 'mini'`.

### 3. `ActivityFeed` — `web/src/components/ActivityFeed.tsx` (new)
- Uses `useRuns()` for list + `useQueries` for stage data of running runs
  (batch: ids derived from `flat.filter(r=>r.status==='running')`).
- Sort: running → failed-24h → rest by `created_ts` desc. Cap at 6 rows.
- Container: `card` with section header "Activity".
- Inner: `<ActivityRow>` (new, inline — lives in same file).

### 4. `ActivityRow` (inside ActivityFeed)
- Full-row clickable → `/runs/{encodeURIComponent(run.run_id)}`.
- Layout variants:
  - **running**: `StatusPill` + run_id + live-dot + elapsed timer + stages count → `MiniStageTimeline` takes full row bottom.
  - **failed**: `StatusPill(err)` + run_id + config + duration + truncated error excerpt (`text-err-muted`).
  - **completed / skipped / interrupted**: `StatusPill` + run_id + config + duration + finished timeago.
- All three variants share top-row shape (status, id, timeago).

### 5. `useKpis` extensions — `web/src/api/hooks/useKpis.ts`
Add to the returned `Kpi` shape:
```ts
runsToday: number                  // started_at within last 24h
successRate7d: number | null       // completed / (completed+failed+interrupted) in last 7 days
medianDuration7d: number | null    // median of durations for terminal runs in last 7 days
failures24h: RunSummary[]          // list (not just count) for Review link
mostRecentTerminal: RunSummary | null  // for "All idle" subline
```
Median computed via standard sort→middle (no deps). `activeRuns` / `failuresLast24h` (count) stay.

### 6. `Overview.tsx` — full rewrite
```tsx
<div className="p-5 space-y-5 max-w-[1400px]">
  <section>
    <h1 className="text-2xl font-semibold gradient-text">Overview</h1>
    <p className="text-xs text-ink-3">Pipeline fleet at a glance.</p>
  </section>
  <HealthHero />
  <ActivityFeed />
</div>
```
Drop: Live runs + Recent columns, KpiStrip, Start-a-run EmptyState
(CTA now in hero).

### 7. Delete unused
- `web/src/components/KpiStrip.tsx` — удаляется (imports только в Overview).
- `web/src/api/hooks/useKpis.ts` **остаётся** (пригодится), но export shape расширяется.

## Files

**Create**
- `web/src/components/HealthHero.tsx`
- `web/src/components/ActivityFeed.tsx` (с ActivityRow внутри)
- `web/src/components/MiniStageTimeline.tsx`

**Modify**
- `web/src/pages/Overview.tsx` — полная переделка
- `web/src/api/hooks/useKpis.ts` — добавить runsToday / successRate7d / medianDuration7d / failures24h / mostRecentTerminal

**Delete**
- `web/src/components/KpiStrip.tsx`

## Verification

1. **Type check**: `cd web && npx tsc --noEmit` → no errors.
2. **Build smoke**: `npm run build` → bundle compiles.
3. **Preview (hot-reload через preview_start `web-frontend`):**
   - **Idle state**: навести `useRuns` на runs/ без running → hero показывает "All idle" + last run name/timeago. Activity показывает 6 recent.
   - **Running state**: запустить фейковый run (touch `runs/test/run.lock` + минимальный `pipeline_state.json` с `status: running` и одним `stage_runs` record) → hero переходит в "1 run live" + live-dot, Activity первая строка — с MiniStageTimeline.
   - **Failed state**: создать run с `status: failed` и `started_at` внутри 24h → hero показывает "1 failure in last 24h" + Review chip. Клик → скроллит к ActivityFeed.
   - **Launch CTA** → `/launch`.
   - **Click любой ActivityRow** → `/runs/{id}`.
4. **Responsive**: resize preview до 1024 — hero и Activity перестраиваются, без обрезания metric-строки (truncate).
5. **Regression**: не трогаем API, тесты в `src/tests/{integration,e2e}/api/` остаются зелёными без запуска (сверяемся с последним прогоном 161 passed).

## Out of Scope (Phase 2)

- Sparkline-чарты (runs/day, success-rate trend) — добавим когда история вырастет.
- Per-run MLflow deeplinks в hero — пока линк с RunDetail.
- Notification toasts на окончание running runs.
- Group-by filter (subgroup) на Activity — сейчас идёт плоско.
