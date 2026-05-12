# `web/.lost-pixel/` — visual regression baselines

Phase-6 visual diffs live here. Three subdirs:

- `baseline/` — committed PNGs (one per Storybook story). Regenerated
  via `npm run visual-test-update`; the resulting tree is `git add`ed
  in the same PR that introduces a new story.
- `current/` — last `npm run visual-test` run output. Git-ignored.
- `difference/` — overlay PNGs when a diff is detected. Git-ignored.

Baselines are committed, not generated on CI, so PR diffs are visible
and reviewable by humans. See [../lost-pixel.config.ts](../lost-pixel.config.ts)
for masks (timestamp regions etc.) and the threshold.

**Status: non-blocking warning** for the first 6 months. After that
window we flip `failOnDifference: true` in the config.
