/**
 * MiniStageTimeline — phase-6 representative "list/row" story.
 *
 * Three variants: a typical 5-stage run, a single-stage edge case
 * (the loop guard branch), and a run with a failure halfway through.
 * The component renders coloured segments rather than text, so it's
 * a natural visual-regression target.
 */

import type { Meta, StoryObj } from '@storybook/react'
import { MiniStageTimeline } from './MiniStageTimeline'
import type { StageRun } from '../api/types'

const FIXED_TS = '2026-05-10T12:00:00Z'

function stage(name: string, status: StageRun['status']): StageRun {
  return {
    name,
    status,
    started_at: FIXED_TS,
    completed_at: FIXED_TS,
  } as StageRun
}

const meta: Meta<typeof MiniStageTimeline> = {
  title: 'Components/MiniStageTimeline',
  component: MiniStageTimeline,
}
export default meta
type Story = StoryObj<typeof MiniStageTimeline>

export const Default: Story = {
  args: {
    stages: [
      stage('prepare', 'completed'),
      stage('train', 'completed'),
      stage('eval', 'completed'),
      stage('export', 'running'),
      stage('upload', 'pending'),
    ],
    variant: 'mini',
  },
}

export const SingleStage: Story = {
  args: {
    stages: [stage('only', 'running')],
    variant: 'mini',
  },
}

export const PartialFailure: Story = {
  args: {
    stages: [
      stage('prepare', 'completed'),
      stage('train', 'failed'),
      stage('eval', 'skipped'),
      stage('export', 'skipped'),
    ],
    variant: 'mini',
  },
}
