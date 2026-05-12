/**
 * StalePluginsBanner — phase-6 representative "banner / edge-case"
 * story. We exercise the empty branch (renders null — useful in the
 * docs page), the one-entry pluralisation branch, and the
 * many-entries branch with `busy` true.
 */

import type { Meta, StoryObj } from '@storybook/react'
import { StalePluginsBanner } from './StalePluginsBanner'
import type { StalePluginEntry } from '../../api/types'

const entry = (
  kind: 'reward' | 'validation' | 'evaluation' | 'reports',
  name: string,
  instanceId: string,
): StalePluginEntry =>
  ({
    plugin_kind: kind,
    plugin_name: name,
    instance_id: instanceId,
    location: `pipeline.stages.${kind}.plugins[${instanceId}]`,
  }) as StalePluginEntry

const meta: Meta<typeof StalePluginsBanner> = {
  title: 'Project/StalePluginsBanner',
  component: StalePluginsBanner,
}
export default meta
type Story = StoryObj<typeof StalePluginsBanner>

export const Empty: Story = {
  args: { entries: [], onRemove: () => {} },
}

export const OneEntry: Story = {
  args: {
    entries: [entry('reward', 'helixql_pairwise', 'rwd-1')],
    onRemove: () => {},
  },
}

export const ManyEntriesBusy: Story = {
  args: {
    entries: [
      entry('reward', 'helixql_pairwise', 'rwd-1'),
      entry('validation', 'syntax_validator', 'val-1'),
      entry('evaluation', 'rouge_l', 'eval-1'),
      entry('reports', 'wandb_uploader', 'rep-1'),
    ],
    onRemove: () => {},
    busy: true,
  },
}
