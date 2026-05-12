/**
 * PluginCatalogCard — phase-6 representative "status/badge" story.
 *
 * Three variants: stable + reward (default), beta + validation (mid
 * stability), experimental + evaluation (most attention-grabbing
 * combo). The card composes stability and kind badges in different
 * combinations, so this is a good single-screenshot regression target.
 */

import type { Meta, StoryObj } from '@storybook/react'
import { PluginCatalogCard } from './PluginCatalogCard'
import type { PluginManifest } from '../api/types'

const baseManifest: PluginManifest = {
  schema_version: 5,
  id: 'helixql_pairwise_reward',
  name: 'HelixQL Pairwise Reward',
  version: '1.2.0',
  description:
    'Pairwise preference reward optimised for HelixQL completions. ' +
    'Compares two candidate completions and assigns +1 / -1 / 0.',
  category: 'reward',
  stability: 'stable',
  kind: 'reward',
  author: 'ryotenkai-core',
  params_schema: { properties: { temperature: { type: 'number' } } },
  thresholds_schema: { properties: { min_score: { type: 'number' } } },
} as PluginManifest

const meta: Meta<typeof PluginCatalogCard> = {
  title: 'Components/PluginCatalogCard',
  component: PluginCatalogCard,
}
export default meta
type Story = StoryObj<typeof PluginCatalogCard>

export const StableReward: Story = {
  args: { plugin: baseManifest, onInfoClick: () => {} },
}

export const BetaValidation: Story = {
  args: {
    plugin: {
      ...baseManifest,
      id: 'syntax_validator_beta',
      name: 'Syntax Validator (beta)',
      stability: 'beta',
      kind: 'validation',
    } as PluginManifest,
    onInfoClick: () => {},
  },
}

export const ExperimentalEvaluation: Story = {
  args: {
    plugin: {
      ...baseManifest,
      id: 'rouge_l_alpha',
      name: 'ROUGE-L (alpha)',
      stability: 'experimental',
      kind: 'evaluation',
      description:
        'Edge-case evaluator with no thresholds defined — exercises ' +
        'the empty-schema branch.',
      thresholds_schema: { properties: {} },
    } as PluginManifest,
    onInfoClick: () => {},
  },
}
