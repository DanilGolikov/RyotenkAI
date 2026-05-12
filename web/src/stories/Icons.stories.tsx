/**
 * Icons — phase-6 representative "pure presentational" story.
 *
 * Renders the full icon set as a grid so visual regression catches
 * SVG path drift in a single screenshot. Three variants stress
 * different sizes and `currentColor` inheritance.
 */

import type { Meta, StoryObj } from '@storybook/react'
import {
  LinkIcon,
  CheckIcon,
  ChevronRightIcon,
  InfoIcon,
  AlertIcon,
  ExpandIcon,
  CollapseIcon,
} from '../components/icons'

const ICONS = [
  { name: 'LinkIcon', Icon: LinkIcon },
  { name: 'CheckIcon', Icon: CheckIcon },
  { name: 'ChevronRightIcon', Icon: ChevronRightIcon },
  { name: 'InfoIcon', Icon: InfoIcon },
  { name: 'AlertIcon', Icon: AlertIcon },
  { name: 'ExpandIcon', Icon: ExpandIcon },
  { name: 'CollapseIcon', Icon: CollapseIcon },
] as const

function IconGrid({
  size,
  color = 'currentColor',
}: {
  size: string
  color?: string
}) {
  return (
    <div
      className="grid grid-cols-4 gap-4"
      style={{ color }}
    >
      {ICONS.map(({ name, Icon }) => (
        <div
          key={name}
          className="flex flex-col items-center gap-1 p-2 rounded border border-line-1"
        >
          <Icon className={size} />
          <span className="font-mono text-2xs text-ink-3">{name}</span>
        </div>
      ))}
    </div>
  )
}

const meta: Meta<typeof IconGrid> = {
  title: 'Design/Icons',
  component: IconGrid,
}
export default meta
type Story = StoryObj<typeof IconGrid>

export const Default: Story = {
  args: { size: 'w-4 h-4' },
}

export const Large: Story = {
  args: { size: 'w-8 h-8' },
}

export const ColoredBrand: Story = {
  args: { size: 'w-6 h-6', color: '#d6305f' },
}
