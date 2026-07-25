import type { ReactNode } from 'react'
import * as RadixTabs from '@radix-ui/react-tabs'
import { cn } from '../../lib/utils'

export interface TabItem {
  value: string
  label: ReactNode
  content: ReactNode
}

/**
 * Segmented tabs built on Radix (accessible: arrow-key nav, roving focus).
 * Pass an array of `{ value, label, content }`. Controlled or uncontrolled.
 */
export function Tabs({
  items,
  value,
  defaultValue,
  onValueChange,
  className,
}: {
  items: TabItem[]
  value?: string
  defaultValue?: string
  onValueChange?: (value: string) => void
  className?: string
}) {
  return (
    <RadixTabs.Root
      value={value}
      defaultValue={defaultValue ?? items[0]?.value}
      onValueChange={onValueChange}
      className={className}
    >
      <RadixTabs.List className="mb-4 inline-flex flex-wrap gap-1 rounded-xl bg-black/[0.04] p-1">
        {items.map((t) => (
          <RadixTabs.Trigger
            key={t.value}
            value={t.value}
            className={cn(
              'rounded-lg px-4 py-1.5 text-sm font-semibold text-fg-muted transition-all',
              'hover:text-fg data-[state=active]:bg-white data-[state=active]:text-fg data-[state=active]:shadow-sm',
            )}
          >
            {t.label}
          </RadixTabs.Trigger>
        ))}
      </RadixTabs.List>
      {items.map((t) => (
        <RadixTabs.Content key={t.value} value={t.value} className="focus:outline-none">
          {t.content}
        </RadixTabs.Content>
      ))}
    </RadixTabs.Root>
  )
}
