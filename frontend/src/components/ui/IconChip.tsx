import type { ReactNode } from 'react'
import { cn } from '../../lib/utils'

type Tone = 'lime' | 'teal' | 'gold' | 'ghost'

// Monochrome icon chips: a dark tile with a white icon (crisp on the light UI).
const tones: Record<Tone, string> = {
  lime: 'bg-[#14141a] text-white',
  teal: 'bg-[#14141a] text-white',
  gold: 'bg-[#14141a] text-white',
  ghost: 'surface text-fg',
}

/** Solid accent icon badge — the reference's circular colored icon chips. */
export function IconChip({
  children,
  tone = 'lime',
  rounded = 'rounded-2xl',
  className,
}: {
  children: ReactNode
  tone?: Tone
  rounded?: string
  className?: string
}) {
  return (
    <span
      className={cn(
        'grid h-11 w-11 shrink-0 place-items-center',
        rounded,
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

/** Cycle accent tones by index for multi-item grids. */
export const toneByIndex = (i: number): Tone =>
  (['lime', 'teal', 'gold'] as const)[i % 3]
