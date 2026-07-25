import type { ReactNode } from 'react'
import { cn } from '../../lib/utils'

/**
 * Small status/label pill. Monochrome tones only (theme is black & white):
 * `solid` = dark fill, `outline` = bordered, `soft` = subtle grey.
 */
export function Pill({
  children,
  tone = 'soft',
  className,
}: {
  children: ReactNode
  tone?: 'solid' | 'outline' | 'soft'
  className?: string
}) {
  const tones = {
    solid: 'bg-[#14141a] text-white',
    outline: 'border border-black/15 text-fg',
    soft: 'bg-black/[0.05] text-fg-muted',
  }
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold',
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}
