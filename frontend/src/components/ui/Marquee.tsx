import type { ReactNode } from 'react'
import { cn } from '../../lib/utils'

/**
 * Infinite horizontal marquee. Renders its children twice and translates by
 * -50% so the loop is seamless. Pauses on hover; edges fade out.
 */
export function Marquee({
  children,
  className,
  duration = 40,
  pauseOnHover = true,
}: {
  children: ReactNode
  className?: string
  duration?: number
  pauseOnHover?: boolean
}) {
  return (
    <div
      className={cn(
        'group relative flex overflow-hidden',
        '[mask-image:linear-gradient(to_right,transparent,#000_8%,#000_92%,transparent)]',
        className,
      )}
    >
      <div
        className={cn(
          'flex shrink-0 items-center gap-4 pr-4 animate-marquee',
          pauseOnHover && 'group-hover:[animation-play-state:paused]',
        )}
        style={{ ['--marquee-duration' as string]: `${duration}s` }}
      >
        {children}
        {children}
      </div>
    </div>
  )
}
