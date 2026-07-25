import type { ReactNode } from 'react'
import { cn } from '../../lib/utils'

/** Small frosted eyebrow/label pill, optionally with a leading dot. */
export function Badge({
  children,
  className,
  dot = true,
}: {
  children: ReactNode
  className?: string
  dot?: boolean
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-2 rounded-full glass px-3.5 py-1.5',
        'text-[12.5px] font-medium tracking-wide text-fg-muted uppercase',
        className,
      )}
    >
      {dot && (
        <span className="relative flex h-1.5 w-1.5">
          <span className="absolute inline-flex h-full w-full rounded-full bg-fence animate-pulse-ring" />
          <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-fence" />
        </span>
      )}
      {children}
    </span>
  )
}
