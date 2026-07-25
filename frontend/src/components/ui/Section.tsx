import type { ReactNode } from 'react'
import { cn } from '../../lib/utils'

/** Standard section: centered max-width container with vertical rhythm. */
export function Section({
  children,
  id,
  className,
  containerClassName,
}: {
  children: ReactNode
  id?: string
  className?: string
  containerClassName?: string
}) {
  return (
    <section id={id} className={cn('relative py-20 md:py-28', className)}>
      <div className={cn('mx-auto w-full max-w-6xl px-5 sm:px-8', containerClassName)}>
        {children}
      </div>
    </section>
  )
}
