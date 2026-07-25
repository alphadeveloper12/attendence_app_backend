import type { ElementType, ReactNode } from 'react'
import { cn } from '../../lib/utils'

type Props = {
  children: ReactNode
  className?: string
  /** stronger blur + brighter surface */
  strong?: boolean
  /** lift + glow on hover */
  hover?: boolean
  as?: ElementType
}

/** Frosted-glass surface — the core building block of the design system. */
export function GlassCard({
  children,
  className,
  strong = false,
  hover = false,
  as: Tag = 'div',
}: Props) {
  return (
    <Tag
      className={cn(
        'rounded-3xl',
        strong ? 'glass-strong' : 'glass',
        hover && 'glass-hover',
        className,
      )}
    >
      {children}
    </Tag>
  )
}
