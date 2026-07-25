import type { ReactNode } from 'react'
import { cn } from '../../lib/utils'
import { Badge } from './Badge'
import { Reveal } from './Reveal'

type Props = {
  eyebrow?: string
  title: ReactNode
  lede?: ReactNode
  align?: 'left' | 'center'
  className?: string
}

/** Consistent section header: eyebrow badge → display heading → lede. */
export function SectionHeading({
  eyebrow,
  title,
  lede,
  align = 'left',
  className,
}: Props) {
  return (
    <Reveal
      className={cn(
        'flex flex-col gap-4',
        align === 'center' && 'items-center text-center',
        className,
      )}
    >
      {eyebrow && <Badge>{eyebrow}</Badge>}
      <h2 className="text-4xl md:text-5xl font-extrabold text-gradient max-w-3xl">
        {title}
      </h2>
      {lede && (
        <p
          className={cn(
            'text-fg-muted text-lg leading-relaxed max-w-2xl',
            align === 'center' && 'mx-auto',
          )}
        >
          {lede}
        </p>
      )}
    </Reveal>
  )
}
