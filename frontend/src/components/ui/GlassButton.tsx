import type { ReactNode } from 'react'
import { cn } from '../../lib/utils'

type Variant = 'primary' | 'glass' | 'ghost'
type Size = 'sm' | 'md' | 'lg'

type Props = {
  children: ReactNode
  href?: string
  variant?: Variant
  size?: Size
  className?: string
  onClick?: () => void
}

const base =
  'inline-flex items-center justify-center gap-2 font-semibold rounded-full ' +
  'transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 ' +
  'focus-visible:ring-black/40 focus-visible:ring-offset-2 focus-visible:ring-offset-transparent ' +
  'active:scale-[0.98] whitespace-nowrap'

const sizes: Record<Size, string> = {
  sm: 'text-sm px-4 py-2',
  md: 'text-[15px] px-5 py-2.5',
  lg: 'text-base px-7 py-3.5',
}

const variants: Record<Variant, string> = {
  primary:
    'text-white bg-[#14141a] hover:bg-black ' +
    'shadow-[0_12px_34px_-12px_rgba(17,17,26,0.45)] hover:shadow-[0_18px_44px_-12px_rgba(17,17,26,0.55)] ' +
    'hover:-translate-y-0.5',
  glass:
    'glass text-fg hover:-translate-y-0.5 hover:border-black/15',
  ghost:
    'text-fg-muted hover:text-fg hover:bg-black/[0.04]',
}

export function GlassButton({
  children,
  href,
  variant = 'primary',
  size = 'md',
  className,
  onClick,
}: Props) {
  const classes = cn(base, sizes[size], variants[variant], className)
  if (href) {
    return (
      <a href={href} className={classes} onClick={onClick}>
        {children}
      </a>
    )
  }
  return (
    <button type="button" className={classes} onClick={onClick}>
      {children}
    </button>
  )
}
