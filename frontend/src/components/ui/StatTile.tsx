import type { ReactNode } from 'react'
import { cn } from '../../lib/utils'

/**
 * KPI tile. Optionally clickable (e.g. band cards that filter a table); when
 * `active` it gets a solid dark treatment to show the current selection.
 */
export function StatTile({
  label,
  value,
  hint,
  icon,
  active = false,
  onClick,
}: {
  label: string
  value: ReactNode
  hint?: ReactNode
  icon?: ReactNode
  active?: boolean
  onClick?: () => void
}) {
  const clickable = !!onClick
  return (
    <button
      type="button"
      disabled={!clickable}
      onClick={onClick}
      className={cn(
        'flex w-full items-start gap-3 rounded-2xl p-4 text-left transition-all',
        active ? 'bg-[#14141a] text-white' : 'glass',
        clickable && !active && 'glass-hover cursor-pointer',
        !clickable && 'cursor-default',
      )}
    >
      {icon && (
        <span
          className={cn(
            'grid h-9 w-9 shrink-0 place-items-center rounded-xl',
            active ? 'bg-white/15 text-white' : 'bg-black/[0.05] text-fg',
          )}
        >
          {icon}
        </span>
      )}
      <div className="min-w-0">
        <p className={cn('text-xs font-medium', active ? 'text-white/70' : 'text-fg-faint')}>
          {label}
        </p>
        <p className="mt-0.5 text-2xl font-bold leading-none">{value}</p>
        {hint && (
          <p className={cn('mt-1 text-xs', active ? 'text-white/60' : 'text-fg-muted')}>{hint}</p>
        )}
      </div>
    </button>
  )
}
