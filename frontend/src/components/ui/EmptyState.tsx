import type { ReactNode } from 'react'
import { Inbox } from 'lucide-react'

/** Friendly placeholder shown when a list/query returns nothing. */
export function EmptyState({
  title = 'Nothing to show',
  message,
  icon,
}: {
  title?: string
  message?: ReactNode
  icon?: ReactNode
}) {
  return (
    <div className="grid place-items-center gap-2 px-6 py-16 text-center">
      <span className="grid h-12 w-12 place-items-center rounded-2xl bg-black/[0.04] text-fg-faint">
        {icon ?? <Inbox size={22} />}
      </span>
      <p className="text-sm font-semibold text-fg">{title}</p>
      {message && <p className="max-w-sm text-sm text-fg-muted">{message}</p>}
    </div>
  )
}
