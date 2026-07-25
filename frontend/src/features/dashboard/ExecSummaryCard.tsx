import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Sparkles, ChevronDown } from 'lucide-react'
import { getExecutiveSummary } from '../../lib/api/dashboard'
import { GlassCard } from '../../components/ui/GlassCard'
import { SkeletonRows } from '../../components/ui/Skeleton'
import { cn } from '../../lib/utils'

/** Severity → left-accent tone (monochrome). */
const sevAccent: Record<string, string> = {
  bad: 'bg-[#14141a]',
  warn: 'bg-[#52525b]',
  good: 'bg-[#a1a1aa]',
  neutral: 'bg-black/15',
}

/**
 * Executive Summary — five plain-language bullets the server derives from the
 * day's data (alerts, absences, document expiry, manpower balance, attendance
 * rate). Toggles daily/weekly. Deterministic by default; the server can polish
 * the prose with an LLM but the numbers are unchanged.
 */
export function ExecSummaryCard() {
  const [period, setPeriod] = useState<'daily' | 'weekly'>('daily')

  const { data, isLoading } = useQuery({
    queryKey: ['exec-summary', period],
    queryFn: () => getExecutiveSummary({ period }),
  })

  return (
    <GlassCard className="p-5">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="grid h-8 w-8 place-items-center rounded-xl bg-black/[0.05] text-fg">
            <Sparkles size={16} />
          </span>
          <h2 className="font-semibold text-fg">Executive summary</h2>
        </div>
        <div className="inline-flex rounded-lg bg-black/[0.04] p-0.5 text-sm">
          {(['daily', 'weekly'] as const).map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => setPeriod(p)}
              className={cn(
                'rounded-md px-3 py-1 font-medium capitalize transition-all',
                period === p ? 'bg-white text-fg shadow-sm' : 'text-fg-muted hover:text-fg',
              )}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <SkeletonRows rows={5} />
      ) : (
        <ul className="space-y-2.5">
          {data?.bullets.map((b, i) => (
            <li key={i} className="flex items-start gap-3">
              <span className={cn('mt-1.5 h-2 w-2 shrink-0 rounded-full', sevAccent[b.severity] ?? sevAccent.neutral)} />
              <p className="text-sm leading-relaxed text-fg">
                <span className="mr-1.5">{b.icon}</span>
                {b.text}
              </p>
            </li>
          ))}
        </ul>
      )}
    </GlassCard>
  )
}

/** Compact wrapper so the card can sit in a collapsible slot if desired. */
export function ExecSummaryCollapsible() {
  const [open, setOpen] = useState(true)
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="mb-2 flex items-center gap-1 text-sm font-medium text-fg-muted hover:text-fg lg:hidden"
      >
        <ChevronDown size={15} className={cn('transition-transform', !open && '-rotate-90')} />
        Executive summary
      </button>
      <div className={cn(!open && 'hidden lg:block')}>
        <ExecSummaryCard />
      </div>
    </div>
  )
}
