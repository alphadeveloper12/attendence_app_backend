import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Download, RotateCcw, History } from 'lucide-react'
import {
  getDistribution,
  distributionExportUrl,
  type DistributionType,
  type DistributionRow,
} from '../../lib/api/distribution'
import { useMe } from '../../lib/hooks/useMe'
import { PageHeader } from '../../components/ui/PageHeader'
import { GlassCard } from '../../components/ui/GlassCard'
import { EmptyState } from '../../components/ui/EmptyState'
import { SkeletonRows } from '../../components/ui/Skeleton'
import { Select, DatePicker } from '../../components/ui/Field'
import { cn } from '../../lib/utils'

const TABS: { key: DistributionType; label: string }[] = [
  { key: 'staff', label: 'Staff' },
  { key: 'worker', label: 'Workers' },
  { key: 'resource', label: 'Resources' },
]

/**
 * Manpower Distribution — the dept-wise / trade-wise / project-wise strength
 * table, in three tabs. Staff & Workers show a site × trade matrix; Resources
 * is a flat per-trade total. The server pre-shapes the rows (dept header, trade,
 * subtotal); we paint them and hand off Excel export to the server.
 */
export default function DistributionPage() {
  const { data: me } = useMe()
  const [type, setType] = useState<DistributionType>('staff')
  const [site, setSite] = useState('')
  // '' = live/today; 'YYYY-MM-DD' = a frozen historical snapshot.
  const [date, setDate] = useState('')

  const { data, isLoading } = useQuery({
    // Historical snapshots are all-sites, so drop the site filter when a date is set.
    queryKey: ['distribution', type, date ? '' : site, date],
    queryFn: () => getDistribution({ type, site: date ? undefined : site || undefined, date: date || undefined }),
  })

  const siteOptions = (me?.sites ?? []).map((s) => ({ value: String(s.id), label: s.name }))
  const isMatrix = type !== 'resource'
  const siteCols = data?.sites ?? []
  const isHistorical = !!date

  return (
    <div>
      <PageHeader
        title="Manpower Distribution"
        subtitle={
          data
            ? `${data.grand_total.toLocaleString()} people · as of ${data.snapshot_date ?? ''}`
            : 'Loading…'
        }
        actions={
          <>
            <DatePicker
              aria-label="As of date"
              value={date}
              onChange={setDate}
              placeholder="As of date"
              maxToday
              className="w-40"
            />
            {isHistorical && (
              <button
                type="button"
                onClick={() => setDate('')}
                className="flex h-10 items-center gap-1.5 rounded-xl border border-black/10 bg-white/60 px-3.5 text-sm font-semibold text-fg transition-colors hover:bg-white"
              >
                <RotateCcw size={15} /> Today
              </button>
            )}
            <Select
              aria-label="Filter by site"
              value={site}
              onChange={setSite}
              options={siteOptions}
              placeholder="All sites"
              className={cn('w-44', isHistorical && 'pointer-events-none opacity-50')}
            />
            <a
              href={distributionExportUrl({ type, site: date ? undefined : site || undefined, date: date || undefined })}
              className="flex h-10 items-center gap-1.5 rounded-xl border border-black/10 bg-white/60 px-4 text-sm font-semibold text-fg transition-colors hover:bg-white"
            >
              <Download size={15} />
              Export
            </a>
          </>
        }
      />

      {/* Historical snapshot banner */}
      {isHistorical && data?.historical && (
        <div className="mb-4 flex items-start gap-2 rounded-xl border border-black/10 bg-black/[0.03] px-4 py-2.5 text-sm text-fg">
          <History size={15} className="mt-0.5 shrink-0 text-fg-muted" />
          <p>
            {data.message ??
              `Showing the saved distribution as of ${data.snapshot_date ?? date} (all sites).`}
          </p>
        </div>
      )}

      {/* Tabs */}
      <div className="mb-4 inline-flex rounded-xl bg-black/[0.04] p-1">
        {TABS.map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setType(t.key)}
            className={cn(
              'rounded-lg px-4 py-1.5 text-sm font-semibold transition-all',
              type === t.key ? 'bg-white text-fg shadow-sm' : 'text-fg-muted hover:text-fg',
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      {isLoading ? (
        <GlassCard className="p-4">
          <SkeletonRows rows={10} />
        </GlassCard>
      ) : !data || data.rows.length === 0 ? (
        <GlassCard>
          <EmptyState message="No distribution data for this selection." />
        </GlassCard>
      ) : (
        <GlassCard className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-black/10 text-xs font-semibold uppercase tracking-wide text-fg-faint">
                  <th className="px-4 py-3 text-left">Trade</th>
                  {isMatrix &&
                    siteCols.map((s) => (
                      <th key={s.id} className="px-3 py-3 text-center">
                        {s.name}
                      </th>
                    ))}
                  {isMatrix && <th className="px-3 py-3 text-center">Leave</th>}
                  <th className="px-4 py-3 text-right">Total</th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((row, i) => (
                  <DistRow
                    key={i}
                    row={row}
                    isMatrix={isMatrix}
                    siteIds={siteCols.map((s) => s.id)}
                  />
                ))}
              </tbody>
              <tfoot>
                <tr className="border-t-2 border-black/15 font-bold text-fg">
                  <td className="px-4 py-3">Grand total</td>
                  {isMatrix && siteCols.map((s) => <td key={s.id} />)}
                  {isMatrix && <td />}
                  <td className="px-4 py-3 text-right">{data.grand_total.toLocaleString()}</td>
                </tr>
              </tfoot>
            </table>
          </div>
        </GlassCard>
      )}
    </div>
  )
}

/** One rendered row — a department header, a trade line, or a subtotal. */
function DistRow({
  row,
  isMatrix,
  siteIds,
}: {
  row: DistributionRow
  isMatrix: boolean
  siteIds: number[]
}) {
  const span = 1 + (isMatrix ? siteIds.length + 1 : 0) + 1

  if (row.kind === 'dept') {
    return (
      <tr className="bg-black/[0.04]">
        <td colSpan={span} className="px-4 py-2 text-xs font-bold uppercase tracking-wide text-fg">
          {row.name}
        </td>
      </tr>
    )
  }

  const isSubtotal = row.kind === 'subtotal'
  const cellCount = (id: number) => (isMatrix ? row.counts?.[String(id)] ?? 0 : 0)

  return (
    <tr
      className={cn(
        'border-b border-black/[0.06] last:border-0',
        isSubtotal ? 'bg-black/[0.02] font-semibold text-fg' : 'text-fg',
      )}
    >
      <td className={cn('px-4 py-2.5', isSubtotal ? '' : 'pl-6')}>
        {isSubtotal ? 'Subtotal' : row.name}
      </td>
      {isMatrix &&
        siteIds.map((id) => (
          <td key={id} className="px-3 py-2.5 text-center tabular-nums">
            {cellCount(id) || <span className="text-fg-faint">·</span>}
          </td>
        ))}
      {isMatrix && (
        <td className="px-3 py-2.5 text-center tabular-nums">
          {row.leave || <span className="text-fg-faint">·</span>}
        </td>
      )}
      <td className="px-4 py-2.5 text-right tabular-nums">
        {isMatrix ? row.total ?? 0 : row.count ?? 0}
      </td>
    </tr>
  )
}
