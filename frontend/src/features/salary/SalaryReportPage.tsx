import { useMemo, useState } from 'react'
import { useQuery, keepPreviousData } from '@tanstack/react-query'
import { Eye, FileDown, Printer, ChevronLeft, ChevronRight, MapPin } from 'lucide-react'
import { getSalaryReport, salarySlipUrl, type SalaryParams, type SalaryRow } from '../../lib/api/salary'
import { useMe } from '../../lib/hooks/useMe'
import { PageHeader } from '../../components/ui/PageHeader'
import { GlassCard } from '../../components/ui/GlassCard'
import { EmptyState } from '../../components/ui/EmptyState'
import { SkeletonRows } from '../../components/ui/Skeleton'
import { Select, SearchInput } from '../../components/ui/Field'
import { Dialog } from '../../components/ui/Dialog'
import { Pill } from '../../components/ui/Pill'
import { useDebounce } from '../../lib/hooks/useDebounce'
import { cn } from '../../lib/utils'

const now = new Date()
const MONTHS = Array.from({ length: 12 }, (_, i) => ({
  value: String(i + 1),
  label: new Date(2000, i, 1).toLocaleString('en', { month: 'long' }),
}))
const YEARS = Array.from({ length: 4 }, (_, i) => ({ value: String(now.getFullYear() - i), label: String(now.getFullYear() - i) }))

/** Format a numeric string/number with the currency code (no decimals — table). */
function money(v: string | number, code: string) {
  const n = typeof v === 'string' ? Number(v) : v
  return `${code} ${(Number.isFinite(n) ? n : 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
}
/** Two-decimal amount (slip breakdown). */
function amount(v: string | number) {
  const n = typeof v === 'string' ? Number(v) : v
  return (Number.isFinite(n) ? n : 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

/**
 * Salary Report (superuser) — computed monthly salary per employee: basic pay,
 * overtime (normal/special), absence deduction and net. Each row's **View**
 * opens a salary-slip modal with the full breakdown, printable and downloadable
 * as PDF. Month, year and site scope the report; only paid employees appear.
 */
export default function SalaryReportPage() {
  const { data: me } = useMe()
  const code = me?.currency_code || 'AED'
  const [month, setMonth] = useState(now.getMonth() + 1)
  const [year, setYear] = useState(now.getFullYear())
  const [site, setSite] = useState('all')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [slip, setSlip] = useState<SalaryRow | null>(null)
  const debounced = useDebounce(search, 300)

  const params: SalaryParams = { month, year, site, search: debounced || undefined, page, per_page: 25 }
  const { data, isLoading, isFetching } = useQuery({
    queryKey: ['salary', month, year, site, debounced, page],
    queryFn: () => getSalaryReport(params),
    placeholderData: keepPreviousData,
  })

  const rows = data?.results ?? []
  const pg = data?.pagination
  const monthName = data?.summary.month_name ?? MONTHS[month - 1].label
  const siteOptions = useMemo(
    () => [{ value: 'all', label: 'All sites' }, ...(data?.sites ?? []).map((s) => ({ value: String(s.id), label: s.name }))],
    [data],
  )
  const reset = () => setPage(1)

  return (
    <div>
      <PageHeader
        title="Salary Report"
        subtitle={data ? `${data.summary.month_name} ${data.summary.year} · ${data.summary.total_employees} employees` : 'Loading…'}
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <Select aria-label="Month" value={String(month)} onChange={(v) => { setMonth(Number(v)); reset() }} options={MONTHS} className="w-32" />
            <Select aria-label="Year" value={String(year)} onChange={(v) => { setYear(Number(v)); reset() }} options={YEARS} className="w-24" />
            <Select aria-label="Site" value={site} onChange={(v) => { setSite(v); reset() }} options={siteOptions} className="w-36" />
            <SearchInput value={search} onChange={(v) => { setSearch(v); reset() }} placeholder="Name or badge…" className="w-48" />
          </div>
        }
      />

      <GlassCard className="overflow-hidden">
        {isLoading ? (
          <div className="p-4"><SkeletonRows rows={10} /></div>
        ) : rows.length === 0 ? (
          <EmptyState message="No employees with a salary for this selection." />
        ) : (
          <div className={cn('overflow-x-auto', isFetching && 'opacity-60')}>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-black/10 text-xs font-semibold uppercase tracking-wide text-fg-faint">
                  <th className="px-4 py-3 text-left">Employee</th>
                  <th className="px-4 py-3 text-left">Site</th>
                  <th className="px-4 py-3 text-right">Basic salary</th>
                  <th className="px-4 py-3 text-center">Days (W/P/A)</th>
                  <th className="px-4 py-3 text-center">OT (N/S)</th>
                  <th className="px-4 py-3 text-right">Bonus / deduct</th>
                  <th className="px-4 py-3 text-right">Net salary</th>
                  <th className="px-4 py-3 text-center">Actions</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id} className="border-b border-black/[0.06] last:border-0 hover:bg-black/[0.02]">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2.5">
                        {r.profile_picture ? (
                          <img src={r.profile_picture} alt="" className="h-8 w-8 rounded-full object-cover" />
                        ) : (
                          <span className="grid h-8 w-8 place-items-center rounded-full bg-black/[0.06] text-[11px] font-bold text-fg-muted">
                            {r.name.slice(0, 2).toUpperCase()}
                          </span>
                        )}
                        <div>
                          <p className="font-semibold text-fg">{r.name}</p>
                          <p className="text-xs text-fg-faint">#{r.badge_number}</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-fg-muted">
                      <span className="inline-flex items-center gap-1"><MapPin size={13} className="text-fg-faint" />{r.site}</span>
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums">{money(r.basic_salary, code)}</td>
                    <td className="px-4 py-3">
                      <div className="flex justify-center gap-1">
                        <Pill tone="soft">{r.working_days}W</Pill>
                        <Pill tone="outline">{r.present_days}P</Pill>
                        <Pill tone="soft">{r.absent_days}A</Pill>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex justify-center gap-1 text-xs text-fg-muted tabular-nums">
                        <span>{r.normal_ot_hours}h N</span>
                        <span className="text-fg-faint">·</span>
                        <span>{r.special_ot_hours}h S</span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right tabular-nums">
                      <p className="font-medium text-fg">+{money(r.normal_ot_pay + r.special_ot_pay, code)}</p>
                      <p className="text-xs text-fg-muted">−{money(r.deduction, code)}</p>
                    </td>
                    <td className="px-4 py-3 text-right font-semibold tabular-nums">{money(r.net_salary, code)}</td>
                    <td className="px-4 py-3 text-center">
                      <button type="button" onClick={() => setSlip(r)}
                        className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-black/10 px-3 text-xs font-semibold text-fg hover:bg-black/[0.04]">
                        <Eye size={14} /> View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {pg && pg.num_pages > 1 && (
          <div className="flex items-center justify-between border-t border-black/10 px-4 py-3 text-sm text-fg-muted">
            <span>{pg.start_index}–{pg.end_index} of {pg.total_items.toLocaleString()}</span>
            <div className="flex items-center gap-1">
              <button type="button" onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={!pg.has_previous}
                className="grid h-8 w-8 place-items-center rounded-lg hover:bg-black/[0.05] hover:text-fg disabled:opacity-40"><ChevronLeft size={16} /></button>
              <span className="px-2 text-xs">Page {pg.current_page} / {pg.num_pages}</span>
              <button type="button" onClick={() => setPage((p) => p + 1)} disabled={!pg.has_next}
                className="grid h-8 w-8 place-items-center rounded-lg hover:bg-black/[0.05] hover:text-fg disabled:opacity-40"><ChevronRight size={16} /></button>
            </div>
          </div>
        )}
      </GlassCard>

      {slip && (
        <SalarySlipDialog row={slip} month={month} year={year} monthName={monthName} code={code} onClose={() => setSlip(null)} />
      )}
    </div>
  )
}

/**
 * Salary-slip modal — the full monthly breakdown for one employee, rendered from
 * the row data (same as the legacy popup). Printable, and downloadable as the
 * server-generated PDF.
 */
function SalarySlipDialog({
  row, month, year, monthName, code, onClose,
}: {
  row: SalaryRow
  month: number
  year: number
  monthName: string
  code: string
  onClose: () => void
}) {
  const otTotal = row.normal_ot_pay + row.special_ot_pay

  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title="Salary slip"
      description={`${monthName} ${year}`}
      className="max-w-lg"
      footer={
        <>
          <button type="button" onClick={onClose}
            className="h-10 rounded-xl border border-black/10 px-4 text-sm font-semibold text-fg hover:bg-black/[0.04]">
            Close
          </button>
          <button type="button" onClick={() => window.print()}
            className="flex h-10 items-center gap-1.5 rounded-xl border border-black/10 px-4 text-sm font-semibold text-fg hover:bg-black/[0.04]">
            <Printer size={15} /> Print
          </button>
          <a href={salarySlipUrl(row.id, month, year)}
            className="flex h-10 items-center gap-1.5 rounded-xl bg-[#14141a] px-4 text-sm font-semibold text-white hover:opacity-90">
            <FileDown size={15} /> Download PDF
          </a>
        </>
      }
    >
      <div className="space-y-4">
        {/* Employee header */}
        <div className="flex items-center gap-3 rounded-xl bg-black/[0.03] p-3">
          {row.profile_picture ? (
            <img src={row.profile_picture} alt="" className="h-11 w-11 rounded-full object-cover" />
          ) : (
            <span className="grid h-11 w-11 place-items-center rounded-full bg-black/[0.08] text-sm font-bold text-fg-muted">
              {row.name.slice(0, 2).toUpperCase()}
            </span>
          )}
          <div className="min-w-0">
            <p className="truncate font-semibold text-fg">{row.name}</p>
            <p className="text-xs text-fg-muted">#{row.badge_number} · {row.department} · {row.site}</p>
          </div>
        </div>

        {/* Attendance summary */}
        <div className="grid grid-cols-3 gap-2 text-center">
          <SlipStat label="Working" value={row.working_days} />
          <SlipStat label="Present" value={row.present_days} />
          <SlipStat label="Absent" value={row.absent_days} />
        </div>

        {/* Breakdown */}
        <div className="overflow-hidden rounded-xl border border-black/10">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-black/10 bg-black/[0.02] text-xs font-semibold uppercase tracking-wide text-fg-faint">
                <th className="px-3 py-2 text-left">Description</th>
                <th className="px-3 py-2 text-right">Amount ({code})</th>
              </tr>
            </thead>
            <tbody>
              <SlipRow label="Basic salary" value={amount(row.basic_salary)} />
              <SlipRow label={`Normal overtime (${row.normal_ot_hours} hrs @ 1.25×)`} value={`+${amount(row.normal_ot_pay)}`} />
              <SlipRow label={`Special overtime (${row.special_ot_hours} hrs @ 1.50×)`} value={`+${amount(row.special_ot_pay)}`} />
              <SlipRow label={`Absence deduction (${row.absent_days} days)`} value={`−${amount(row.deduction)}`} muted />
              <tr className="border-t-2 border-black/15 font-bold text-fg">
                <td className="px-3 py-2.5">Net payable</td>
                <td className="px-3 py-2.5 text-right tabular-nums">{code} {amount(row.net_salary)}</td>
              </tr>
            </tbody>
          </table>
        </div>

        {otTotal === 0 && row.deduction === 0 && (
          <p className="text-center text-xs text-fg-faint">No overtime or deductions this month.</p>
        )}
        <p className="text-center text-[11px] text-fg-faint">This is a computer-generated salary slip.</p>
      </div>
    </Dialog>
  )
}

function SlipStat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-xl bg-black/[0.03] py-2.5">
      <p className="text-lg font-bold leading-none text-fg tabular-nums">{value}</p>
      <p className="mt-1 text-xs text-fg-faint">{label}</p>
    </div>
  )
}

function SlipRow({ label, value, muted }: { label: string; value: string; muted?: boolean }) {
  return (
    <tr className="border-b border-black/[0.06]">
      <td className="px-3 py-2 text-fg-muted">{label}</td>
      <td className={cn('px-3 py-2 text-right tabular-nums', muted ? 'text-fg-muted' : 'text-fg')}>{value}</td>
    </tr>
  )
}
