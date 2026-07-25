import { useMemo, useState, type ReactNode } from 'react'
import { useQuery, keepPreviousData } from '@tanstack/react-query'
import {
  Users, TrendingUp, Clock, Timer, Download, ChevronLeft, ChevronRight,
  ArrowUp, ArrowDown, Trophy, MapPin,
} from 'lucide-react'
import {
  getMonthlyReport, getMonthlyEmployeeCalendar, monthlyExportUrl,
  type MonthlyParams, type MonthlyRow, type MonthlyReport, type LeaderEmployee, type LeaderSite,
} from '../../lib/api/monthly'
import { PageHeader } from '../../components/ui/PageHeader'
import { GlassCard } from '../../components/ui/GlassCard'
import { StatTile } from '../../components/ui/StatTile'
import { EmptyState } from '../../components/ui/EmptyState'
import { SkeletonRows } from '../../components/ui/Skeleton'
import { Select } from '../../components/ui/Field'
import { Pill } from '../../components/ui/Pill'
import { Dialog } from '../../components/ui/Dialog'
import { BarChart, LineChart } from '../../components/ui/Charts'
import { AttendanceCalendar, CalendarLegend } from '../../components/ui/AttendanceCalendar'
import { cn } from '../../lib/utils'

const now = new Date()
const MONTHS = Array.from({ length: 12 }, (_, i) => ({
  value: String(i + 1),
  label: new Date(2000, i, 1).toLocaleString('en', { month: 'long' }),
}))
const YEARS = Array.from({ length: 4 }, (_, i) => {
  const y = now.getFullYear() - i
  return { value: String(y), label: String(y) }
})

/** Attendance % → pill tone band. */
function pctTone(pct: number): 'solid' | 'outline' | 'soft' {
  if (pct >= 85) return 'solid'
  if (pct >= 70) return 'outline'
  return 'soft'
}

/** Minimal row shape the calendar drill-down needs (satisfied by table + leaderboard rows). */
type DrillRow = Pick<MonthlyRow, 'id' | 'name' | 'days_present' | 'working_days'>

type TrendMode = 'present' | 'absent' | 'late' | 'all'
const TREND_TABS: { value: TrendMode; label: string }[] = [
  { value: 'present', label: 'Present' },
  { value: 'absent', label: 'Absent' },
  { value: 'late', label: 'Late' },
  { value: 'all', label: 'All' },
]
/** Monochrome series colours (dark → light) so the three lines stay distinct. */
const TREND_SERIES: Record<TrendMode, { key: string; label: string; color: string }[]> = {
  present: [{ key: 'present', label: 'Present', color: '#14141a' }],
  absent: [{ key: 'absent', label: 'Absent', color: '#a1a1aa' }],
  late: [{ key: 'late', label: 'Late', color: '#71717a' }],
  all: [
    { key: 'present', label: 'Present', color: '#14141a' },
    { key: 'late', label: 'Late', color: '#71717a' },
    { key: 'absent', label: 'Absent', color: '#a1a1aa' },
  ],
}

/**
 * Monthly Report — a month's attendance analytics: summary KPIs with
 * month-over-month deltas, performance bands, a daily trend line, weekday and
 * department bar charts, and a per-employee table. Clicking a row opens that
 * employee's calendar drill-down. Excel export streams from the server.
 */
export default function MonthlyReportPage() {
  const [month, setMonth] = useState(now.getMonth() + 1)
  const [year, setYear] = useState(now.getFullYear())
  const [site, setSite] = useState('all')
  const [page, setPage] = useState(1)
  const [drill, setDrill] = useState<DrillRow | null>(null)
  const [trendMode, setTrendMode] = useState<TrendMode>('present')

  const params: MonthlyParams = { month, year, site, page, per_page: 20 }
  const { data, isLoading, isFetching } = useQuery({
    queryKey: ['monthly', month, year, site, page],
    queryFn: () => getMonthlyReport(params),
    placeholderData: keepPreviousData,
  })

  const s = data?.summary
  const cmp = data?.comparison
  const bands = data?.bands
  const rows = data?.results ?? []
  const pg = data?.pagination

  const siteOptions = useMemo(
    () => [{ value: 'all', label: 'All sites' }, ...(data?.sites ?? []).map((x) => ({ value: String(x.id), label: x.name }))],
    [data],
  )
  const reset = () => setPage(1)

  return (
    <div>
      <PageHeader
        title="Monthly Report"
        subtitle={s ? `${s.month_name} ${s.year}` : 'Loading…'}
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <Select aria-label="Month" value={String(month)} onChange={(v) => { setMonth(Number(v)); reset() }} options={MONTHS} className="w-32" />
            <Select aria-label="Year" value={String(year)} onChange={(v) => { setYear(Number(v)); reset() }} options={YEARS} className="w-24" />
            <Select aria-label="Site" value={site} onChange={(v) => { setSite(v); reset() }} options={siteOptions} className="w-36" />
            <a href={monthlyExportUrl({ month, year, site })}
              className="flex h-10 items-center gap-1.5 rounded-xl bg-[#14141a] px-4 text-sm font-semibold text-white hover:opacity-90">
              <Download size={15} /> Export
            </a>
          </div>
        }
      />

      {/* Summary tiles */}
      <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="Employees" value={s?.total_employees ?? 0} icon={<Users size={18} />} />
        <StatTile label="Avg attendance" value={s ? `${s.avg_attendance_effective}%` : '—'} icon={<TrendingUp size={18} />}
          hint={cmp ? `${cmp.delta_avg_pct >= 0 ? '+' : ''}${cmp.delta_avg_pct}% vs prev` : undefined} />
        <StatTile label="Total late" value={s?.total_late ?? 0} icon={<Clock size={18} />} />
        <StatTile label="Total OT (hrs)" value={s?.total_overtime ?? 0} icon={<Timer size={18} />} />
      </div>

      {/* Bands */}
      {bands && (
        <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {([['Champion', bands.champion, '≥95%'], ['Steady', bands.steady, '85–95%'], ['At risk', bands.at_risk, '70–85%'], ['Critical', bands.critical, '<70%']] as const).map(
            ([label, val, hint]) => (
              <GlassCard key={label} className="p-4">
                <p className="text-xs font-medium text-fg-faint">{label}</p>
                <p className="mt-0.5 text-2xl font-bold text-fg">{val}</p>
                <p className="text-xs text-fg-muted">{hint}</p>
              </GlassCard>
            ),
          )}
        </div>
      )}

      {/* Charts */}
      <div className="mb-5 grid gap-4 lg:grid-cols-2">
        <GlassCard className="p-5 lg:col-span-2">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <h2 className="font-semibold text-fg">Daily attendance trend</h2>
            <div className="inline-flex rounded-xl bg-black/[0.04] p-1">
              {TREND_TABS.map((t) => (
                <button key={t.value} type="button" onClick={() => setTrendMode(t.value)}
                  className={cn('rounded-lg px-3 py-1 text-xs font-semibold transition-all',
                    trendMode === t.value ? 'bg-[#14141a] text-white shadow-sm' : 'text-fg-muted hover:text-fg')}>
                  {t.label}
                </button>
              ))}
            </div>
          </div>
          {isLoading ? <SkeletonRows rows={5} /> : (
            <LineChart data={data?.trend ?? []} xKey="day" series={TREND_SERIES[trendMode]} />
          )}
        </GlassCard>
        <GlassCard className="p-5">
          <h2 className="mb-3 font-semibold text-fg">By weekday (avg present)</h2>
          {isLoading ? <SkeletonRows rows={5} /> : (
            <BarChart data={data?.weekday_split ?? []} xKey="weekday" yKey="avg_present" height={220} />
          )}
        </GlassCard>
        <GlassCard className="p-5">
          <h2 className="mb-3 font-semibold text-fg">By department (avg %)</h2>
          {isLoading ? <SkeletonRows rows={5} /> : (
            <BarChart data={(data?.department_breakdown ?? []).slice(0, 8)} xKey="department" yKey="avg_pct" height={220} color="#52525b" />
          )}
        </GlassCard>
      </div>

      {/* Top & bottom performers */}
      <div className="mb-5">
        <PerformersPanel data={data} isLoading={isLoading} onSelectEmployee={(row) => setDrill(row)} />
      </div>

      {/* Employee table */}
      <GlassCard className="overflow-hidden">
        {isLoading ? (
          <div className="p-4"><SkeletonRows rows={8} /></div>
        ) : rows.length === 0 ? (
          <EmptyState message="No employees for this month." />
        ) : (
          <div className={cn('overflow-x-auto', isFetching && 'opacity-60')}>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-black/10 text-xs font-semibold uppercase tracking-wide text-fg-faint">
                  <th className="px-4 py-3 text-left">Employee</th>
                  <th className="px-4 py-3 text-left">Site</th>
                  <th className="px-4 py-3 text-center">Present</th>
                  <th className="px-4 py-3 text-center">Late</th>
                  <th className="px-4 py-3 text-center">OT</th>
                  <th className="px-4 py-3 text-right">Attendance</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id} onClick={() => setDrill(r)}
                    className="cursor-pointer border-b border-black/[0.06] transition-colors last:border-0 hover:bg-black/[0.02]">
                    <td className="px-4 py-3">
                      <p className="font-semibold text-fg">{r.name}</p>
                      <p className="text-xs text-fg-faint">#{r.badge_number || '—'} · {r.department}</p>
                    </td>
                    <td className="px-4 py-3 text-fg-muted">{r.site}</td>
                    <td className="px-4 py-3 text-center tabular-nums">{r.days_present}/{r.working_days}</td>
                    <td className="px-4 py-3 text-center tabular-nums">{r.late_count}</td>
                    <td className="px-4 py-3 text-center tabular-nums">{r.overtime_hours}</td>
                    <td className="px-4 py-3 text-right">
                      <Pill tone={pctTone(r.attendance_percentage)}>{r.attendance_percentage}%</Pill>
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

      {drill && (
        <CalendarDrillDialog row={drill} month={month} year={year} onClose={() => setDrill(null)} />
      )}
    </div>
  )
}

/** Modal showing one employee's attendance calendar for the selected month. */
function CalendarDrillDialog({ row, month, year, onClose }: { row: DrillRow; month: number; year: number; onClose: () => void }) {
  const { data, isLoading } = useQuery({
    queryKey: ['monthly-calendar', row.id, month, year],
    queryFn: () => getMonthlyEmployeeCalendar(row.id, { month, year }),
  })

  const days = (data?.attendance ?? []).map((a) => ({
    date: a.date,
    status: (a.status || '').toLowerCase(),
    title: a.check_in_time ? new Date(a.check_in_time).toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit' }) : undefined,
  }))

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()} title={row.name}
      description={`${data?.month_name ?? ''} ${year} · ${row.days_present}/${row.working_days} present`} className="max-w-lg">
      {isLoading ? (
        <SkeletonRows rows={6} />
      ) : (
        <div className="space-y-4">
          <AttendanceCalendar year={year} month={month} days={days} />
          <CalendarLegend />
        </div>
      )}
    </Dialog>
  )
}

/** % → text colour band (green/amber/red equivalents in our muted palette). */
function pctColor(pct: number): string {
  if (pct >= 80) return 'text-fg'
  if (pct >= 60) return 'text-fg-muted'
  return 'text-fg-faint'
}

/**
 * Top & Bottom Performers — ranked leaderboards for the month. Toggle between
 * employees (by effective attendance %, bottom excludes zero-attendance) and
 * sites (by average attendance %). Clicking an employee opens their calendar.
 */
function PerformersPanel({
  data,
  isLoading,
  onSelectEmployee,
}: {
  data?: MonthlyReport
  isLoading: boolean
  onSelectEmployee: (row: DrillRow) => void
}) {
  const [mode, setMode] = useState<'employees' | 'sites'>('employees')
  const emp = data?.employee_leaderboard
  const st = data?.site_leaderboard

  return (
    <GlassCard className="p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="grid h-8 w-8 place-items-center rounded-xl bg-black/[0.05] text-fg"><Trophy size={16} /></span>
          <h2 className="font-semibold text-fg">Top &amp; bottom performers</h2>
        </div>
        <div className="inline-flex rounded-xl bg-black/[0.04] p-1">
          {(['employees', 'sites'] as const).map((m) => (
            <button key={m} type="button" onClick={() => setMode(m)}
              className={cn('rounded-lg px-3 py-1 text-xs font-semibold capitalize transition-all',
                mode === m ? 'bg-[#14141a] text-white shadow-sm' : 'text-fg-muted hover:text-fg')}>
              {m}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <SkeletonRows rows={6} />
      ) : (
        <div className="grid gap-5 md:grid-cols-2">
          <LeaderColumn
            title="Top" icon={<ArrowUp size={13} />}
            rows={mode === 'employees' ? emp?.top ?? [] : st?.top ?? []}
            mode={mode} onSelectEmployee={onSelectEmployee}
          />
          <LeaderColumn
            title="Bottom" icon={<ArrowDown size={13} />}
            rows={mode === 'employees' ? emp?.bottom ?? [] : st?.bottom ?? []}
            mode={mode} onSelectEmployee={onSelectEmployee}
          />
        </div>
      )}
    </GlassCard>
  )
}

/** One ranked column (top or bottom) of the performers panel. */
function LeaderColumn({
  title,
  icon,
  rows,
  mode,
  onSelectEmployee,
}: {
  title: string
  icon: ReactNode
  rows: (LeaderEmployee | LeaderSite)[]
  mode: 'employees' | 'sites'
  onSelectEmployee: (row: DrillRow) => void
}) {
  return (
    <div>
      <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-fg-faint">
        {icon} {title} {mode === 'employees' ? 'employees' : 'sites'}
      </p>
      {rows.length === 0 ? (
        <p className="py-4 text-center text-sm text-fg-muted">No data.</p>
      ) : (
        <ol className="space-y-1">
          {rows.map((r, i) =>
            mode === 'employees' ? (
              <EmployeeLeaderRow key={(r as LeaderEmployee).id} rank={i + 1} row={r as LeaderEmployee} onSelect={onSelectEmployee} />
            ) : (
              <SiteLeaderRow key={(r as LeaderSite).site_id} rank={i + 1} row={r as LeaderSite} />
            ),
          )}
        </ol>
      )}
    </div>
  )
}

function RankBadge({ rank }: { rank: number }) {
  return (
    <span className="grid h-6 w-6 shrink-0 place-items-center rounded-lg bg-black/[0.05] text-xs font-bold text-fg-muted tabular-nums">
      {rank}
    </span>
  )
}

function EmployeeLeaderRow({ rank, row, onSelect }: { rank: number; row: LeaderEmployee; onSelect: (r: DrillRow) => void }) {
  return (
    <li>
      <button type="button" onClick={() => onSelect(row)}
        className="flex w-full items-center gap-2.5 rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-black/[0.03]">
        <RankBadge rank={rank} />
        {row.profile_picture ? (
          <img src={row.profile_picture} alt="" className="h-7 w-7 rounded-full object-cover" />
        ) : (
          <span className="grid h-7 w-7 place-items-center rounded-full bg-black/[0.06] text-[10px] font-bold text-fg-muted">
            {row.name.slice(0, 2).toUpperCase()}
          </span>
        )}
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-semibold text-fg">{row.name}</span>
          <span className="block truncate text-xs text-fg-faint">{row.site} · {row.department || '—'}</span>
        </span>
        <span className={cn('shrink-0 text-sm font-bold tabular-nums', pctColor(row.attendance_percentage))}>
          {row.attendance_percentage}%
        </span>
      </button>
    </li>
  )
}

function SiteLeaderRow({ rank, row }: { rank: number; row: LeaderSite }) {
  return (
    <li className="flex items-center gap-2.5 rounded-lg px-2 py-1.5">
      <RankBadge rank={rank} />
      <span className="grid h-7 w-7 place-items-center rounded-full bg-black/[0.06] text-fg-muted"><MapPin size={14} /></span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-semibold text-fg">{row.site}</span>
        <span className="block truncate text-xs text-fg-faint">{row.employee_count} employees</span>
      </span>
      <span className={cn('shrink-0 text-sm font-bold tabular-nums', pctColor(row.avg_pct))}>{row.avg_pct}%</span>
    </li>
  )
}
