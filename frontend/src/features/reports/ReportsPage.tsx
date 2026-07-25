import { useMemo, useState } from 'react'
import { useQuery, keepPreviousData } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  UserCheck, UserX, Clock, LogOut, MapPin, Download, ChevronLeft, ChevronRight, Sparkles, Radio,
} from 'lucide-react'
import { getReportData, getReportSummary, reportExportUrl, type ReportParams } from '../../lib/api/reports'
import { PageHeader } from '../../components/ui/PageHeader'
import { GlassCard } from '../../components/ui/GlassCard'
import { StatTile } from '../../components/ui/StatTile'
import { EmptyState } from '../../components/ui/EmptyState'
import { SkeletonRows } from '../../components/ui/Skeleton'
import { Select, DatePicker } from '../../components/ui/Field'
import { Pill } from '../../components/ui/Pill'
import { Dialog } from '../../components/ui/Dialog'
import { GeofenceMap } from '../../components/ui/GeofenceMap'
import { BarChart, DoughnutChart, ChartLegend } from '../../components/ui/Charts'
import { cn } from '../../lib/utils'

const todayISO = () => new Date().toISOString().slice(0, 10)

const QUICK = [
  { value: '', label: 'All' },
  { value: 'late', label: 'Late' },
  { value: 'missing_checkout', label: 'Missing checkout' },
  { value: 'geofence', label: 'Geofence' },
  { value: 'now', label: 'Checking in now' },
]

/** Present/Absent status tabs — filter rows by whether the employee checked in. */
const STATUS_TABS = [
  { value: '', label: 'All' },
  { value: 'Present', label: 'Present' },
  { value: 'Absent', label: 'Absent' },
]

/** Auto-refresh interval while live mode is on (matches the legacy 30s). */
const LIVE_MS = 30_000

/**
 * Daily Reports — one date's attendance picture: KPI stats with day-over-day
 * comparison, an hourly check-in histogram, a department breakdown doughnut, an
 * executive summary, and the attendance table with quick filters + a geofence
 * map. Excel export streams from the server.
 */
export default function ReportsPage() {
  const [date, setDate] = useState(todayISO())
  const [site, setSite] = useState('all')
  const [quick, setQuick] = useState('')
  const [status, setStatus] = useState('') // '', 'Present', 'Absent'
  const [live, setLive] = useState(false)
  const [page, setPage] = useState(1)
  const [mapOpen, setMapOpen] = useState(false)
  const navigate = useNavigate()

  const params: ReportParams = { date, site, quick: quick || undefined, status: status || undefined, page, per_page: 20 }
  const { data, isLoading, isFetching, dataUpdatedAt } = useQuery({
    queryKey: ['report-data', date, site, quick, status, page],
    queryFn: () => getReportData(params),
    placeholderData: keepPreviousData,
    // Live mode silently re-polls the current view every 30s (keepPreviousData
    // avoids skeleton flicker, so refreshes are seamless).
    refetchInterval: live ? LIVE_MS : false,
  })
  const { data: summary } = useQuery({
    queryKey: ['report-summary', date, site],
    queryFn: () => getReportSummary({ date, site }),
  })

  const stats = data?.stats
  const cmp = data?.comparison
  const hourly = data?.hourly_histogram ?? []
  const deptRows = (data?.dept_rows ?? []).slice(0, 6)
  const geoList = data?.geofence_violation_list ?? []
  const rows = data?.results ?? []
  const pg = data?.pagination

  const siteOptions = useMemo(
    () => [{ value: 'all', label: 'All sites' }, ...(data?.sites ?? []).map((s) => ({ value: String(s.id), label: s.name }))],
    [data],
  )

  const reset = () => setPage(1)

  return (
    <div>
      <PageHeader
        title="Reports"
        subtitle={data ? `Attendance for ${data.selected_date}` : 'Loading…'}
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" onClick={() => setLive((v) => !v)}
              className={cn('flex h-10 items-center gap-1.5 rounded-xl px-3.5 text-sm font-semibold transition-all',
                live ? 'bg-[#14141a] text-white' : 'border border-black/10 bg-white/60 text-fg hover:bg-white')}>
              <span className="relative flex h-2 w-2">
                {live && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-white/70" />}
                <span className={cn('relative inline-flex h-2 w-2 rounded-full', live ? 'bg-white' : 'bg-fg-faint')} />
              </span>
              {live ? 'Live ON · 30s' : 'Live OFF'}
            </button>
            <DatePicker aria-label="Date" value={date} onChange={(v) => { setDate(v); reset() }} className="w-40" />
            <Select aria-label="Site" value={site} onChange={(v) => { setSite(v); reset() }} options={siteOptions} className="w-40" />
            <a href={reportExportUrl({ date, site })}
              className="flex h-10 items-center gap-1.5 rounded-xl bg-[#14141a] px-4 text-sm font-semibold text-white hover:opacity-90">
              <Download size={15} /> Export
            </a>
          </div>
        }
      />

      {live && dataUpdatedAt > 0 && (
        <p className="-mt-2 mb-4 flex items-center gap-1.5 text-xs text-fg-muted">
          <Radio size={12} /> Synced {new Date(dataUpdatedAt).toLocaleTimeString()}
        </p>
      )}

      {/* KPI tiles */}
      <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-5">
        <StatTile label="Present" value={stats?.present ?? 0} icon={<UserCheck size={18} />}
          hint={cmp ? `${cmp.delta_yesterday >= 0 ? '+' : ''}${cmp.delta_yesterday} vs yest.` : undefined} />
        <StatTile label="Absent" value={stats?.absent ?? 0} icon={<UserX size={18} />} />
        <StatTile label="Late" value={stats?.late ?? 0} icon={<Clock size={18} />} />
        <StatTile label="Missing out" value={stats?.missing_checkout ?? 0} icon={<LogOut size={18} />} />
        <StatTile label="Geofence" value={stats?.geofence_violations ?? 0} icon={<MapPin size={18} />}
          hint={geoList.length ? 'view on map' : undefined}
          onClick={geoList.length ? () => setMapOpen(true) : undefined} />
      </div>

      {/* Charts */}
      <div className="mb-5 grid gap-4 lg:grid-cols-3">
        <GlassCard className="p-5 lg:col-span-2">
          <h2 className="mb-3 font-semibold text-fg">Check-ins by hour</h2>
          {isLoading ? <SkeletonRows rows={6} /> : (
            <BarChart data={hourly.map((h) => ({ hour: `${h.hour}:00`, count: h.count }))} xKey="hour" yKey="count" />
          )}
        </GlassCard>
        <GlassCard className="p-5">
          <h2 className="mb-3 font-semibold text-fg">By department</h2>
          {isLoading ? <SkeletonRows rows={6} /> : deptRows.length === 0 ? (
            <p className="py-8 text-center text-sm text-fg-muted">No data.</p>
          ) : (
            <div className="grid grid-cols-[1fr_auto] items-center gap-3">
              <DoughnutChart data={deptRows} nameKey="department" valueKey="present" height={180} />
              <ChartLegend items={deptRows.map((d) => ({ label: d.department, value: `${d.pct}%` }))} />
            </div>
          )}
        </GlassCard>
      </div>

      {/* Exec summary */}
      {summary && summary.narrative.length > 0 && (
        <GlassCard className="mb-5 p-5">
          <div className="mb-3 flex items-center gap-2">
            <span className="grid h-8 w-8 place-items-center rounded-xl bg-black/[0.05] text-fg"><Sparkles size={16} /></span>
            <h2 className="font-semibold text-fg">Executive summary</h2>
          </div>
          <ul className="grid gap-2 sm:grid-cols-2">
            {summary.narrative.map((b, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-fg">
                {/* This endpoint returns Remix-icon class names (not emoji), so
                    we render a neutral bullet rather than the raw class string. */}
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-black/30" />
                <span>{b.text}</span>
              </li>
            ))}
          </ul>
        </GlassCard>
      )}

      {/* Status tabs (present/absent) + quick filters — combine (AND) */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="inline-flex rounded-xl bg-black/[0.04] p-1">
          {STATUS_TABS.map((s) => (
            <button key={s.value} type="button" onClick={() => { setStatus(s.value); reset() }}
              className={cn('rounded-lg px-3.5 py-1.5 text-sm font-semibold transition-all',
                status === s.value ? 'bg-[#14141a] text-white shadow-sm' : 'text-fg-muted hover:text-fg')}>
              {s.label}
            </button>
          ))}
        </div>
        <div className="inline-flex flex-wrap rounded-xl bg-black/[0.04] p-1">
          {QUICK.map((q) => (
            <button key={q.value} type="button" onClick={() => { setQuick(q.value); reset() }}
              className={cn('rounded-lg px-3.5 py-1.5 text-sm font-semibold transition-all',
                quick === q.value ? 'bg-white text-fg shadow-sm' : 'text-fg-muted hover:text-fg')}>
              {q.label}
            </button>
          ))}
        </div>
      </div>

      <GlassCard className="overflow-hidden">
        {isLoading ? (
          <div className="p-4"><SkeletonRows rows={8} /></div>
        ) : rows.length === 0 ? (
          <EmptyState message="No attendance rows for this filter." />
        ) : (
          <div className={cn('overflow-x-auto', isFetching && 'opacity-60')}>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-black/10 text-xs font-semibold uppercase tracking-wide text-fg-faint">
                  <th className="px-4 py-3 text-left">Employee</th>
                  <th className="px-4 py-3 text-left">Site</th>
                  <th className="px-4 py-3 text-left">In</th>
                  <th className="px-4 py-3 text-left">Out</th>
                  <th className="px-4 py-3 text-left">Late</th>
                  <th className="px-4 py-3 text-left">Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id} onClick={() => navigate(`/dashboard/user/${r.id}-v2`)}
                    className="cursor-pointer border-b border-black/[0.06] transition-colors last:border-0 hover:bg-black/[0.02]">
                    <td className="px-4 py-3">
                      <p className="font-semibold text-fg">{r.name}</p>
                      <p className="text-xs text-fg-faint">#{r.badge_number || '—'} · {r.department || '—'}</p>
                    </td>
                    <td className="px-4 py-3 text-fg-muted">{r.site}</td>
                    <td className="px-4 py-3 text-fg-muted">{r.check_in}</td>
                    <td className="px-4 py-3 text-fg-muted">{r.check_out}</td>
                    <td className="px-4 py-3">{r.late_minutes > 0 ? <Pill tone="outline">{r.late_minutes}m</Pill> : <span className="text-fg-faint">—</span>}</td>
                    <td className="px-4 py-3">
                      <Pill tone={r.status === 'Present' ? 'solid' : 'soft'}>{r.status}</Pill>
                      {r.is_geofence_violation && <Pill tone="outline" className="ml-1">geo</Pill>}
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

      <Dialog open={mapOpen} onOpenChange={setMapOpen} title="Geofence violations"
        description={`${geoList.length} out-of-bounds check-in${geoList.length === 1 ? '' : 's'}`} className="max-w-2xl">
        <GeofenceMap
          markers={geoList.filter((g) => g.latitude != null && g.longitude != null)
            .map((g) => ({ lat: g.latitude as number, lng: g.longitude as number }))}
          height={440}
        />
      </Dialog>
    </div>
  )
}
